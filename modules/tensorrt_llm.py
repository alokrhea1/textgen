from pathlib import Path
from threading import Event, Lock, Thread

from modules import shared
from modules.logging_colors import logger


class TensorRTLLMModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path_to_model):
        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)
        result = cls()
        runtime_python = getattr(shared.args, 'tensorrt_llm_python', None)
        user_data = getattr(shared, 'user_data_dir', Path(__file__).resolve().parents[1] / 'user_data')
        default_runtime = Path(user_data) / 'tensorrt_runtime/bin/python'
        if runtime_python or default_runtime.is_file():
            from transformers import AutoTokenizer
            from modules.tensorrt_proxy import TensorRTWorker

            result.tokenizer = AutoTokenizer.from_pretrained(
                str(path_to_model), trust_remote_code=getattr(shared.args, 'trust_remote_code', False))
            result.llm = TensorRTWorker(runtime_python or default_runtime, path_to_model,
                                       getattr(shared.args, 'ctx_size', 0) or 8192,
                                       trust_remote_code=getattr(shared.args, 'trust_remote_code', False))
            result.sampling_params_type = dict
        else:
            try:
                from tensorrt_llm._tensorrt_engine import LLM
                from tensorrt_llm.llmapi import SamplingParams
            except ImportError as error:
                raise ModuleNotFoundError(
                    'TensorRT-LLM runtime is unavailable. Run python scripts/setup_tensorrt.py '
                    'to install its isolated CUDA 12.8 environment, or specify '
                    '--tensorrt-llm-python /path/to/runtime/bin/python.'
                ) from error
            result.llm = LLM(model=str(path_to_model), skip_tokenizer_init=False,
                             trust_remote_code=getattr(shared.args, 'trust_remote_code', False),
                             tensor_parallel_size=1, max_seq_len=getattr(shared.args, 'ctx_size', 0) or 8192)
            result.tokenizer = result.llm.tokenizer
            result.sampling_params_type = SamplingParams
        return result

    def generate_with_streaming(self, prompt, state):
        prompt_ids = self.encode_prompt(prompt, add_bos_token=state['add_bos_token'])
        self.last_prompt_token_count = len(prompt_ids)
        self.last_completion_token_count = 0

        if state.get('_rewrite_generation_guard'):
            from modules.text_generation import get_max_prompt_length

            budget = get_max_prompt_length(state)
            if budget <= 0 or self.last_prompt_token_count > budget:
                raise ValueError(
                    f'Sentence rewrite prompt requires {self.last_prompt_token_count} tokens; '
                    f'only {budget} prompt tokens are available. References will not be '
                    'silently truncated. Reduce K/window size or increase the context limit.'
                )

        stop_event = state.get('stop_event')
        if shared.stop_everything or (stop_event and stop_event.is_set()):
            return

        sampling_params_type = getattr(self, 'sampling_params_type', None)
        if sampling_params_type is None:
            from tensorrt_llm.llmapi import SamplingParams
            sampling_params_type = SamplingParams
        sampling_params = sampling_params_type(
            max_tokens=state['max_new_tokens'] if not state['auto_max_new_tokens']
                       else state['truncation_length'] - self.last_prompt_token_count,
            end_id=shared.tokenizer.eos_token_id,
            temperature=state['temperature'],
            top_k=state['top_k'],
            top_p=state['top_p'],
            min_p=state['min_p'],
            repetition_penalty=state['repetition_penalty'],
            presence_penalty=state['presence_penalty'],
            frequency_penalty=state['frequency_penalty'],
            no_repeat_ngram_size=state['no_repeat_ngram_size'] if state['no_repeat_ngram_size'] > 0 else None,
            seed=state['seed'],
            ignore_eos=state['ban_eos_token'],
            add_special_tokens=state['add_bos_token'],
            skip_special_tokens=state['skip_special_tokens'],
        )

        # Submit the IDs we counted, avoiding a second, potentially different encoding.
        result = self.llm.generate_async(prompt_ids, sampling_params=sampling_params, streaming=True)
        finished = Event()
        abort_lock = Lock()
        aborted = False

        def abort():
            nonlocal aborted
            with abort_lock:
                if not aborted:
                    result.abort()
                    aborted = True

        def watch_stop():
            # The iterator may block waiting for the first token. Abort the native
            # request directly so a local Stop can also interrupt that wait.
            while not finished.wait(0.05):
                if shared.stop_everything or (stop_event and stop_event.is_set()):
                    try:
                        abort()
                    except Exception:
                        logger.exception('Failed to abort TensorRT-LLM generation')
                    return

        watcher = Thread(target=watch_stop, daemon=True)
        watcher.start()

        cumulative_reply = ''
        try:
            for output in result:
                if shared.stop_everything or (stop_event and stop_event.is_set()):
                    break

                self.last_completion_token_count = len(output.outputs[0].token_ids)
                text_diff = output.outputs[0].text_diff
                if text_diff:
                    cumulative_reply += text_diff
                    yield cumulative_reply
        finally:
            finished.set()
            try:
                abort()
            finally:
                watcher.join(timeout=0.2)
                if hasattr(result, 'close'):
                    result.close()

    def encode_prompt(self, prompt, add_bos_token=True):
        tokenizer = getattr(self, 'tokenizer', shared.tokenizer)
        return list(tokenizer.encode(str(prompt), add_special_tokens=add_bos_token))

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def unload(self):
        if hasattr(self, 'llm') and self.llm is not None:
            self.llm.shutdown()
            self.llm = None

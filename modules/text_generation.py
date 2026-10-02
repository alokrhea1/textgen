import ast
import copy
import html
import pprint
import random
import time

import numpy as np

import modules.shared as shared
from modules import models
from modules.callbacks import Iteratorize
from modules.extensions import apply_extensions
from modules.html_generator import generate_basic_html
from modules.logging_colors import logger
from modules.utils import check_model_loaded


def generate_reply(*args, **kwargs):
    models.load_model_if_idle_unloaded()

    state = args[1] if len(args) > 1 else kwargs.get('state', {})
    use_parallel = (
        state.get('stop_event') is not None
        and shared.model.__class__.__name__ in ['Exllamav3Model', 'LlamaServer', 'TensorRTLLMModel']
        and (shared.model.__class__.__name__ != 'LlamaServer' or shared.args.parallel > 1)
    )

    stop_event = state.get('stop_event')
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError('Generation was interrupted before starting.')
    if not use_parallel:
        if stop_event is None:
            shared.generation_lock.acquire()
        else:
            while not shared.generation_lock.acquire(timeout=0.1):
                if stop_event.is_set():
                    raise InterruptedError('Generation was interrupted while waiting for the model.')
            if stop_event.is_set():
                shared.generation_lock.release()
                raise InterruptedError('Generation was interrupted before starting.')

    with models._generation_count_lock:
        models.active_generation_count += 1

    try:
        yield from _generate_reply(*args, **kwargs)
    finally:
        with models._generation_count_lock:
            models.active_generation_count -= 1

        models.last_generation_time = time.time()
        if not use_parallel:
            shared.generation_lock.release()


def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False, for_ui=False):
    rewrite_guard = state.get('_rewrite_generation_guard', False)
    rewrite_stop = state.get('_rewrite_stop_predicate') if rewrite_guard else None
    rewrite_event = state.get('stop_event') if rewrite_guard else None
    rewrite_kind = state.get('_rewrite_prompt_kind') if rewrite_guard else None
    automatic_prefill = state.get('_notebook_auto_prefill_suffix') if rewrite_guard else None
    automatic = state.get('_notebook_auto_generation', False)
    automatic_stop = state.get('_notebook_auto_stop_predicate') if automatic else None
    automatic_event = state.get('stop_event') if automatic else None
    automatic_info = state.get('_notebook_auto_generation_info') if automatic else None
    # Find the appropriate generation function
    generate_func = apply_extensions('custom_generate_reply')
    if generate_func is None:
        model_is_loaded, error_message = check_model_loaded()
        if not model_is_loaded:
            yield ''
            return

        if shared.model.__class__.__name__ in ['LlamaServer', 'Exllamav3Model', 'TensorRTLLMModel']:
            generate_func = generate_reply_custom
        else:
            generate_func = generate_reply_HF

    if generate_func != generate_reply_HF and shared.args.verbose:
        logger.info("PROMPT=")
        print_prompt(question)

    # Prepare the input
    original_question = question
    if not is_chat:
        state = apply_extensions('state', state)
        question = apply_extensions('input', question, state)
    if automatic:
        # Drafts retain ordinary Notebook context truncation, while operation
        # cancellation, sentence stopping, and errors must survive state hooks.
        state = dict(state)
        state['_notebook_auto_generation'] = True
        state['_notebook_auto_stop_predicate'] = automatic_stop
        state['_notebook_auto_generation_info'] = automatic_info
        if automatic_event is not None:
            state['stop_event'] = automatic_event
        state['skip_special_tokens'] = False
        state['auto_max_new_tokens'] = False
        state['stream'] = True
    if rewrite_guard:
        state = dict(state)
        state['_rewrite_generation_guard'] = True
        state['_rewrite_stop_predicate'] = rewrite_stop
        if rewrite_kind is not None:
            state['_rewrite_prompt_kind'] = rewrite_kind
        if automatic_prefill is not None:
            state['_notebook_auto_prefill_suffix'] = automatic_prefill
        if rewrite_event is not None:
            state['stop_event'] = rewrite_event
        state['skip_special_tokens'] = False
        state['auto_max_new_tokens'] = False
        state['stream'] = True
        validate_rewrite_prompt(question, state)

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, state['custom_stopping_strings']):
        if type(st) is str:
            st = ast.literal_eval(f"[{st}]")

        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    shared.stop_everything = False
    reply = ''
    is_stream = state['stream']
    if len(all_stop_strings) > 0 and not state['stream']:
        original_logits_processor = state.get('logits_processor')
        stop_event_ref = state.pop('stop_event', None)
        state = copy.deepcopy(state)
        if stop_event_ref is not None:
            state['stop_event'] = stop_event_ref
        if original_logits_processor is not None:
            state['logits_processor'] = original_logits_processor
        state['stream'] = True

    # Generate
    last_update = -1
    latency_threshold = 1 / 1000
    generated = generate_func(question, original_question, state, stopping_strings, is_chat=is_chat)
    try:
        for reply in generated:
            cur_time = time.monotonic()
            if automatic_info is not None:
                automatic_info['raw_reply'] = reply
            reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
            if escape_html:
                reply = html.escape(reply)
            if automatic:
                terminal = stop_found or shared.stop_everything or (
                    automatic_event is not None and automatic_event.is_set())
                if terminal and automatic_info is not None:
                    automatic_info['terminal'] = True
                if not terminal and automatic_stop is not None and automatic_stop(reply):
                    if automatic_info is not None:
                        automatic_info['boundary'] = True
                    break
            if rewrite_stop is not None and rewrite_stop(reply):
                break

            if is_stream:
                # Limit number of tokens/second to make text readable in real time
                if state['max_tokens_second'] > 0:
                    diff = 1 / state['max_tokens_second'] - (cur_time - last_update)
                    if diff > 0:
                        time.sleep(diff)

                    last_update = time.monotonic()
                    yield reply

                # Limit updates to avoid lag in the Gradio UI
                # API updates are not limited
                else:
                    # If 'generate_func' takes less than 0.001 seconds to yield the next token
                    # (equivalent to more than 1000 tok/s), assume that the UI is lagging behind and skip yielding
                    if (cur_time - last_update) > latency_threshold:
                        yield reply
                    last_update = time.monotonic()

            stop_event = state.get('stop_event')
            if stop_found or shared.stop_everything or (stop_event and stop_event.is_set()):
                break
    finally:
        close = getattr(generated, 'close', None)
        if close is not None:
            close()

    if not is_chat:
        reply = apply_extensions('output', reply, state)

    yield reply


def validate_rewrite_prompt(question, state, input_ids=None, inputs_embeds=None, original_budget=None):
    """Opt-in no-truncation guard after extension transformations."""
    prefill = state.get('_notebook_auto_prefill_suffix')
    if prefill and not question.endswith(prefill):
        raise ValueError('Automatic continuation prefill was changed by an extension. The prompt must end with the exact unfinished sentence prefix; it will not be silently repaired.')
    budget = get_max_prompt_length(state)
    if original_budget is not None:
        budget = min(budget, original_budget)
    if input_ids is None and inputs_embeds is None:
        count = get_rewrite_prompt_length(question, state, use_extensions=False)
    else:
        lengths = []
        if input_ids is not None:
            lengths.append(int(input_ids.shape[-1]))
        if inputs_embeds is not None:
            lengths.append(int(inputs_embeds.shape[-2]))
        count = max(lengths)
    if budget <= 0 or count > budget:
        if state.get('_rewrite_prompt_kind') == 'Automatic continuation':
            raise ValueError(
                f'Automatic continuation prompt requires {count} tokens after extension hooks; '
                f'only {budget} prompt tokens are available. Draft instructions will not be '
                'silently truncated. Reduce the generation allowance or increase the context limit.'
            )
        raise ValueError(
            f'Sentence rewrite prompt requires {count} tokens after extension hooks; '
            f'only {budget} prompt tokens are available. References will not be '
            'silently truncated. Reduce K/window size or increase the context limit.'
        )
    return count


def automatic_prefill_token_suffix(question, state, input_ids):
    """Locate the native token tail added by prefill, including boundary merges."""
    prefill = state.get('_notebook_auto_prefill_suffix')
    if not prefill:
        return None
    if not question.endswith(prefill):
        raise ValueError('Automatic continuation prefill must end with the exact unfinished sentence prefix.')
    base = encode(question[:-len(prefill)], add_bos_token=state['add_bos_token'], truncation_length=None)
    before, after = base[0].tolist(), input_ids[0].tolist()
    common = 0
    for old, new in zip(before, after):
        if old != new:
            break
        common += 1
    suffix = after[common:]
    if not suffix:
        raise ValueError('The native tokenizer did not produce a verifiable automatic continuation prefill.')
    return suffix


def validate_automatic_prefill_tokens(input_ids, inputs_embeds, suffix):
    if suffix is None:
        return
    if inputs_embeds is not None:
        raise ValueError('Automatic continuation cannot verify the prefill cursor when a tokenizer extension supplies input embeddings. Disable that extension or use raw continuation.')
    if input_ids is None or input_ids[0].tolist()[-len(suffix):] != suffix:
        raise ValueError('A tokenizer extension changed the automatic continuation prefill cursor. The exact native token suffix must be preserved.')


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        models.load_model_if_idle_unloaded()
        if shared.tokenizer is None:
            raise ValueError('No tokenizer is loaded')

    # llama.cpp case
    if shared.model.__class__.__name__ == 'LlamaServer':
        input_ids = shared.tokenizer.encode(str(prompt), add_bos_token=add_bos_token)
        input_ids = np.array(input_ids).reshape(1, len(input_ids))

        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        return input_ids

    # All other model types
    else:
        import torch

        from modules.torch_utils import get_device

        if shared.model.__class__.__name__ in ['Exllamav3Model', 'TensorRTLLMModel']:
            if shared.model.__class__.__name__ == 'TensorRTLLMModel':
                input_ids = shared.model.encode_prompt(str(prompt), add_bos_token=add_bos_token)
            else:
                input_ids = shared.tokenizer.encode(str(prompt))
            if shared.model.__class__.__name__ not in ['Exllamav3Model']:
                input_ids = np.array(input_ids).reshape(1, len(input_ids))
        else:
            input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)
            if hasattr(shared.tokenizer, 'bos_token_id') and shared.tokenizer.bos_token_id is not None:
                if add_bos_token:
                    # Add BOS token if missing
                    if (len(input_ids[0]) > 0 and input_ids[0][0] != shared.tokenizer.bos_token_id) or len(input_ids[0]) == 0:
                        bos_tensor = torch.tensor([[shared.tokenizer.bos_token_id]])
                        input_ids = torch.cat((bos_tensor, input_ids), 1)

                # Always prevent double BOS tokens (regardless of add_bos_token setting)
                while len(input_ids[0]) > 1 and input_ids[0][0] == shared.tokenizer.bos_token_id and input_ids[0][1] == shared.tokenizer.bos_token_id:
                    input_ids = input_ids[:, 1:]

        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        if shared.model.__class__.__name__ in ['Exllamav3Model', 'TensorRTLLMModel'] or shared.args.cpu:
            return input_ids
        else:
            device = get_device()
            if device:
                return input_ids.to(device)

            return input_ids


def decode(output_ids, skip_special_tokens=True):
    if shared.tokenizer is None:
        models.load_model_if_idle_unloaded()
        if shared.tokenizer is None:
            raise ValueError('No tokenizer is loaded')

    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


def get_encoded_length(prompt):
    length_after_extensions = apply_extensions('tokenized_length', prompt)
    if length_after_extensions is not None:
        return length_after_extensions

    return len(encode(prompt)[0])


def get_rewrite_prompt_length(prompt, state, use_extensions=True):
    """Count the tokens the selected loader will actually use for rewriting."""
    if use_extensions:
        length = apply_extensions('tokenized_length', prompt)
        if length is not None:
            return length

    model_name = shared.model.__class__.__name__
    if model_name == 'Exllamav3Model':
        return int(shared.model.encode_rewrite_prompt(str(prompt), state).shape[-1])
    if model_name == 'TensorRTLLMModel':
        return len(shared.model.encode_prompt(str(prompt), add_bos_token=state['add_bos_token']))
    return len(encode(prompt, add_bos_token=state['add_bos_token'])[0])


def get_token_ids(prompt):
    tokens = encode(prompt)[0]
    decoded_tokens = [shared.tokenizer.decode([int(i)]) for i in tokens]

    output = ''
    for row in list(zip(tokens, decoded_tokens)):
        output += f"{str(int(row[0])).ljust(5)}  -  {repr(row[1])}\n"

    return output


def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']


def generate_reply_wrapper(question, state, stopping_strings=None):
    """
    Returns formatted outputs for the UI
    """
    model_is_loaded, error_message = check_model_loaded()
    if not model_is_loaded:
        import gradio as gr
        raise gr.Error(error_message)

    reply = question if not shared.is_seq2seq else ''
    yield formatted_outputs(reply, shared.model_name)

    for reply in generate_reply(question, state, stopping_strings, is_chat=False, escape_html=True, for_ui=True):
        if not shared.is_seq2seq:
            reply = question + reply

        yield formatted_outputs(reply, shared.model_name)


def formatted_outputs(reply, model_name):
    return html.unescape(reply), generate_basic_html(reply)


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    if shared.args.loader != 'llama.cpp':
        import torch
        from transformers import is_torch_npu_available, is_torch_xpu_available

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif is_torch_xpu_available():
            torch.xpu.manual_seed_all(seed)
        elif is_torch_npu_available():
            torch.npu.manual_seed_all(seed)

    return seed


def stop_everything_event():
    shared.stop_everything = True


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def get_reply_from_output_ids(output_ids, state=None, starting_from=0):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            # try to decode the bytes to a string
            # if it fails, which means it's not a string in this turn, just ignore it
            try:
                first_token = first_token.decode('utf8')
            except UnicodeDecodeError:
                first_token = ''

        if first_token.startswith('▁'):
            reply = ' ' + reply

    return reply


def generate_reply_HF(question, original_question, state, stopping_strings=None, is_chat=False):
    import torch
    import transformers
    from transformers import LogitsProcessorList

    from modules.grammar.grammar_utils import initialize_grammar
    from modules.grammar.logits_process import (
        GrammarConstrainedLogitsProcessor
    )
    from modules.torch_utils import clear_torch_cache, get_device
    from modules.transformers_loader import (
        Stream,
        _StopEverythingStoppingCriteria,
        get_eos_token_ids
    )

    if shared.args.loader == 'Transformers':
        clear_torch_cache()

    seed = set_manual_seed(state['seed'])

    generate_params = {}
    for k in [
        'temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'min_p',
        'top_p',
        'top_k',
        'typical_p',
        'xtc_threshold',
        'xtc_probability',
        'tfs',
        'top_a',
        'top_n_sigma',
        'adaptive_target',
        'adaptive_decay',
        'dry_multiplier',
        'dry_allowed_length',
        'dry_base',
        'repetition_penalty',
        'frequency_penalty',
        'presence_penalty',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'repetition_penalty_range',
        'penalty_alpha',
        'guidance_scale',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'max_new_tokens',
        'do_sample',
        'dynamic_temperature',
        'temperature_last',
        'dry_sequence_breakers',
    ]:
        if k in state:
            generate_params[k] = state[k]

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['prompt_lookup_num_tokens'] > 0:
        generate_params['prompt_lookup_num_tokens'] = state['prompt_lookup_num_tokens']

    eos_token_ids = get_eos_token_ids(shared.model, shared.tokenizer)
    if state['ban_eos_token']:
        if getattr(getattr(shared.model, 'config', None), 'model_type', None) == 'gemma4_unified':
            generate_params['suppress_tokens'] = list(eos_token_ids)
        else:
            generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if state['static_cache']:
        generate_params['cache_implementation'] = 'static'

    if isinstance(state['sampler_priority'], list) and len(state['sampler_priority']) > 0:
        generate_params['sampler_priority'] = state['sampler_priority']
    elif isinstance(state['sampler_priority'], str) and state['sampler_priority'].strip() != '':
        generate_params['sampler_priority'] = [x.strip() for x in state['sampler_priority'].replace('\n', ',').split(',') if x.strip()]

    if state['custom_token_bans']:
        to_ban = [int(x.strip()) for x in state['custom_token_bans'].split(',') if x.strip()]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    generate_params.update({'use_cache': not shared.args.no_cache})

    # Encode the input
    rewrite_guard = state.get('_rewrite_generation_guard', False)
    automatic = state.get('_notebook_auto_generation', False)
    automatic_event = state.get('stop_event') if automatic else None
    rewrite_event = state.get('stop_event') if rewrite_guard else None
    rewrite_kind = state.get('_rewrite_prompt_kind') if rewrite_guard else None
    automatic_prefill = state.get('_notebook_auto_prefill_suffix') if rewrite_guard else None
    rewrite_budget = get_max_prompt_length(state) if rewrite_guard else None
    input_ids = encode(question, add_bos_token=state['add_bos_token'],
                       truncation_length=None if rewrite_guard else get_max_prompt_length(state))
    output = input_ids[0]
    shared.model.last_prompt_token_count = input_ids.shape[-1]
    shared.model.last_completion_token_count = 0
    prefill_tokens = automatic_prefill_token_suffix(question, state, input_ids) if rewrite_guard else None
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    if automatic:
        state['_notebook_auto_generation'] = True
        state['stream'] = True
        state['skip_special_tokens'] = False
        state['auto_max_new_tokens'] = False
        if automatic_event is not None:
            state['stop_event'] = automatic_event
    if rewrite_guard:
        state['_rewrite_generation_guard'] = True
        state['skip_special_tokens'] = False
        if rewrite_kind is not None:
            state['_rewrite_prompt_kind'] = rewrite_kind
        if automatic_prefill is not None:
            state['_notebook_auto_prefill_suffix'] = automatic_prefill
        if rewrite_event is not None:
            state['stop_event'] = rewrite_event
        validate_automatic_prefill_tokens(input_ids, inputs_embeds, prefill_tokens)
        shared.model.last_prompt_token_count = validate_rewrite_prompt(
            question, state, input_ids, inputs_embeds, original_budget=rewrite_budget,
        )
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria(state.get('stop_event')))

    # Logits processor
    processor = state.get('logits_processor', LogitsProcessorList([]))
    if not isinstance(processor, LogitsProcessorList):
        processor = LogitsProcessorList([processor])

    # Grammar
    if state['grammar_string'].strip() != '':
        grammar = initialize_grammar(state['grammar_string'])
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        processor.append(grammar_processor)

    apply_extensions('logits_processor', processor, input_ids)
    generate_params['logits_processor'] = processor

    if shared.args.verbose:
        logger.info("GENERATE_PARAMS=")
        filtered_params = {key: value for key, value in generate_params.items() if not isinstance(value, torch.Tensor)}
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(filtered_params)
        print()

        logger.info("PROMPT=")
        print_prompt(decode(input_ids[0], skip_special_tokens=False))

    t0 = time.time()
    try:
        if not is_chat and not shared.is_seq2seq:
            yield ''

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                device = get_device()
                if device:
                    output = output.to(device)

            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            shared.model.last_completion_token_count = len(output) - starting_from
            yield get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:

            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None,
                                   raise_exceptions=bool(rewrite_guard or automatic))

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''
                prompt_len = 0 if shared.is_seq2seq else len(input_ids[0])
                starting_from = prompt_len
                for output in generator:
                    if output[-1] in eos_token_ids:
                        break

                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                    # check the partial unicode character
                    if chr(0xfffd) in new_content:
                        continue

                    cumulative_reply += new_content
                    shared.model.last_completion_token_count = len(output) - prompt_len
                    starting_from = len(output)
                    yield cumulative_reply

    except Exception:
        logger.exception("Failed to generate reply (HF)")
        if rewrite_guard or state.get('_notebook_auto_generation', False):
            raise
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        logger.info(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        if not rewrite_guard and not state.get('_notebook_auto_generation', False):
            return


def generate_reply_custom(question, original_question, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """

    stop_event_ref = state.pop('stop_event', None)
    state = copy.deepcopy(state)
    if stop_event_ref is not None:
        state['stop_event'] = stop_event_ref
    state['seed'] = set_manual_seed(state['seed'])
    t0 = time.time()
    reply = ''
    try:
        if not is_chat:
            yield ''

        if not state['stream']:
            reply = shared.model.generate(question, state)
            yield reply
        else:
            model_stream = shared.model.generate_with_streaming(question, state)
            try:
                for reply in model_stream:
                    yield reply
            finally:
                close = getattr(model_stream, 'close', None)
                if close is not None:
                    close()

    except Exception:
        logger.exception("Failed to generate reply (custom)")
        if state.get('_rewrite_generation_guard', False) or state.get('_notebook_auto_generation', False):
            raise
    finally:
        t1 = time.time()

        if state.get('_rewrite_generation_guard', False) or state.get('_notebook_auto_generation', False):
            # Cleanup must not tokenize again: llama.cpp tokenization is an
            # HTTP request and could block Stop or mask the original failure.
            context = getattr(shared.model, 'last_prompt_token_count', None)
            context_info = f', context {context}' if context is not None else ''
            logger.info(
                f'Output generated in {(t1-t0):.2f} seconds '
                f'({len(reply)} characters{context_info}, seed {state["seed"]})'
            )
        elif hasattr(shared.model, 'last_prompt_token_count'):
            original_tokens = shared.model.last_prompt_token_count
            new_tokens = len(encode(reply)[0]) if reply else 0
        else:
            original_tokens = len(encode(original_question)[0])
            new_tokens = len(encode(original_question + reply)[0]) - original_tokens

        if not state.get('_rewrite_generation_guard', False) and not state.get('_notebook_auto_generation', False):
            logger.info(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {state["seed"]})')
            return


def print_prompt(prompt, max_chars=-1):
    DARK_YELLOW = "\033[38;5;3m"
    RESET = "\033[0m"

    if max_chars > 0 and len(prompt) > max_chars:
        half_chars = max_chars // 2
        hidden_len = len(prompt[half_chars:-half_chars])
        hidden_msg = f"{DARK_YELLOW}[...{hidden_len} characters hidden...]{RESET}"
        print(prompt[:half_chars] + hidden_msg + prompt[-half_chars:])
    else:
        print(prompt)

    print()

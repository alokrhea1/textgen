"""Native integration boundaries exercised with isolated model dependencies."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def native(monkeypatch):
    import modules

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        if name.startswith('modules.') and name.count('.') == 1:
            monkeypatch.setattr(modules, name.split('.')[1], module, raising=False)
        return module

    shared = stub('modules.shared', stop_everything=False,
                  model=SimpleNamespace(), tokenizer=SimpleNamespace(eos_token_id=None),
                  generation_lock=threading.Lock(),
                  args=SimpleNamespace(parallel=1, verbose=False, loader='stub', no_cache=False))
    models = stub('modules.models', load_model_if_idle_unloaded=lambda: None,
                  _generation_count_lock=threading.Lock(), active_generation_count=0,
                  last_generation_time=0)
    stub('modules.callbacks', Iteratorize=object)
    extensions = stub('modules.extensions', apply_extensions=lambda kind, *args, **kwargs: None)
    stub('modules.html_generator', generate_basic_html=lambda value: value)
    stub('modules.logging_colors', logger=SimpleNamespace(info=lambda *args: None, exception=lambda *args: None))
    stub('modules.utils', check_model_loaded=lambda: (True, None))
    spec = importlib.util.spec_from_file_location(
        '_rewrite_test_native_generation', Path(__file__).parents[1] / 'modules' / 'text_generation.py',
    )
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    encodes = []

    def encode(prompt, **kwargs):
        encodes.append(kwargs)
        return np.zeros((1, len(prompt)), dtype=np.int64)

    monkeypatch.setattr(generation, 'encode', encode)
    return SimpleNamespace(generation=generation, shared=shared, models=models,
                           stub=stub, encodes=encodes)


def settings(**extra):
    value = dict(truncation_length=20, max_new_tokens=5, add_bos_token=True,
                 custom_stopping_strings=[], stream=True, max_tokens_second=0,
                 _rewrite_generation_guard=True, skip_special_tokens=True)
    value.update(extra)
    return value


def test_post_state_and_input_hooks_cannot_silently_truncate(native):
    generated = []

    def backend(*args, **kwargs):
        generated.append(True)
        yield 'Replacement.'

    def extensions(kind, *args, **kwargs):
        if kind == 'custom_generate_reply':
            return backend
        if kind == 'state':
            state = dict(args[0])
            state.pop('_rewrite_generation_guard')
            return state
        if kind == 'input':
            return args[0] + ' expansion beyond available budget'
        return args[0]

    native.generation.apply_extensions = extensions
    with pytest.raises(ValueError, match='after extension hooks'):
        list(native.generation.generate_reply('prompt', settings()))
    assert not generated
    assert not native.shared.generation_lock.locked()
    assert native.models.active_generation_count == 0


def test_ordinary_generation_does_not_opt_into_rewrite_budget(native):
    def backend(*args, **kwargs):
        yield 'Ordinary output.'

    def extensions(kind, *args, **kwargs):
        if kind == 'custom_generate_reply':
            return backend
        if kind == 'input':
            return args[0] + ' long extension expansion' * 10
        return args[0]

    native.generation.apply_extensions = extensions
    assert list(native.generation.generate_reply('prompt', settings(_rewrite_generation_guard=False)))[-1] == 'Ordinary output.'
    assert not native.encodes


def test_early_stop_runs_output_hook_once_and_closes_backend(native):
    calls, closed = [], []

    def backend(prompt, original, state, *args, **kwargs):
        assert state['_rewrite_generation_guard'] and not state['skip_special_tokens']
        try:
            yield 'A better'
            yield 'A better sentence. '
            raise AssertionError('Native stop predicate failed to stop consumption')
        finally:
            closed.append(True)

    def extensions(kind, *args, **kwargs):
        calls.append(kind)
        if kind == 'custom_generate_reply':
            return backend
        if kind == 'state':
            return {key: value for key, value in args[0].items() if not key.startswith('_rewrite')}
        if kind == 'output':
            return 'Fresh ' + args[0]
        return args[0]

    native.generation.apply_extensions = extensions
    output = list(native.generation.generate_reply(
        'prompt', settings(_rewrite_stop_predicate=lambda reply: 'sentence. ' in reply),
    ))
    assert output[-1] == 'Fresh A better sentence. '
    assert closed == [True]
    assert all(calls.count(kind) == 1 for kind in ['state', 'input', 'output'])
    assert not native.shared.generation_lock.locked()


@pytest.mark.parametrize('with_embeds', [False, True])
def test_hf_post_tokenizer_expansion_is_guarded_without_initial_truncation(native, with_embeds):
    import torch

    native.stub('transformers', LogitsProcessorList=list)
    native.stub('modules.grammar.grammar_utils', initialize_grammar=lambda *args: None)
    native.stub('modules.grammar.logits_process', GrammarConstrainedLogitsProcessor=object)
    native.stub('modules.torch_utils', clear_torch_cache=lambda: None, get_device=lambda: None)
    native.stub('modules.transformers_loader', Stream=object, _StopEverythingStoppingCriteria=object,
                get_eos_token_ids=lambda *args: [])
    native.generation.set_manual_seed = lambda seed: seed

    def extensions(kind, *args, **kwargs):
        if kind == 'tokenizer':
            state, prompt, ids, embeds = args
            # Try dropping the guard, too: the HF path captures it before hooks.
            state.pop('_rewrite_generation_guard')
            return prompt, np.zeros((1, 4 if with_embeds else 30)), torch.zeros((1, 30, 4)) if with_embeds else None

    native.generation.apply_extensions = extensions
    state = settings(seed=0, epsilon_cutoff=0, eta_cutoff=0, prompt_lookup_num_tokens=0,
                     ban_eos_token=False, static_cache=False, sampler_priority=[],
                     custom_token_bans='', negative_prompt='', auto_max_new_tokens=False)
    with pytest.raises(ValueError, match='after extension hooks'):
        list(native.generation.generate_reply_HF('prompt', 'prompt', state))
    assert native.encodes[0]['truncation_length'] is None


def test_cancelled_lock_wait_does_not_reset_global_stop_or_count(native):
    event = threading.Event()
    entered_wait = threading.Event()
    real_lock = native.shared.generation_lock

    class ObservedLock:
        def acquire(self, *args, **kwargs):
            if 'timeout' in kwargs:
                entered_wait.set()
            return real_lock.acquire(*args, **kwargs)

        def release(self):
            real_lock.release()

    native.shared.generation_lock = ObservedLock()
    native.shared.generation_lock.acquire()
    native.shared.stop_everything = True
    outcome = []

    def request():
        try:
            list(native.generation.generate_reply('prompt', settings(stop_event=event)))
        except InterruptedError:
            outcome.append('cancelled')

    worker = threading.Thread(target=request)
    worker.start()
    waiting = entered_wait.wait(timeout=2)
    event.set()
    worker.join(timeout=2)
    try:
        assert waiting and not worker.is_alive() and outcome == ['cancelled']
        assert native.shared.stop_everything is True
        assert native.models.active_generation_count == 0
    finally:
        native.shared.generation_lock.release()


def test_actual_hf_criterion_observes_local_and_global_stop(native):
    # Execute the actual class without importing unrelated heavyweight loaders.
    import ast

    source = (Path(__file__).parents[1] / 'modules' / 'transformers_loader.py').read_text()
    criterion = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == '_StopEverythingStoppingCriteria')
    namespace = dict(transformers=SimpleNamespace(StoppingCriteria=type('StoppingCriteria', (), {})),
                     torch=SimpleNamespace(LongTensor=object, FloatTensor=object), shared=native.shared)
    exec(compile(ast.Module(body=[criterion], type_ignores=[]), '<native HF criterion>', 'exec'), namespace)
    event = threading.Event()
    stop = namespace['_StopEverythingStoppingCriteria'](event)
    assert stop(None, None) is False
    event.set()
    assert stop(None, None) is True
    event.clear()
    native.shared.stop_everything = True
    assert stop(None, None) is True


def test_trt_early_close_aborts_async_result(native):
    native.stub('tensorrt_llm', __path__=[])
    native.stub('tensorrt_llm._tensorrt_engine', LLM=object)
    native.stub('tensorrt_llm.llmapi', SamplingParams=lambda **kwargs: kwargs)
    native.shared.tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2], eos_token_id=0)
    spec = importlib.util.spec_from_file_location(
        '_rewrite_test_trt', Path(__file__).parents[1] / 'modules' / 'tensorrt_llm.py',
    )
    trt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trt)

    class Result:
        aborted = False

        def __iter__(self):
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[3], text_diff='Sentence. ')])

        def abort(self):
            self.aborted = True

    result = Result()
    model = trt.TensorRTLLMModel()
    model.llm = SimpleNamespace(generate_async=lambda *args, **kwargs: result)
    state = settings(auto_max_new_tokens=False, temperature=1, top_k=20, top_p=.9,
                     min_p=0, repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
                     no_repeat_ngram_size=0, seed=0, ban_eos_token=False)
    stream = model.generate_with_streaming('prompt', state)
    assert next(stream) == 'Sentence. '
    stream.close()
    assert result.aborted


def _loader_function(name, namespace):
    import ast

    source = (Path(__file__).parents[1] / 'modules' / 'transformers_loader.py').read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<native Transformers loader>', 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize(('model_type', 'encoder_decoder', 'name', 'expected'), [
    ('gemma4_unified', False, 'gemma', 'multimodal'),
    ('llama', False, 'llama', 'causal'),
    ('t5', True, 't5', 'seq2seq'),
    ('chatglm', False, 'chatglm', 'base'),
])
def test_gemma_unified_loader_is_narrow_and_decoder_only(monkeypatch, model_type, encoder_decoder, name, expected):
    selected = []

    def loader(kind):
        return SimpleNamespace(from_pretrained=lambda *args, **kwargs: selected.append(kind) or SimpleNamespace())

    transformers = ModuleType('transformers')
    transformers.AutoModelForMultimodalLM = loader('multimodal')
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    config = SimpleNamespace(model_type=model_type, torch_dtype='bfloat16',
                             to_dict=lambda: {'is_encoder_decoder': encoder_decoder})
    shared = SimpleNamespace(
        args=SimpleNamespace(model_dir='/models', attn_implementation='sdpa', force_safetensors=False,
                             bf16=False, cpu=False, load_in_8bit=False, load_in_4bit=False,
                             disk=False, cpu_memory=None),
        original_args=SimpleNamespace(trust_remote_code=False), is_seq2seq=False,
    )
    namespace = dict(
        torch=SimpleNamespace(_dynamo=SimpleNamespace(config=SimpleNamespace(disable=False)),
                              float16='float16', bfloat16='bfloat16'),
        shared=shared, Path=Path, AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: config),
        AutoModel=loader('base'), AutoModelForCausalLM=loader('causal'), AutoModelForSeq2SeqLM=loader('seq2seq'),
        get_device=lambda: None, logger=SimpleNamespace(info=lambda *args: None),
        pprint=SimpleNamespace(PrettyPrinter=lambda **kwargs: SimpleNamespace(pprint=lambda *args: None)),
    )
    _loader_function('load_model_HF', namespace)(name)
    assert selected == [expected]
    assert shared.is_seq2seq is encoder_decoder


@pytest.mark.parametrize(('model_type', 'configured', 'tokenizer_id', 'expected'), [
    ('gemma4_unified', [1, 106], 1, [1, 106]),
    ('gemma4_unified', 106, 1, [1, 106]),
    ('gemma4_unified', [1, 106], None, [1, 106]),
    ('llama', [1, 106], 1, [1]),
    ('llama', [1, 106], None, []),
])
def test_checkpoint_turn_eos_merge_is_gemma_only(model_type, configured, tokenizer_id, expected):
    ids = _loader_function('get_eos_token_ids', {})
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type),
                            generation_config=SimpleNamespace(eos_token_id=configured))
    assert ids(model, SimpleNamespace(eos_token_id=tokenizer_id)) == expected


def test_gemma_ban_eos_suppresses_checkpoint_turn_terminator(native):
    import torch

    native.stub('transformers', LogitsProcessorList=list, StoppingCriteriaList=list)
    native.stub('modules.grammar.grammar_utils', initialize_grammar=lambda *args: None)
    native.stub('modules.grammar.logits_process', GrammarConstrainedLogitsProcessor=object)
    native.stub('modules.torch_utils', clear_torch_cache=lambda: None, get_device=lambda: None)
    native.stub('modules.transformers_loader', Stream=object,
                _StopEverythingStoppingCriteria=lambda event: None,
                get_eos_token_ids=_loader_function('get_eos_token_ids', {}))
    captured = []

    def generate(**kwargs):
        captured.append(kwargs)
        return torch.tensor([[0, 0, 0, 0, 0, 0, 2]])

    native.shared.model = SimpleNamespace(config=SimpleNamespace(model_type='gemma4_unified'),
                                          generation_config=SimpleNamespace(eos_token_id=[1, 106]), generate=generate)
    native.shared.tokenizer.eos_token_id = 1
    native.shared.is_seq2seq = False
    native.generation.set_manual_seed = lambda seed: seed
    native.generation.get_reply_from_output_ids = lambda *args, **kwargs: 'Sentence.'

    def extensions(kind, *args, **kwargs):
        if kind == 'tokenizer':
            return args[1:]
        return None

    native.generation.apply_extensions = extensions
    state = settings(stream=False, _rewrite_generation_guard=False, seed=0,
                     epsilon_cutoff=0, eta_cutoff=0, prompt_lookup_num_tokens=0,
                     ban_eos_token=True, static_cache=False, sampler_priority=[], grammar_string='',
                     custom_token_bans='', negative_prompt='', auto_max_new_tokens=False)
    assert list(native.generation.generate_reply_HF('prompt', 'prompt', state))[-1] == 'Sentence.'
    assert captured[0]['eos_token_id'] == [1, 106]
    assert captured[0]['suppress_tokens'] == [1, 106]

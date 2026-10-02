"""Automatic Notebook drafting preserves native semantics behind an opt-in."""
import ast
from contextlib import nullcontext
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_rewrite_native import native, settings


def draft_settings(**extra):
    return settings(_rewrite_generation_guard=False, _notebook_auto_generation=True,
                    **extra)


def test_draft_state_hook_keeps_local_cancellation_and_boundary_metadata(native):
    event, info, closed, boundary_calls = threading.Event(), {}, [], []

    def boundary(reply):
        boundary_calls.append(reply)
        return reply.endswith('. ')

    def backend(prompt, original, state, *args, **kwargs):
        assert state['_notebook_auto_generation'] is True
        assert state['_notebook_auto_generation_info'] is info
        assert state['_notebook_auto_stop_predicate'] is boundary
        assert state['stop_event'] is event
        assert state['stream'] is True
        assert state['auto_max_new_tokens'] is False
        assert state['skip_special_tokens'] is False
        assert not state.get('_rewrite_generation_guard', False)
        assert len(prompt) > 20
        try:
            yield 'Draft sentence. '
            raise AssertionError('Sentence boundary did not close the backend')
        finally:
            closed.append(True)

    def extensions(kind, *args, **kwargs):
        if kind == 'custom_generate_reply':
            return backend
        if kind == 'state':
            state = {key: value for key, value in args[0].items()
                     if not key.startswith('_notebook') and key != 'stop_event'}
            state.update(stream=False, auto_max_new_tokens=True, skip_special_tokens=True)
            return state
        if kind == 'input':
            return args[0] + ' ordinary context extension' * 10
        if kind == 'output':
            return 'Final ' + args[0]
        return args[0]

    native.generation.apply_extensions = extensions
    output = list(native.generation.generate_reply(
        'prompt', draft_settings(stop_event=event, _notebook_auto_generation_info=info,
                                 _notebook_auto_stop_predicate=boundary)))
    assert output[-1] == 'Final Draft sentence. '
    assert boundary_calls == ['Draft sentence. ']
    assert info == {'raw_reply': 'Draft sentence. ', 'boundary': True}
    assert closed == [True]
    assert not native.encodes, 'Drafting must not acquire the rewrite no-truncation guard'
    assert not native.shared.generation_lock.locked()
    assert native.models.active_generation_count == 0


def test_draft_custom_stopping_string_overrides_sentence_boundary(native):
    info, boundaries, outputs, closed = {}, [], [], []

    def backend(*args, **kwargs):
        try:
            yield 'Draft sentence. HALT'
            raise AssertionError('Custom stopping string was ignored')
        finally:
            closed.append(True)

    def extensions(kind, *args, **kwargs):
        if kind == 'custom_generate_reply':
            return backend
        if kind == 'output':
            outputs.append(args[0])
            return args[0] + '[hook]'
        return args[0]

    native.generation.apply_extensions = extensions
    output = list(native.generation.generate_reply(
        'prompt', draft_settings(custom_stopping_strings=['HALT'],
                                 _notebook_auto_generation_info=info,
                                 _notebook_auto_stop_predicate=lambda reply: boundaries.append(reply) or True)))
    assert output[-1] == 'Draft sentence. [hook]'
    assert outputs == ['Draft sentence. ']
    assert boundaries == []
    assert info == {'raw_reply': 'Draft sentence. HALT', 'terminal': True}
    assert closed == [True]


def test_new_local_event_does_not_inherit_previous_global_stop(native):
    native.shared.stop_everything = True
    event, info = threading.Event(), {}

    def backend(*args, **kwargs):
        assert not native.shared.stop_everything
        yield 'Fresh draft. '

    native.generation.apply_extensions = lambda kind, *args, **kwargs: backend if kind == 'custom_generate_reply' else args[0]
    result = list(native.generation.generate_reply(
        'prompt', draft_settings(stop_event=event, _notebook_auto_generation_info=info,
                                 _notebook_auto_stop_predicate=lambda reply: reply.endswith('. '))))
    assert result[-1] == 'Fresh draft. '
    assert info['boundary'] is True and not info.get('terminal')
    assert not event.is_set()


def test_draft_iterator_close_releases_backend_and_native_generation_lock(native):
    closed = []

    def backend(*args, **kwargs):
        try:
            yield 'A provisional draft'
            yield 'that was never accepted'
        finally:
            closed.append(True)

    native.generation.apply_extensions = lambda kind, *args, **kwargs: backend if kind == 'custom_generate_reply' else args[0]
    generator = native.generation.generate_reply('prompt', draft_settings(stop_event=threading.Event()))
    assert next(generator) == 'A provisional draft'
    assert native.shared.generation_lock.locked()
    generator.close()
    assert closed == [True]
    assert not native.shared.generation_lock.locked()
    assert native.models.active_generation_count == 0


def test_draft_hf_retains_ordinary_prompt_truncation(native):
    import torch

    native.stub('transformers', LogitsProcessorList=list, StoppingCriteriaList=list)
    native.stub('modules.grammar.grammar_utils', initialize_grammar=lambda *args: None)
    native.stub('modules.grammar.logits_process', GrammarConstrainedLogitsProcessor=object)
    native.stub('modules.torch_utils', clear_torch_cache=lambda: None, get_device=lambda: None)
    native.stub('modules.transformers_loader', Stream=object,
                _StopEverythingStoppingCriteria=lambda event: event, get_eos_token_ids=lambda *args: [])
    native.shared.model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[0, 0, 1]]))
    native.shared.is_seq2seq = False
    native.generation.set_manual_seed = lambda seed: seed
    native.generation.get_reply_from_output_ids = lambda *args, **kwargs: 'Draft.'

    class Streaming:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return iter([np.array([0, 0, 1])])

        def __exit__(self, *args):
            return False

    native.generation.Iteratorize = Streaming

    def encode(prompt, **kwargs):
        native.encodes.append(kwargs)
        return np.zeros((1, 2), dtype=np.int64)

    native.generation.encode = encode
    native.generation.apply_extensions = lambda kind, *args, **kwargs: args[1:] if kind == 'tokenizer' else None
    state = draft_settings(seed=0, stream=True, epsilon_cutoff=0, eta_cutoff=0,
                           prompt_lookup_num_tokens=0, ban_eos_token=False, static_cache=False,
                           sampler_priority=[], grammar_string='', custom_token_bans='',
                           negative_prompt='', auto_max_new_tokens=False)
    assert list(native.generation.generate_reply_HF('long prompt', 'long prompt', state))[-1] == 'Draft.'
    assert native.encodes[0]['truncation_length'] == native.generation.get_max_prompt_length(state)
    assert native.encodes[0]['truncation_length'] is not None


@pytest.mark.parametrize('automatic', [False, True])
def test_custom_backend_errors_are_loud_only_for_automatic_opt_in(native, automatic):
    native.generation.set_manual_seed = lambda seed: seed
    closed = []

    class LlamaServer:
        last_prompt_token_count = 3

        def generate_with_streaming(self, prompt, state):
            try:
                raise RuntimeError('Custom backend failed while drafting')
                yield ''
            finally:
                closed.append(True)

    native.shared.model = LlamaServer()
    generated = native.generation.generate_reply_custom(
        'prompt', 'prompt', settings(seed=0, _rewrite_generation_guard=False,
                                     _notebook_auto_generation=automatic))
    if automatic:
        with pytest.raises(RuntimeError, match='failed while drafting'):
            list(generated)
    else:
        assert list(generated) == ['']
    assert closed == [True]


@pytest.mark.parametrize('automatic', [False, True])
def test_hf_backend_errors_are_loud_only_for_automatic_opt_in(native, automatic):
    source = (Path(__file__).parents[1] / 'modules' / 'text_generation.py').read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'generate_reply_HF')
    block = next(node for node in function.body if isinstance(node, ast.Try))
    wrapper = ast.FunctionDef(name='generate', args=ast.arguments(posonlyargs=[], args=[],
                             kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=[*ast.parse('output = []').body, block], decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))

    def fail(**kwargs):
        raise RuntimeError('HF automatic draft failed')

    native.shared.model = SimpleNamespace(generate=fail)
    native.shared.is_seq2seq = False
    namespace = dict(shared=native.shared, state={'stream': False, '_notebook_auto_generation': automatic},
                     is_chat=False, rewrite_guard=False, torch=SimpleNamespace(no_grad=nullcontext),
                     generate_params={}, output=[], original_input_ids=[[1, 2]], time=time,
                     t0=time.time(), seed=0, logger=native.generation.logger)
    exec(compile(tree, '<HF automatic exception boundary>', 'exec'), namespace)
    if automatic:
        with pytest.raises(RuntimeError, match='automatic draft failed'):
            list(namespace['generate']())
    else:
        assert list(namespace['generate']()) == ['']


def test_prefill_suffix_survives_state_hook_before_drafting(native):
    suffix = 'unfin'
    prompt = 'Context. <assistant>' + suffix
    passed = []

    def backend(question, original, state, *args, **kwargs):
        passed.append(state)
        assert question == prompt
        assert state['_notebook_auto_prefill_suffix'] == suffix
        assert state['_notebook_auto_generation'] is True
        assert state['_rewrite_generation_guard'] is True
        assert state['_rewrite_prompt_kind'] == 'Automatic continuation'
        yield 'ished sentence.'

    def extensions(kind, *args, **kwargs):
        if kind == 'custom_generate_reply':
            return backend
        if kind == 'state':
            return {key: value for key, value in args[0].items()
                    if not key.startswith('_notebook') and not key.startswith('_rewrite')}
        return args[0]

    native.generation.apply_extensions = extensions
    state = settings(truncation_length=100, _notebook_auto_generation=True,
                     _notebook_auto_prefill_suffix=suffix,
                     _rewrite_prompt_kind='Automatic continuation')
    assert list(native.generation.generate_reply(prompt, state))[-1] == 'ished sentence.'
    assert len(passed) == 1


def test_input_hook_cannot_move_the_automatic_prefill_cursor(native):
    started = []

    def backend(*args, **kwargs):
        started.append(True)
        yield 'A draft.'

    def extensions(kind, *args, **kwargs):
        if kind == 'custom_generate_reply':
            return backend
        if kind == 'input':
            return args[0] + ' appended instructions'
        return args[0]

    native.generation.apply_extensions = extensions
    state = settings(truncation_length=100, _notebook_auto_generation=True,
                     _notebook_auto_prefill_suffix='unfin',
                     _rewrite_prompt_kind='Automatic continuation')
    with pytest.raises(ValueError):
        list(native.generation.generate_reply('Context. <assistant>unfin', state))
    assert not started
    assert not native.shared.generation_lock.locked()
    assert native.models.active_generation_count == 0


def test_prompt_validation_rejects_a_malformed_automatic_prefill_suffix(native):
    state = settings(truncation_length=100, _notebook_auto_generation=True,
                     _notebook_auto_prefill_suffix='unfin',
                     _rewrite_prompt_kind='Automatic continuation')
    with pytest.raises(ValueError):
        native.generation.validate_rewrite_prompt('Context. <assistant>changed', state)


def _prefill_hf(native, token_hook):
    """A cursor spans a merged boundary token: standalone suffix IDs are wrong."""
    native.stub('transformers', LogitsProcessorList=list, StoppingCriteriaList=list)
    native.stub('modules.grammar.grammar_utils', initialize_grammar=lambda *args: None)
    native.stub('modules.grammar.logits_process', GrammarConstrainedLogitsProcessor=object)
    native.stub('modules.torch_utils', clear_torch_cache=lambda: None, get_device=lambda: None)
    native.stub('modules.transformers_loader', Stream=object,
                _StopEverythingStoppingCriteria=lambda event: event,
                get_eos_token_ids=lambda *args: [])
    native.shared.model = SimpleNamespace()
    native.shared.is_seq2seq = False
    native.generation.set_manual_seed = lambda seed: seed
    native.generation.get_reply_from_output_ids = lambda *args, **kwargs: 'ished sentence.'
    suffix = 'unfin'
    base = 'Context. <assistant>'
    prompt = base + suffix
    encodes, generated = [], []

    def encode(question, **kwargs):
        encodes.append((question, kwargs))
        # The preceding token merges with the first prefill characters. The
        # protected token tail must therefore start at 30, not merely at 40.
        if question == prompt:
            return np.array([[1, 10, 30, 40]], dtype=np.int64)
        if question == base:
            return np.array([[1, 10, 20]], dtype=np.int64)
        if question == suffix:
            raise AssertionError('A merged prefill cannot be validated by encoding the suffix alone')
        raise AssertionError('Unexpected prompt tokenization: ' + question)

    class Streaming:
        def __init__(self, function, args, kwargs, **extra):
            generated.append(kwargs)
            self.kwargs = kwargs

        def __enter__(self):
            output = np.append(self.kwargs['inputs'][0], 99)
            return iter([output])

        def __exit__(self, *args):
            return False

    def extensions(kind, *args, **kwargs):
        if kind == 'tokenizer':
            return token_hook(*args)
        return None

    native.generation.encode = encode
    native.generation.Iteratorize = Streaming
    native.generation.apply_extensions = extensions
    state = settings(truncation_length=100, _notebook_auto_generation=True,
                     _notebook_auto_prefill_suffix=suffix,
                     _rewrite_prompt_kind='Automatic continuation', seed=0,
                     epsilon_cutoff=0, eta_cutoff=0, prompt_lookup_num_tokens=0,
                     ban_eos_token=False, static_cache=False, sampler_priority=[],
                     grammar_string='', custom_token_bans='', negative_prompt='',
                     auto_max_new_tokens=False)
    return SimpleNamespace(prompt=prompt, base=base, suffix=suffix, state=state,
                           encodes=encodes, generated=generated)


@pytest.mark.parametrize('changed_context', [False, True])
def test_hf_prefill_guard_retains_merged_token_tail_and_allows_earlier_context_changes(native, changed_context):
    observed = []

    def hook(state, question, ids, embeds):
        # Extensions cannot discard metadata to disable the cursor check.
        state.pop('_notebook_auto_prefill_suffix')
        state.pop('_notebook_auto_generation')
        state.pop('_rewrite_prompt_kind')
        state.pop('_rewrite_generation_guard')
        observed.append(state)
        if changed_context:
            ids = np.array([[8, 9, 30, 40]], dtype=np.int64)
        return question, ids, None

    h = _prefill_hf(native, hook)
    assert list(native.generation.generate_reply_HF(h.prompt, h.prompt, h.state))[-1] == 'ished sentence.'
    assert len(h.generated) == 1
    assert h.generated[0]['inputs'].tolist() == ([[8, 9, 30, 40]] if changed_context else [[1, 10, 30, 40]])
    assert observed[0]['_notebook_auto_prefill_suffix'] == h.suffix
    assert observed[0]['_notebook_auto_generation'] is True
    assert observed[0]['_rewrite_generation_guard'] is True
    assert observed[0]['_rewrite_prompt_kind'] == 'Automatic continuation'
    assert any(question == h.base for question, kwargs in h.encodes)
    assert all(question != h.suffix for question, kwargs in h.encodes)


@pytest.mark.parametrize('corruption', ['append', 'replace', 'merged_boundary', 'text_suffix', 'inputs_embeds'])
def test_hf_prefill_guard_rejects_tokenizer_cursor_changes_before_worker_start(native, corruption):
    def hook(state, question, ids, embeds):
        state.pop('_notebook_auto_prefill_suffix')
        state.pop('_notebook_auto_generation')
        state.pop('_rewrite_prompt_kind')
        state.pop('_rewrite_generation_guard')
        if corruption == 'append':
            ids = np.array([[1, 10, 30, 40, 55]], dtype=np.int64)
        elif corruption == 'replace':
            ids = np.array([[1, 10, 30, 55]], dtype=np.int64)
        elif corruption == 'merged_boundary':
            ids = np.array([[1, 10, 55, 40]], dtype=np.int64)
        elif corruption == 'text_suffix':
            question += ' instructions'
        else:
            embeds = np.zeros((1, len(ids[0]), 4), dtype=np.float32)
        return question, ids, embeds

    h = _prefill_hf(native, hook)
    with pytest.raises(ValueError):
        list(native.generation.generate_reply_HF(h.prompt, h.prompt, h.state))
    assert h.generated == []


@pytest.mark.parametrize('manual', [False, True])
def test_nonautomatic_hf_generation_allows_embedding_hooks_without_a_prefill_guard(native, manual):
    def hook(state, question, ids, embeds):
        ids = np.array([[1, 10, 30, 55]], dtype=np.int64)
        return question, ids, np.zeros((1, 4, 4), dtype=np.float32)

    h = _prefill_hf(native, hook)
    h.state.pop('_notebook_auto_prefill_suffix')
    h.state['_notebook_auto_generation'] = False
    h.state['_rewrite_generation_guard'] = manual
    assert list(native.generation.generate_reply_HF(h.prompt, h.prompt, h.state))[-1] == 'ished sentence.'
    assert len(h.generated) == 1
    assert 'inputs_embeds' in h.generated[0]
    assert all(question != h.base for question, kwargs in h.encodes)

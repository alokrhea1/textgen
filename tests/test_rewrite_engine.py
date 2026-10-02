import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from modules.sentence_rewrite.engine import generate_rewrite, prepare_prompt
from modules.sentence_rewrite.sentences import last_sentence


@pytest.fixture
def native(monkeypatch):
    import modules

    shared = ModuleType('modules.shared')
    shared.stop_everything = False
    shared.bos_token, shared.eos_token = '<bos>', '<eos>'
    generation = ModuleType('modules.text_generation')
    generation.get_rewrite_prompt_length = lambda prompt, state: len(prompt)
    generation.get_max_prompt_length = lambda state: state['truncation_length'] - state['max_new_tokens']
    generation.replies = ['A vivid replacement. ']
    generation.closed = False
    generation.calls = []

    def generate(prompt, state, **kwargs):
        shared.stop_everything = False
        generation.calls.append((prompt, state, kwargs))
        try:
            yield from generation.replies
        finally:
            generation.closed = True

    generation.generate_reply = generate
    models = ModuleType('modules.models')
    models.load_model_if_idle_unloaded = lambda: None
    utils = ModuleType('modules.utils')
    utils.check_model_loaded = lambda: (True, None)
    for name, module in [('shared', shared), ('text_generation', generation), ('models', models), ('utils', utils)]:
        monkeypatch.setitem(sys.modules, 'modules.' + name, module)
        monkeypatch.setattr(modules, name, module, raising=False)
    return SimpleNamespace(shared=shared, generation=generation, utils=utils)


def state(**changes):
    return dict(truncation_length=10000, max_new_tokens=100, stream=False,
                auto_max_new_tokens=True, temperature=0.73, instruction_template_str='', **changes)


def hits():
    return [SimpleNamespace(text='Evocative prose examples.', source='reference.txt', start=5, end=29)]


def test_plain_prompt_and_all_references(native):
    notebook = 'Earlier context. Original sentence. unfinished suffix'
    plan = prepare_prompt(notebook, last_sentence(notebook), hits(), state(), guidance='Use a lively rhythm.')
    assert not plan.context_trimmed
    assert plan.used_hits[0].source == 'reference.txt'
    assert 'Earlier context.' in plan.prompt and 'unfinished suffix' in plan.prompt
    assert 'Use a lively rhythm.' in plan.prompt
    assert plan.token_count == len(plan.prompt)


def test_trim_context_but_keep_all_hits(native):
    references = hits() + [SimpleNamespace(text='Another essential example.', source='second.txt', start=0, end=26)]
    notebook = 'Older sentence. ' * 500 + 'Original sentence. trailing context'
    minimum = prepare_prompt('Original sentence.', last_sentence('Original sentence.'), references, state())
    settings = state()
    settings['truncation_length'] = minimum.token_count + settings['max_new_tokens'] + 90
    plan = prepare_prompt(notebook, last_sentence(notebook), references, settings)
    assert plan.context_trimmed and len(plan.used_hits) == 2
    assert all(hit.text in plan.prompt for hit in references)
    assert plan.token_count <= settings['truncation_length'] - settings['max_new_tokens']


def test_oversized_reference_is_explicit_error(native):
    settings = state()
    settings['truncation_length'] = 120
    with pytest.raises(ValueError, match='All 1 retrieved references'):
        prepare_prompt('Original sentence.', last_sentence('Original sentence.'), hits(), settings)


def test_selected_template_renderer(native, monkeypatch):
    chat = ModuleType('modules.chat')
    captured = {}

    class Template:
        def render(self, **kwargs):
            captured.update(kwargs)
            return '<template>' + kwargs['messages'][0]['content']

    chat.get_compiled_template = lambda template: Template()
    monkeypatch.setitem(sys.modules, 'modules.chat', chat)
    settings = state()
    settings['instruction_template_str'] = 'selected template'
    plan = prepare_prompt('Original sentence.', last_sentence('Original sentence.'), hits(), settings)
    assert plan.prompt.startswith('<template>')
    assert captured['add_generation_prompt'] is True
    assert captured['bos_token'] == '<bos>'


def test_rewrite_preserves_context_sampling_and_closes(native):
    settings = state()
    notebook = '  Earlier sentence.\n Original sentence.  unfinished'
    progress = list(generate_rewrite(notebook, hits(), settings))
    assert progress[-1].done
    assert progress[0].text == ''
    assert progress[-1].text == 'A vivid replacement.'
    assert native.generation.closed
    generated_state = native.generation.calls[0][1]
    assert generated_state['temperature'] == settings['temperature']
    assert generated_state['max_new_tokens'] == 100
    assert generated_state['stream'] and not generated_state['auto_max_new_tokens']
    assert generated_state['skip_special_tokens'] is False
    assert generated_state['_rewrite_generation_guard'] is True
    assert not settings['stream'] and settings['auto_max_new_tokens']


@pytest.mark.parametrize('reply', ['', 'An unfinished replacement'])
def test_reject_invalid_output_and_close(native, reply):
    native.generation.replies = [reply]
    with pytest.raises(ValueError):
        list(generate_rewrite('Original sentence.', hits(), state()))
    assert native.generation.closed


@pytest.mark.parametrize('reply', ['One sentence. Another sentence.', 'One sentence. extra fragment', 'One sentence. '])
def test_first_sentence_only_from_multi_token_chunks(native, reply):
    native.generation.replies = [reply]
    progress = list(generate_rewrite('Original sentence.', hits(), state()))
    assert progress[-1].text == 'One sentence.'
    assert progress[-1].done and native.generation.closed


def test_stream_preview_and_quoted_period(native):
    native.generation.replies = ['He said', 'He said “Go.”', 'He said “Go.” More']
    progress = list(generate_rewrite('Original sentence.', hits(), state()))
    assert progress[1].text == 'He said' and not progress[1].done
    assert progress[-1].text == 'He said “Go.”'


def test_inline_html_is_preserved(native):
    native.generation.replies = ['A <em>vivid</em> sentence. ']
    assert list(generate_rewrite('Original sentence.', hits(), state()))[-1].text == 'A <em>vivid</em> sentence.'


def test_reasoning_is_removed(native):
    native.generation.replies = ['<think>Consider options.', '<think>Consider options.</think>A better sentence. ']
    assert list(generate_rewrite('Original sentence.', hits(), state()))[-1].text == 'A better sentence.'


@pytest.mark.parametrize('reply', [
    '<think>Reasoning first.</think>A better sentence.</s>',
    '<|channel|>analysis<|message|>Reasoning first.<|end|><|channel|>final<|message|>A better sentence.<|return|>',
    '<|im_start|>assistant\nA better sentence.<|im_end|>',
    '<start_of_turn>model\nA better sentence.<end_of_turn>',
    '<|channel>thought\nReasoning first.<channel|>A better sentence.<turn|>',
    'A better sentence.<turn|><eos>',
])
def test_special_tokens_kept_for_reasoning_then_final_framing_removed(native, reply):
    native.generation.replies = [reply]
    progress = list(generate_rewrite('Original sentence.', hits(), state()))
    assert progress[-1].text == 'A better sentence.'


def test_prompt_open_reasoning_is_suppressed(native, monkeypatch):
    from modules.sentence_rewrite import engine

    original_prepare = engine.prepare_prompt

    def prepare(*args, **kwargs):
        plan = original_prepare(*args, **kwargs)
        return engine.PromptPlan(plan.prompt + '<think>\n', plan.used_hits, plan.context_trimmed, plan.token_count)

    monkeypatch.setattr(engine, 'prepare_prompt', prepare)
    native.generation.replies = ['Consider several options. ', 'Consider several options. </think>A better sentence. ']
    progress = list(generate_rewrite('Original sentence.', hits(), state()))
    assert [item.text for item in progress] == ['', 'A better sentence.']


def test_fresh_local_event_allows_retry_after_previous_global_stop(native):
    native.shared.stop_everything = True
    assert list(generate_rewrite('Original sentence.', hits(), state(), cancel_event=threading.Event()))[-1].done


def test_global_stop_during_native_generation_is_respected(native):
    def generate(*args, **kwargs):
        try:
            native.shared.stop_everything = True
            yield 'An unfinished replacement'
        finally:
            native.generation.closed = True

    native.generation.generate_reply = generate
    with pytest.raises(InterruptedError):
        list(generate_rewrite('Original sentence.', hits(), state(), cancel_event=threading.Event()))
    assert native.generation.closed


def test_stop_closes_native_generator(native):
    cancel = threading.Event()

    def generate(*args, **kwargs):
        try:
            cancel.set()
            yield 'Incomplete'
        finally:
            native.generation.closed = True

    native.generation.generate_reply = generate
    with pytest.raises(InterruptedError):
        list(generate_rewrite('Original sentence.', hits(), state(), cancel_event=cancel))
    assert native.generation.closed


def test_missing_model_and_preexisting_cancel(native):
    native.utils.check_model_loaded = lambda: (False, 'No model loaded')
    with pytest.raises(ValueError, match='No model'):
        list(generate_rewrite('Original sentence.', hits(), state()))
    native.shared.stop_everything = True
    with pytest.raises(InterruptedError):
        list(generate_rewrite('Original sentence.', hits(), state()))


def test_native_iterator_exit_waits_for_worker(native, monkeypatch):
    import importlib.util
    from pathlib import Path

    logging = ModuleType('modules.logging_colors')
    logging.logger = SimpleNamespace(exception=lambda *args: None)
    monkeypatch.setitem(sys.modules, 'modules.logging_colors', logging)
    spec = importlib.util.spec_from_file_location(
        'rewrite_test_callbacks', Path(__file__).parents[1] / 'modules' / 'callbacks.py',
    )
    callbacks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(callbacks)
    release_worker = threading.Event()
    worker_finished = threading.Event()

    def worker(callback):
        try:
            callback('first token')
            release_worker.wait()
            callback('next token')
        finally:
            worker_finished.set()

    with callbacks.Iteratorize(worker) as iterator:
        assert next(iterator) == 'first token'
        release_worker.set()
    assert worker_finished.is_set()
    assert not iterator.thread.is_alive()


def test_prompt_planning_passes_generation_state_to_native_counter(native):
    settings = state()
    settings['add_bos_token'] = False
    measured = []

    def count(prompt, actual_state):
        measured.append(actual_state)
        return len(prompt) + (1 if actual_state['add_bos_token'] else 0)

    native.generation.get_rewrite_prompt_length = count
    plan = prepare_prompt('Original sentence.', last_sentence('Original sentence.'), hits(), settings)
    assert plan.token_count == len(plan.prompt)
    assert measured and all(actual_state is settings for actual_state in measured)

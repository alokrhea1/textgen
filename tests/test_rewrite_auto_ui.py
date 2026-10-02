"""Automatic generation staging, opt-in routing, and guarded application."""
import asyncio
import sys
from types import ModuleType, SimpleNamespace

import gradio as gr
import pytest

from modules.sentence_rewrite import automatic
from test_rewrite_ui import assert_locked, make_ui


def auto_args(h, source=None, text='Existing sentence.', left=None, enabled=True, state=None):
    left = text if left is None else left
    if source is None:
        source = text if h.mode == 'notebook' else left
    state = {'temperature': .7, 'enable_thinking': True} if state is None else state
    return [source, h.session, enabled, text, left, 'draft', state,
            'Vary the syntax.', False, False, 4,
            7, 'sentences', 2, .25, 'directional', .2, False, False, 35, .65, .45, .7,
            *h.config]


def progress(document, completed=0, status='Drafting', target='', hits=None,
             replacement=None, context_trimmed=False):
    return SimpleNamespace(document=document, completed=completed, status=status,
                           target=target, hits=hits, replacement=replacement,
                           context_trimmed=context_trimmed,
                           phase='accepted' if status.startswith('Accepted') else
                                 'rewriting' if replacement is not None else 'drafting')


def ready(h, monkeypatch, operation):
    models = ModuleType('modules.models')
    models.load_model_if_idle_unloaded = lambda: None
    utils = ModuleType('modules.utils')
    utils.check_model_loaded = lambda: (True, None)
    monkeypatch.setitem(sys.modules, 'modules.models', models)
    monkeypatch.setitem(sys.modules, 'modules.utils', utils)
    import modules
    monkeypatch.setattr(modules, 'models', models, raising=False)
    monkeypatch.setattr(modules, 'utils', utils, raising=False)
    monkeypatch.setattr(automatic, 'generate_automatic', operation)
    h.shared.args.extensions = []
    h.shared.model = SimpleNamespace()
    if h.session.index is not None:
        h.session.index.assert_current = lambda *args, **kwargs: None
    return models, utils


def unreachable_native(*args, **kwargs):
    raise AssertionError('Automatic generation fell back to ordinary generation')


def dispatch(h, wrapper=unreachable_native):
    return h.c['generation_dispatch'](wrapper)


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_off_delegates_unchanged_without_corpus_model_or_file_guards(make_ui, monkeypatch, mode):
    h = make_ui(mode)
    h.session.index = h.session.encoder = None
    state = {'sampler': object()}
    calls, closed = [], []
    h.shared.args.multi_user = True

    def native(*args):
        calls.append(args)
        try:
            yield ('First output.', '<p>First output.</p>')
            yield ('Final output.', '<p>Final output.</p>')
        finally:
            closed.append(True)

    result = list(dispatch(h, native)(*auto_args(h, source='Native source', enabled=False, state=state)))
    assert calls == ([('Native source', state, 'draft')] if mode == 'notebook' else [('Native source', state)])
    assert calls[0][1] is state
    output = h.shared.gradio['textbox-notebook' if mode == 'notebook' else 'output_textbox']
    html = h.shared.gradio[f'html-{mode}']
    assert result == [{output: 'First output.', html: '<p>First output.</p>'},
                      {output: 'Final output.', html: '<p>Final output.</p>'}]
    assert closed == [True]
    assert not h.session.busy and not h.session.lock.locked()


def test_off_completion_does_not_apply_or_discard_manual_review_proposal(make_ui):
    h = make_ui()
    proposal = {'review': True, 'after': 'Reviewed manual proposal.'}
    h.session.pending = proposal
    h.shared.args.multi_user = True
    result = h.c['generation_commit'](h.session, 'Existing.', 'Existing.', 'draft', {})
    assert result == {h.c['session']: h.session}
    assert h.session.pending is proposal and not h.session.history


@pytest.mark.parametrize('invalid', ['no_index', 'stale_config', 'no_model', 'multi_user'])
def test_enabled_preflight_fails_before_core_and_unlocks(make_ui, monkeypatch, invalid):
    h = make_ui()
    calls = []

    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        yield progress('Must never appear.')

    _, utils = ready(h, monkeypatch, operation)
    args = auto_args(h)
    expected = {'no_index': 'Build the corpus first', 'stale_config': 'Corpus settings changed',
                'no_model': 'No loaded model', 'multi_user': 'disabled in multi-user'}
    if invalid == 'no_index':
        h.session.index = None
    elif invalid == 'stale_config':
        args[-4] = 4
    elif invalid == 'no_model':
        utils.check_model_loaded = lambda: (False, 'No loaded model')
    else:
        h.shared.args.multi_user = True
    events = list(dispatch(h)(*args))
    assert expected[invalid] in events[-1][h.c['status']]
    assert calls == [] and h.session.pending is None
    assert not h.session.busy and not h.session.lock.locked()
    if invalid != 'multi_user':
        assert_locked(h, events[0], True)
        assert_locked(h, events[-1], False)


@pytest.mark.parametrize('failure', ['changed', 'deleted', 'cancelled'])
def test_source_current_preflight_runs_before_models_or_controller(make_ui, monkeypatch, failure):
    h = make_ui()
    calls, checked = [], []

    def operation(*args, **kwargs):
        calls.append('controller')
        yield progress('Must never draft.')

    models, _ = ready(h, monkeypatch, operation)
    models.load_model_if_idle_unloaded = lambda: calls.append('model')

    def assert_current(config, encoder, report, cancel_event):
        checked.append((config, encoder, cancel_event))
        report('Checking corpus sources')
        if failure == 'cancelled':
            cancel_event.set()
            return
        raise ValueError('Corpus source changed or was deleted. Build the corpus again.')

    h.session.index.assert_current = assert_current
    events = list(dispatch(h)(*auto_args(h)))
    assert checked == [(h.session.config, h.session.encoder, h.session.cancel)]
    assert calls == []
    assert ('Stopped automatic generation' if failure == 'cancelled' else 'Corpus source changed or was deleted') in events[-1][h.c['status']]
    assert h.session.pending is None and not h.session.lock.locked()


@pytest.mark.parametrize('nuance', [False, True])
def test_automatic_uses_current_retrieval_guidance_template_and_thinking_settings(make_ui, monkeypatch, nuance):
    h = make_ui()
    searches, core = [], []
    h.session.reranker = SimpleNamespace(load=lambda report: None, release=lambda: None)
    h.session.reranker_config = ('cpu', True)
    original_search = h.session.index.search

    def search(*args, **kwargs):
        searches.append((args, kwargs))
        return original_search(*args, **kwargs)

    h.session.index.search = search

    def operation(source, state, retrieve, **kwargs):
        core.append((source, state, kwargs))
        hits = retrieve('A generated sentence.')
        yield progress(source, status='References found', target='A generated sentence.', hits=hits)

    ready(h, monkeypatch, operation)
    args = auto_args(h)
    args[18] = nuance
    state_before = args[6].copy()
    events = list(dispatch(h)(*args))
    source, passed_state, options = core[0]
    assert source == args[0]
    assert passed_state is not args[6] and args[6] == state_before
    assert passed_state['enable_thinking'] is False
    assert options == {'guidance': 'Vary the syntax.', 'use_template': False,
                       'cancel_event': h.session.cancel, 'max_sentences': 4}
    assert searches[0][0][0] == 'A generated sentence.'
    values = searches[0][1]
    assert {name: values[name] for name in ('top_k', 'length_mode', 'sentence_count', 'token_tolerance',
                                            'scoring', 'diversity', 'exclude_exact', 'rerank_pool',
                                            'min_length_ratio', 'min_semantic_score', 'max_contradiction_score')} == {
        'top_k': 7, 'length_mode': 'sentences', 'sentence_count': 2, 'token_tolerance': .25,
        'scoring': 'directional', 'diversity': .2, 'exclude_exact': False, 'rerank_pool': 35,
        'min_length_ratio': .65, 'min_semantic_score': .45 if nuance else 0.0,
        'max_contradiction_score': .7 if nuance else 1.0}
    assert h.c['seed'] not in h.c['generation_inputs']
    assert h.c['review'] not in h.c['generation_inputs']
    assert h.c['seed'] not in h.c['generation_commit_inputs']
    assert any('Reference example.' in event.get(h.c['references'], '') for event in events)


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_accepted_snapshots_are_staged_then_fresh_committed_with_one_step_undo(make_ui, monkeypatch, mode):
    h = make_ui(mode)
    text = 'Original document.' if mode == 'notebook' else ''
    left = text if mode == 'notebook' else 'Input document.'
    args = auto_args(h, text=text, left=left)
    after = args[0] + ' First accepted. Second accepted.'

    def operation(source, state, retrieve, **kwargs):
        yield progress(source, status='Drafting sentence 1')
        yield progress(source + ' First accepted.', completed=1, status='Accepted sentence 1')
        yield progress(after, completed=2, status='Accepted sentence 2', context_trimmed=True)

    ready(h, monkeypatch, operation)
    events = list(dispatch(h)(*args))
    output = h.shared.gradio['textbox-notebook' if mode == 'notebook' else 'output_textbox']
    for event in events:
        if output in event:
            assert isinstance(event[output], dict) and 'value' not in event[output]
    assert any(event.get(h.c['auto_preview']) == after for event in events)
    assert h.session.pending['after'] == after and h.session.pending['completed'] == 2
    assert h.session.pending['review'] is False
    assert h.session.history == []
    state = {'existing_setting': 9}
    committed = h.c['generation_commit'](h.session, text, left, 'draft', state)
    assert committed[output] == after
    assert committed[h.shared.gradio['interface_state']]['existing_setting'] == 9
    assert state == {'existing_setting': 9}
    assert 'Applied 2 completed rewrite(s)' in committed[h.c['status']]
    assert 'Older notebook context was trimmed' in committed[h.c['status']]
    assert h.c['seed'] not in committed
    assert len(h.session.history) == 1 and h.session.pending is None
    if mode == 'notebook':
        assert committed[h.shared.gradio['last_input-notebook']] == text
    undone = h.callbacks['undo'].fn(h.session, after, left, 'draft', state)
    assert undone[output] == text
    assert h.session.history == []


@pytest.mark.parametrize('changed', ['text', 'left', 'prompt'])
def test_stale_edit_rejects_automatic_commit(make_ui, monkeypatch, changed):
    h = make_ui('default')

    def operation(source, state, retrieve, **kwargs):
        yield progress(source + ' Accepted sentence.', completed=1, status='Accepted')

    ready(h, monkeypatch, operation)
    list(dispatch(h)(*auto_args(h, text='Old output.', left='Original input.')))
    values = {'text': 'Old output.', 'left': 'Original input.', 'prompt': 'draft'}
    values[changed] += ' edited'
    result = h.c['generation_commit'](h.session, values['text'], values['left'], values['prompt'], {})
    assert 'Automatic generation was not applied' in result[h.c['status']]
    assert h.shared.gradio['output_textbox'] not in result
    assert h.session.pending is None and h.session.history == []


@pytest.mark.parametrize('failure', ['stop', 'error'])
def test_stop_or_failure_retains_accepted_sentences_and_discards_provisional_draft(make_ui, monkeypatch, failure):
    h = make_ui()
    after = 'Existing sentence. Accepted sentence.'

    def operation(source, state, retrieve, **kwargs):
        yield progress(after, completed=1, status='Accepted sentence 1')
        yield progress(after, completed=1, status='Generating rewrite 2', replacement='Unfinished')
        if failure == 'stop':
            h.session.cancel.set()
            raise InterruptedError('User stopped the run')
        raise ValueError('No qualifying references')

    ready(h, monkeypatch, operation)
    events = list(dispatch(h)(*auto_args(h)))
    assert h.session.pending['after'] == after and h.session.pending['completed'] == 1
    assert 'Unfinished' not in h.session.pending['after']
    assert ('Stopped automatic generation' if failure == 'stop' else 'FAILED: No qualifying references') in events[-1][h.c['status']]
    assert 'unfinished drafts were discarded' in events[-1][h.c['status']]
    assert_locked(h, events[-1], False)
    committed = h.c['generation_commit'](h.session, 'Existing sentence.', 'Existing sentence.', 'draft', {})
    assert committed[h.shared.gradio['textbox-notebook']] == after
    assert failure == 'stop' or 'FAILED: No qualifying references' in committed[h.c['status']]


def test_failure_before_acceptance_preserves_document_and_retry_clears_cancel(make_ui, monkeypatch):
    h = make_ui()
    calls = []

    def operation(source, state, retrieve, **kwargs):
        calls.append(kwargs['cancel_event'].is_set())
        if len(calls) == 1:
            h.session.cancel.set()
            raise InterruptedError('Stopped before acceptance')
        yield progress(source + ' Accepted.', completed=1, status='Accepted')

    ready(h, monkeypatch, operation)
    failed = list(dispatch(h)(*auto_args(h)))
    assert h.session.pending is None
    assert 'Notebook text is unchanged' in failed[-1][h.c['status']]
    list(dispatch(h)(*auto_args(h)))
    assert calls == [False, False]
    assert h.session.pending['completed'] == 1 and not h.session.cancel.is_set()


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_automatic_locks_native_controls_and_stop_is_local(make_ui, monkeypatch, mode):
    h = make_ui(mode)
    with h.blocks:
        for name in [f'Generate-{mode}', f'Stop-{mode}', f'get_logits-{mode}',
                     'Regenerate-notebook' if mode == 'notebook' else 'Continue-default']:
            h.shared.gradio[name] = gr.Button(name)
        h.module.create_event_handlers(mode)
    functions = h.blocks.fns.values() if isinstance(h.blocks.fns, dict) else h.blocks.fns
    callbacks = {item.fn.__name__: item for item in functions}
    other = h.module.RewriteSession()
    closed = []

    def operation(source, state, retrieve, **kwargs):
        try:
            yield progress(source, status='Drafting')
            if kwargs['cancel_event'].is_set():
                raise InterruptedError('Stopped')
            raise AssertionError('Local Stop did not cancel the operation')
        finally:
            closed.append(True)

    ready(h, monkeypatch, operation)
    generator = dispatch(h)(*auto_args(h))
    first = next(generator)
    assert h.session.busy and h.session.phase == 'automatic'
    assert_locked(h, first, True)
    for name in [f'Generate-{mode}', f'get_logits-{mode}', f'prompt_menu-{mode}',
                 'textbox-notebook' if mode == 'notebook' else 'output_textbox',
                 'Regenerate-notebook' if mode == 'notebook' else 'Continue-default']:
        assert first[h.shared.gradio[name]]['interactive'] is False
    assert first[h.shared.gradio[f'Stop-{mode}']]['interactive'] is True
    next(generator)
    stop_response = callbacks['stop'].fn(h.session)
    assert isinstance(stop_response, dict) and 'value' not in stop_response
    assert h.session.cancel.is_set() and not other.cancel.is_set()
    final = list(generator)[-1]
    assert 'Stopped automatic generation' in final[h.c['status']]
    # An asynchronous Stop response may be delivered after this final stream
    # update. Actual Gradio processing must preserve its existing status value.
    late_response = asyncio.run(h.blocks.postprocess_data(callbacks['stop'], stop_response, None))
    assert len(late_response) == 1 and 'value' not in late_response[0]
    assert_locked(h, final, False)
    assert final[h.shared.gradio[f'Generate-{mode}']]['interactive'] is True
    assert closed == [True] and not h.session.lock.locked()


def test_native_stop_cancels_only_active_automatic_session(make_ui):
    h = make_ui()
    assert isinstance(h.module.stop_notebook_generation(h.session), dict)
    assert not h.session.cancel.is_set()
    h.session.busy, h.session.phase = True, 'rewrite'
    h.module.stop_notebook_generation(h.session)
    assert not h.session.cancel.is_set()
    h.session.phase = 'automatic'
    update = h.module.stop_notebook_generation(h.session)
    assert isinstance(update, dict) and 'value' not in update
    assert h.session.cancel.is_set()


def test_complete_progress_keeps_the_last_proposed_replacement_visible(make_ui, monkeypatch):
    h = make_ui()
    after = 'Existing sentence. Accepted sentence.'

    def operation(source, state, retrieve, **kwargs):
        yield automatic.AutomaticProgress(after, 'Accepted sentence 1', phase='accepted',
                                          replacement='Accepted sentence.', completed=1)
        # The real complete event defaults replacement to the empty string.
        yield automatic.AutomaticProgress(after, 'Automatic generation complete',
                                          phase='complete', completed=1, done=True)

    ready(h, monkeypatch, operation)
    events = list(dispatch(h)(*auto_args(h)))
    accepted = next(event for event in events if event.get(h.c['result']) == 'Accepted sentence.')
    assert accepted[h.c['result']] == 'Accepted sentence.'
    complete = next(event for event in events
                    if event.get(h.c['status'], '').startswith('Automatic generation complete'))
    assert h.c['result'] not in complete
    visible = ''
    for event in events:
        if h.c['result'] in event:
            visible = event[h.c['result']]
    assert visible == 'Accepted sentence.'


def test_generator_close_releases_operation_session_lock_and_busy_state(make_ui, monkeypatch):
    h = make_ui()
    closed = []

    def operation(source, state, retrieve, **kwargs):
        try:
            yield progress(source + ' Accepted.', completed=1, status='Accepted')
            raise AssertionError('Closed generator consumed more model output')
        finally:
            closed.append(True)

    ready(h, monkeypatch, operation)
    generator = dispatch(h)(*auto_args(h))
    next(generator)
    next(generator)
    assert h.session.lock.locked()
    generator.close()
    assert closed == [True]
    assert not h.session.busy and not h.session.lock.locked()
    assert h.session.pending is None
    assert h.session.cancel.is_set()


def test_new_generation_progress_outputs_pass_gradio_postprocessing(make_ui, monkeypatch):
    h = make_ui()

    def operation(source, state, retrieve, **kwargs):
        yield progress(source + ' Accepted.', completed=1, status='Accepted')

    ready(h, monkeypatch, operation)
    with h.blocks:
        event = h.c['rewrite'].click(dispatch(h), [h.shared.gradio['textbox-notebook'], *h.c['generation_inputs']],
                                      h.c['generation_outputs'], api_name=False)
    callback = h.blocks.fns[event['id']]
    generator = callback.fn(*auto_args(h))
    try:
        update = next(generator)
        processed = asyncio.run(h.blocks.postprocess_data(callback, update, None))
        assert len(processed) == len(h.c['generation_outputs'])
    finally:
        generator.close()

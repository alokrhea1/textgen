"""Real Gradio callback wiring with fake retrieval/generation, no model downloads."""
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import gradio as gr
import pytest


@pytest.fixture
def make_ui(monkeypatch, tmp_path):
    import modules

    def factory(mode='notebook', multi_user=False):
        shared = ModuleType('modules.shared')
        shared.args = SimpleNamespace(multi_user=multi_user)
        shared.user_data_dir = tmp_path
        shared.model_name = 'fake-model'
        shared.tokenizer = SimpleNamespace(name_or_path='fake-tokenizer')
        shared.gradio = {}
        shared.input_elements = ['native-setting']
        ui = ModuleType('modules.ui')
        ui.gather_interface_values = lambda value: {'native-setting': value}
        logging = ModuleType('modules.logging_colors')
        logging.logger = SimpleNamespace(exception=lambda *args: None)
        html = ModuleType('modules.html_generator')
        html.generate_basic_html = lambda text: '<p>' + text + '</p>'
        generation = ModuleType('modules.text_generation')
        generation.get_encoded_length = len
        generation.stop_everything_event = lambda: None
        for name, stub in [('shared', shared), ('ui', ui), ('logging_colors', logging),
                           ('html_generator', html), ('text_generation', generation)]:
            monkeypatch.setitem(sys.modules, 'modules.' + name, stub)
            monkeypatch.setattr(modules, name, stub, raising=False)
        spec = importlib.util.spec_from_file_location(
            'rewrite_ui_under_test', Path(__file__).parents[1] / 'modules/ui_sentence_rewrite.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        with gr.Blocks() as blocks:
            for name in ['textbox-notebook', 'output_textbox', 'textbox-default',
                         'prompt_menu-notebook', 'prompt_menu-default', 'native-setting']:
                shared.gradio[name] = gr.Textbox()
            shared.gradio['html-notebook'] = gr.HTML()
            shared.gradio['html-default'] = gr.HTML()
            shared.gradio['interface_state'] = gr.State({})
            shared.gradio['last_input-notebook'] = gr.State('')
            module.create_ui(mode)
            module.create_event_handlers(mode)
        functions = blocks.fns.values() if isinstance(blocks.fns, dict) else blocks.fns
        callbacks = {item.fn.__name__: item for item in functions}
        c, controls = shared.gradio['rewrite-ui-' + mode]
        config = ['references.txt', True, module.DEFAULT_MODEL, '', 'cpu', 256, 16, True, 3, 'conservative', False]
        session = module.RewriteSession()
        session.config, session.embedding_config = module._configs(config)
        session.encoder = SimpleNamespace(release=lambda: None)
        hit = SimpleNamespace(score=.9, sentence_count=1, token_count=4,
                              source='references.txt', start=0, end=18, text='Reference example.')
        session.index = SimpleNamespace(path=tmp_path / 'fake-index', search=lambda *args, **kwargs: [hit], clear=lambda: None)
        calls = []

        def engine(*args, **kwargs):
            calls.append((args, kwargs))
            yield SimpleNamespace(text='Better', status='Generating', done=False, plan=None)
            yield SimpleNamespace(text='Better sentence.', status='Complete', done=True, plan=None)

        monkeypatch.setattr(module, 'generate_rewrite', engine)
        return SimpleNamespace(module=module, shared=shared, blocks=blocks, callbacks=callbacks,
                               c=c, controls=controls, config=config, session=session, calls=calls,
                               mode=mode, gather=ui.gather_interface_values)
    return factory


def run_args(h, text='Earlier sentence. Original sentence. trailing', left=None, review=False, seed=''):
    left = text if left is None else left
    return [h.session, text, left, 'draft', {'temperature': .7}, seed, '', True, review, False,
            5, 'sentences', 1, .15, 'symmetric', 0, True, False, 50, *h.config]


def assert_locked(h, update, locked):
    for component in h.controls:
        if component in [h.c['stop'], h.c['apply'], h.c['undo']]:
            continue
        assert update[component]['interactive'] is (not locked)


def test_native_gather_only_on_explicit_rewrite_click(make_ui):
    h = make_ui()
    gather = h.callbacks['<lambda>']
    assert gather.fn is h.gather
    dependencies = h.blocks.config['dependencies']
    gather_edges = [edge for edge in dependencies if edge['id'] == gather._id]
    assert len(gather_edges) == 1
    assert gather_edges[0]['targets'] == [(h.c['rewrite']._id, 'click')]
    assert h.callbacks['search'].inputs[0] is h.c['session']
    assert h.callbacks['rewrite'].outputs == h.callbacks['search'].outputs
    assert all(h.shared.gradio['textbox-notebook'] not in callback.outputs
               for name, callback in h.callbacks.items() if name in ['build', 'search', 'rewrite'])


@pytest.mark.parametrize('callback', ['search', 'rewrite'])
def test_controls_and_source_preserved_until_commit(make_ui, callback):
    h = make_ui()
    events = list(h.callbacks[callback].fn(*run_args(h)))
    assert_locked(h, events[0], True)
    assert_locked(h, events[-1], False)
    assert not h.session.busy and not h.session.lock.locked()
    assert all(h.shared.gradio['textbox-notebook'] not in event for event in events)
    assert bool(h.session.pending) == (callback == 'rewrite')
    assert bool(h.calls) == (callback == 'rewrite')
    assert events[0][h.c['stop']]['interactive'] == (callback == 'rewrite')


@pytest.mark.parametrize('callback', ['search', 'rewrite'])
def test_failure_and_retry_restore_controls(make_ui, callback):
    h = make_ui()
    h.session.index.search = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('bad corpus'))
    events = list(h.callbacks[callback].fn(*run_args(h)))
    assert 'FAILED' in events[-1][h.c['status']]
    assert_locked(h, events[-1], False)
    assert h.session.pending is None and not h.session.lock.locked()
    h.session.index.search = lambda *args, **kwargs: []
    retried = list(h.callbacks[callback].fn(*run_args(h)))
    assert 'FAILED' not in retried[-1][h.c['status']]


@pytest.mark.parametrize('callback', ['build', 'search', 'rewrite'])
def test_generator_close_releases_busy_lock(make_ui, callback):
    h = make_ui()
    args = [h.session, *h.config, False] if callback == 'build' else run_args(h)
    generator = h.callbacks[callback].fn(*args)
    next(generator)
    assert h.session.busy and h.session.lock.locked()
    generator.close()
    assert not h.session.busy and not h.session.lock.locked()
    assert h.session.phase == ''


def test_build_failure_retry_and_progress(make_ui, monkeypatch):
    h = make_ui()
    attempts, progress = [], []

    class Encoder:
        def __init__(self, config):
            pass

        def load(self, report):
            report('Loading fake encoder')

        def release(self):
            pass

    class Index:
        def __init__(self, path):
            self.path = path

        def build(self, config, encoder, report, force):
            attempts.append(force)
            report('Indexing fake corpus')
            if len(attempts) == 1:
                raise ValueError('unreadable corpus')
            return dict(files=1, candidates=1, occurrences=1,
                        cleanup=dict(mode=config.cleanup, join_hyphenated_lines=config.join_hyphenated_lines,
                                     counts={'soft_hyphens_removed': 2}, samples=[]))

    h.session.encoder = None
    monkeypatch.setattr(h.module, 'LateInteractionEncoder', Encoder)
    monkeypatch.setattr(h.module, 'CorpusIndex', Index)
    callback = h.callbacks['build'].fn
    first = list(callback(h.session, *h.config, True, progress=lambda *args, **kwargs: progress.append(kwargs)))
    assert 'INDEX BUILD FAILED' in first[-1][h.c['status']]
    assert_locked(h, first[-1], False)
    second = list(callback(h.session, *h.config, True, progress=lambda *args, **kwargs: progress.append(kwargs)))
    assert 'Corpus ready' in second[-1][h.c['status']]
    assert 'soft hyphens removed: 2' in second[-1][h.c['cleanup_report']]
    assert_locked(h, second[-1], False)
    assert len(progress) == 4 and attempts == [True, True]
    assert not h.session.lock.locked()


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_commit_clears_seed_updates_state_and_undo(make_ui, mode):
    h = make_ui(mode)
    text = 'Earlier sentence. Original sentence. trailing'
    args = run_args(h, text=text, seed='Seed sentence.')
    list(h.callbacks['rewrite'].fn(*args))
    committed = h.callbacks['automatic_commit'].fn(h.session, text, text, 'draft', {'keep': 42}, 'Seed sentence.')
    notebook = h.shared.gradio['textbox-notebook' if mode == 'notebook' else 'output_textbox']
    assert committed[notebook] == 'Earlier sentence. Better sentence. trailing'
    assert committed[h.c['seed']] == ''
    assert committed[h.shared.gradio['interface_state']]['keep'] == 42
    assert h.session.pending is None
    assert h.callbacks['undo'].fn(h.session, 'edited', text, 'draft', {})[h.c['status']].startswith('The notebook changed')
    undone = h.callbacks['undo'].fn(h.session, committed[notebook], text, 'draft', {})
    assert undone[notebook] == text
    assert not h.session.history


@pytest.mark.parametrize('changed', ['text', 'left', 'prompt'])
def test_stale_commit_rejected(make_ui, changed):
    h = make_ui()
    args = run_args(h)
    list(h.callbacks['rewrite'].fn(*args))
    values = [args[1], args[2], 'draft']
    values[['text', 'left', 'prompt'].index(changed)] = 'changed'
    result = h.callbacks['commit'].fn(h.session, *values, {})
    assert 'not applied' in result[h.c['status']]
    assert h.session.pending is None and not h.session.history
    assert h.shared.gradio['textbox-notebook'] not in result


def test_review_prevents_automatic_commit(make_ui):
    h = make_ui()
    args = run_args(h, review=True)
    list(h.callbacks['rewrite'].fn(*args))
    callback = h.callbacks['automatic_commit']
    result = callback.fn(h.session, args[1], args[2], 'draft', {})
    assert result == {h.c['session']: h.session}
    processed = asyncio.run(h.blocks.postprocess_data(callback, result, None))
    assert len(processed) == len(callback.outputs)
    assert h.session.pending is not None
    assert h.shared.gradio['textbox-notebook'] in h.callbacks['commit'].fn(h.session, args[1], args[2], 'draft', {})


def test_default_empty_output_undo_restores_empty_output(make_ui):
    h = make_ui('default')
    left = 'Input sentence.'
    list(h.callbacks['rewrite'].fn(*run_args(h, text='', left=left)))
    result = h.callbacks['automatic_commit'].fn(h.session, '', left, 'draft', {})
    output = h.shared.gradio['output_textbox']
    assert result[output] == 'Better sentence.'
    undone = h.callbacks['undo'].fn(h.session, result[output], left, 'draft', {})
    assert undone[output] == ''


def test_multiuser_disables_controls_and_guards_callbacks(make_ui):
    h = make_ui(multi_user=True)
    assert all(component.interactive is False for component in h.controls)
    for callback in ['search', 'rewrite']:
        events = list(h.callbacks[callback].fn(*run_args(h)))
        assert 'disabled in multi-user' in events[-1][h.c['status']]
        assert all(events[-1][component]['interactive'] is False for component in h.controls)
    for callback in ['commit', 'undo']:
        with pytest.raises(ValueError, match='multi-user'):
            h.callbacks[callback].fn(h.session, '', '', 'draft', {})
    for callback in ['clear', 'stop']:
        with pytest.raises(ValueError, match='multi-user'):
            h.callbacks[callback].fn(h.session)


def test_generation_failure_after_preview_preserves_source_and_retry(make_ui, monkeypatch):
    h = make_ui()
    successful_engine = h.module.generate_rewrite

    def failing_engine(*args, **kwargs):
        yield SimpleNamespace(text='Partial preview', status='Generating', done=False, plan=None)
        raise ValueError('model failed')

    monkeypatch.setattr(h.module, 'generate_rewrite', failing_engine)
    events = list(h.callbacks['rewrite'].fn(*run_args(h)))
    assert any(event.get(h.c['result']) == 'Partial preview' for event in events)
    assert 'REWRITE FAILED: model failed' in events[-1][h.c['status']]
    assert_locked(h, events[-1], False)
    assert h.session.pending is None and not h.session.lock.locked()
    assert all(h.shared.gradio['textbox-notebook'] not in event for event in events)
    monkeypatch.setattr(h.module, 'generate_rewrite', successful_engine)
    list(h.callbacks['rewrite'].fn(*run_args(h)))
    assert h.session.pending is not None


def test_changed_settings_require_rebuild_before_generation(make_ui):
    h = make_ui()
    args = run_args(h)
    args[-3] = 4
    events = list(h.callbacks['rewrite'].fn(*args))
    assert 'Corpus settings changed' in events[-1][h.c['status']]
    assert not h.calls and h.session.pending is None
    assert_locked(h, events[-1], False)


def test_no_pending_commit_passes_actual_gradio_postprocessing(make_ui):
    h = make_ui()
    callback = h.callbacks['automatic_commit']
    result = callback.fn(h.session, '', '', 'draft', {})
    processed = asyncio.run(h.blocks.postprocess_data(callback, result, None))
    assert len(processed) == len(callback.outputs)
    assert result == {h.c['session']: h.session}


def test_html_commit_and_undo_escape_markup_preserve_raw_text(make_ui, monkeypatch):
    import html

    h = make_ui()
    original = '<b>Earlier sentence.</b> Original sentence.'
    replacement = '<img src=x onerror=alert(1)> Better sentence.'

    def engine(*args, **kwargs):
        yield SimpleNamespace(text=replacement, status='Complete', done=True, plan=None)

    monkeypatch.setattr(h.module, 'generate_rewrite', engine)
    list(h.callbacks['rewrite'].fn(*run_args(h, text=original)))
    result = h.callbacks['commit'].fn(h.session, original, original, 'draft', {})
    raw = result[h.shared.gradio['textbox-notebook']]
    assert replacement in raw
    assert result[h.shared.gradio['html-notebook']] == '<p>' + html.escape(raw) + '</p>'
    undone = h.callbacks['undo'].fn(h.session, raw, original, 'draft', {})
    assert undone[h.shared.gradio['textbox-notebook']] == original
    assert undone[h.shared.gradio['html-notebook']] == '<p>' + html.escape(original) + '</p>'


def test_stop_is_session_local_and_retry_clears_cancel(make_ui, monkeypatch):
    h = make_ui()
    other = h.module.RewriteSession()
    global_stops = []
    monkeypatch.setattr(sys.modules['modules.text_generation'], 'stop_everything_event',
                        lambda: global_stops.append(True))
    h.session.busy, h.session.phase = True, 'rewrite'
    assert 'Stopping' in h.callbacks['stop'].fn(h.session)
    assert h.session.cancel.is_set() and not other.cancel.is_set()
    assert global_stops == []
    h.session.busy, h.session.phase = False, ''
    list(h.callbacks['rewrite'].fn(*run_args(h)))
    assert not h.session.cancel.is_set() and h.session.pending is not None


def test_clear_retains_disk_cache_used_by_another_session(make_ui):
    h = make_ui()
    other = h.module.RewriteSession()
    clears, releases = [], []
    index = h.session.index
    index.clear = lambda: clears.append(True)
    h.session.encoder.release = lambda: releases.append(True)
    other.index = index
    result = h.callbacks['clear'].fn(h.session)
    assert clears == [] and releases == [True]
    assert 'another open Rewrite tab' in result[h.c['status']]
    assert other.index is index and h.session.index is None
    other.encoder = SimpleNamespace(release=lambda: None)
    h.callbacks['clear'].fn(other)
    assert clears == [True]


def test_failed_new_encoder_load_releases_resources(make_ui, monkeypatch):
    h = make_ui()
    previous = h.session.index
    releases = []

    class Encoder:
        def __init__(self, config):
            pass

        def load(self, report):
            raise ValueError('checkpoint failed')

        def release(self):
            releases.append(True)

    monkeypatch.setattr(h.module, 'LateInteractionEncoder', Encoder)
    result = list(h.callbacks['build'].fn(h.session, *h.config, True))[-1]
    assert 'checkpoint failed' in result[h.c['status']]
    assert releases == [True]
    assert h.session.index is previous and h.session.build_path is None
    assert not h.session.lock.locked()


def test_live_session_progress_update_passes_gradio_postprocessing(make_ui):
    h = make_ui()
    callback = h.callbacks['rewrite']
    generator = callback.fn(*run_args(h))
    try:
        update = next(generator)
        processed = asyncio.run(h.blocks.postprocess_data(callback, update, None))
        assert len(processed) == len(callback.outputs)
        assert h.session.lock.locked()
    finally:
        generator.close()
    assert not h.session.lock.locked()


@pytest.mark.parametrize('callback', ['automatic_commit', 'commit'])
def test_fresh_seed_survives_committing_previous_rewrite(make_ui, callback):
    h = make_ui()
    args = run_args(h, seed='Previous seed sentence.')
    list(h.callbacks['rewrite'].fn(*args))
    assert h.session.pending['seed'] == 'Previous seed sentence.'
    result = h.callbacks[callback].fn(h.session, args[1], args[2], 'draft', {}, 'Next seed sentence.')
    assert h.shared.gradio['textbox-notebook'] in result
    assert h.c['seed'] not in result
    assert h.callbacks[callback].inputs[-1] is h.c['seed']


@pytest.mark.parametrize('callback', ['automatic_commit', 'commit'])
def test_commit_clears_only_unchanged_used_seed(make_ui, callback):
    h = make_ui()
    args = run_args(h, seed='Used seed sentence.')
    list(h.callbacks['rewrite'].fn(*args))
    result = h.callbacks[callback].fn(h.session, args[1], args[2], 'draft', {}, 'Used seed sentence.')
    assert result[h.c['seed']] == ''


@pytest.mark.parametrize('enabled', [False, True])
def test_thinking_checkbox_overrides_copied_generation_state_only(make_ui, enabled):
    h = make_ui()
    args = run_args(h)
    args[4]['enable_thinking'] = not enabled
    args[9] = enabled
    original_state = args[4].copy()
    list(h.callbacks['rewrite'].fn(*args))
    passed_state = h.calls[0][0][2]
    assert passed_state['enable_thinking'] is enabled
    assert passed_state is not args[4] and args[4] == original_state
    assert h.c['thinking'].value is False
    assert h.callbacks['rewrite'].inputs[9] is h.c['thinking']
    assert h.callbacks['search'].inputs[9] is h.c['thinking']


def test_preview_never_generates_even_when_thinking_enabled(make_ui):
    h = make_ui()
    args = run_args(h)
    args[9] = True
    list(h.callbacks['search'].fn(*args))
    assert h.calls == []


def test_evidence_balanced_score_components_and_legacy_fallback(make_ui):
    h = make_ui()
    fields = dict(score=.8, late_score=.7, nuance_score=.9, sentence_count=1, token_count=5,
                  source='source.txt', start=3, end=18, text='Example sentence.')
    rich = SimpleNamespace(**fields, semantic_score=.85, entailment_score=.75, contradiction_score=.05)
    evidence = h.module._evidence([rich])
    assert 'Nuance score 0.9000' in evidence and 'Late interaction 0.7000' in evidence
    assert 'Semantic similarity 0.8500' in evidence
    assert 'Entailment 0.7500' in evidence and 'Contradiction 0.0500' in evidence
    legacy = h.module._evidence([SimpleNamespace(**fields)])
    assert 'Nuance score' in legacy and 'Semantic similarity' not in legacy
    plain_fields = dict(fields)
    del plain_fields['nuance_score']
    plain = h.module._evidence([SimpleNamespace(**plain_fields)])
    assert 'Late interaction 0.8000' in plain and 'Nuance score' not in plain


def test_cleanup_defaults_controls_and_config_wiring(make_ui):
    h = make_ui()
    assert h.c['cleanup'].value == 'conservative'
    assert h.c['join_hyphenated_lines'].value is False
    assert h.c['cleanup_report'].interactive is False
    assert h.c['cleanup_report'] not in h.controls
    assert h.callbacks['build'].inputs[-3:-1] == [h.c['cleanup'], h.c['join_hyphenated_lines']]
    for callback in ['search', 'rewrite']:
        assert h.callbacks[callback].inputs[-2:] == [h.c['cleanup'], h.c['join_hyphenated_lines']]
    config = list(h.config)
    config[-2:] = ['scanned_book', True]
    corpus, embedding = h.module._configs(config)
    assert corpus.cleanup == 'scanned_book' and corpus.join_hyphenated_lines is True


def test_cleanup_report_counts_and_samples_are_bounded(make_ui):
    h = make_ui()
    report = dict(mode='scanned_book', join_hyphenated_lines=True, files_changed=2,
                  counts={'number_lines_removed': 7, 'soft_hyphens_removed': 3},
                  samples=[dict(source='book.txt', before='before ' + 'x' * 1000,
                                after='after ' + 'y' * 1000) for _ in range(8)])
    text = h.module._cleanup_report(report)
    assert 'number lines removed: 7' in text and 'soft hyphens removed: 3' in text
    assert 'Before:' in text and 'After:' in text
    assert text.count('Sample ') == 3 and 'Additional samples omitted.' in text
    assert 'x' * 241 not in text and 'y' * 241 not in text
    assert 'original decoded characters' in text


def test_cleanup_preference_restoration(make_ui):
    import json

    h = make_ui()
    path = h.module._settings_path('notebook')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cleanup='scanned_book', join_hyphenated_lines=True)))
    defaults = h.module._defaults('notebook')
    assert defaults['cleanup'] == 'scanned_book' and defaults['join_hyphenated_lines'] is True

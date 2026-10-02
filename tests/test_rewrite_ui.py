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
        config = ['references.txt', True, module.DEFAULT_MODEL, '', 'cpu', 256, 16, True, 3, 'conservative', False, 'balanced']
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
            5, 'sentences', 1, .15, 'symmetric', 0, True, False, 50, .5, .3, .8, *h.config]


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
                                     counts={'soft_hyphens_removed': 2}, samples=[]),
                        quality=dict(policy=config.quality_policy, analyzed_spans=2, excluded_spans=1,
                                     excluded_windows=1, flag_counts={'suspected_fragment': 1},
                                     excluded_reasons={'suspected_fragment': 1}, samples=[]))

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
    assert 'Sentence spans excluded: 1' in second[-1][h.c['quality_report']]
    assert 'Excluded — suspected fragment: 1' in second[-1][h.c['quality_report']]
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
    args[-4] = 4
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
    assert result[h.c['quality_report']] == ''
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


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_reference_quality_flags_visible_as_plain_text_and_short_is_a_note(make_ui, mode):
    h = make_ui(mode)
    fields = dict(score=.8, sentence_count=1, token_count=5,
                  source='<script>alert(1)</script>.txt', start=3, end=18, text='<b>Short fragment.</b>')
    flagged = SimpleNamespace(**fields, quality_flags=('short_reference', 'suspicious_boundary'))
    h.config[-1] = 'off'
    h.session.config, h.session.embedding_config = h.module._configs(h.config)
    h.session.index.search = lambda *args, **kwargs: [flagged]
    events = list(h.callbacks['search'].fn(*run_args(h)))
    evidence = next(event[h.c['references']] for event in events if event.get(h.c['references']))
    assert isinstance(h.c['references'], gr.Textbox) and h.c['references'].interactive is False
    assert 'Quality note: A short reference; retained for suitably short queries.' in evidence
    assert 'Quality warning (suspected damage; inspect source):' in evidence
    assert h.module.QUALITY_REASONS['suspicious_boundary'] in evidence
    assert '<script>alert(1)</script>.txt [3:18]' in evidence
    assert '<b>Short fragment.</b>' in evidence
    short_only = h.module._evidence([SimpleNamespace(**fields, quality_flags=('short_reference',))])
    assert 'Quality note:' in short_only and 'Quality warning' not in short_only
    legacy = h.module._evidence([SimpleNamespace(**fields)])
    assert 'Quality note:' not in legacy and 'Quality warning' not in legacy


def test_reference_unknown_quality_flags_bounded(make_ui):
    h = make_ui()
    hit = SimpleNamespace(score=.8, sentence_count=1, token_count=5, source='book.txt', start=0,
                          end=5, text='Text.', quality_flags=tuple('x' * 1000 + str(i) for i in range(10)))
    evidence = h.module._evidence([hit])
    assert 'x' * 241 not in evidence
    assert evidence.count('Quality flag:') == 8
    assert 'Additional quality flags omitted.' in evidence


def test_cleanup_defaults_controls_and_config_wiring(make_ui):
    h = make_ui()
    assert h.c['cleanup'].value == 'conservative'
    assert h.c['join_hyphenated_lines'].value is False
    assert h.c['cleanup_report'].interactive is False
    assert h.c['cleanup_report'] not in h.controls
    assert h.callbacks['build'].inputs[-4:-1] == [h.c['cleanup'], h.c['join_hyphenated_lines'], h.c['quality_policy']]
    for callback in ['search', 'rewrite']:
        assert h.callbacks[callback].inputs[-3:] == [h.c['cleanup'], h.c['join_hyphenated_lines'], h.c['quality_policy']]
    config = list(h.config)
    config[-3:] = ['scanned_book', True, 'off']
    corpus, embedding = h.module._configs(config)
    assert corpus.cleanup == 'scanned_book' and corpus.join_hyphenated_lines is True
    assert corpus.quality_policy == 'off'


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


@pytest.mark.parametrize('mode', ['notebook', 'default'])
@pytest.mark.parametrize('callback', ['search', 'rewrite'])
@pytest.mark.parametrize('nuance', [False, True])
def test_quality_retrieval_controls_forwarded_and_locked(make_ui, mode, callback, nuance):
    h = make_ui(mode)
    captured = []
    search = h.session.index.search

    def capture(*args, **kwargs):
        captured.append(kwargs)
        return search(*args, **kwargs)

    h.session.index.search = capture
    h.session.reranker = SimpleNamespace(load=lambda progress: None, release=lambda: None)
    h.session.reranker_config = ('cpu', True)
    args = run_args(h)
    args[17], args[19], args[20], args[21] = nuance, .65, .45, .7
    events = list(h.callbacks[callback].fn(*args))
    assert captured[0]['min_length_ratio'] == .65
    assert captured[0]['min_semantic_score'] == (.45 if nuance else 0.0)
    assert captured[0]['max_contradiction_score'] == (.7 if nuance else 1.0)
    assert h.c['quality_policy'].value == 'balanced'
    assert h.c['min_length_ratio'].value == .5
    assert h.c['min_semantic_score'].value == .3
    assert h.c['max_contradiction_score'].value == .8
    assert h.callbacks[callback].inputs[19:22] == [h.c['min_length_ratio'], h.c['min_semantic_score'],
                                                h.c['max_contradiction_score']]
    assert_locked(h, events[0], True)
    assert_locked(h, events[-1], False)
    for name in ['quality_policy', 'min_length_ratio', 'min_semantic_score', 'max_contradiction_score']:
        assert h.c[name] in h.controls
    assert any('Retrieved 1 of requested 5 references.' in event.get(h.c['status'], '') for event in events)


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_quality_report_is_plain_readonly_text_with_bounded_samples(make_ui, mode):
    h = make_ui(mode)
    report = dict(policy='balanced', analyzed_spans=12, excluded_spans=2, excluded_windows=5,
                  flag_counts={'suspected_fragment': 2}, excluded_reasons={'suspected_fragment': 2},
                  samples=[dict(source='<script>alert(1)</script>.txt', start=10, end=40,
                                text='<b>fragment</b> ' + 'x' * 1000,
                                flags=['suspected_fragment'], reasons=['suspected_fragment'],
                                context_before='before ' + 'y' * 1000,
                                context_after='after ' + 'z' * 1000) for _ in range(8)])
    text = h.module._quality_report(report)
    assert isinstance(h.c['quality_report'], gr.Textbox)
    assert h.c['quality_report'].interactive is False
    assert h.c['quality_report'] not in h.controls
    assert h.c['quality_report'] in h.callbacks['build'].outputs
    assert 'Sentence spans analyzed: 12' in text and 'Sentence spans excluded: 2' in text
    assert 'Candidate windows excluded: 5' in text
    assert '<script>alert(1)</script>.txt [10:40]' in text
    assert 'Exclusion reasons: suspected_fragment' in text
    assert 'Before: before' in text and 'After: after' in text
    assert text.count('Sample ') == 3 and 'Additional samples omitted.' in text
    assert all(character * 241 not in text for character in 'xyz')
    assert 'original decoded characters' in text
    assert 'Quality exclusions are disabled' in h.module._quality_report(dict(report, policy='off'))


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_quality_report_prioritizes_later_exclusions_stably(make_ui, mode):
    h = make_ui(mode)
    retained = [dict(source=f'retained-{i}.txt', start=0, end=4, text='Yes.',
                     flags=['short_reference'], reasons=[]) for i in range(10)]
    excluded = [dict(source=f'excluded-{i}.txt', start=20, end=30, text='the house.',
                     flags=['suspicious_boundary'], reasons=['suspicious_boundary']) for i in range(2)]
    samples = retained + excluded
    report = dict(policy='balanced', analyzed_spans=12, excluded_spans=2, excluded_windows=2, samples=samples)
    text = h.module._quality_report(report)
    assert 'Sample 1: excluded-0.txt [20:30]' in text
    assert 'Sample 2: excluded-1.txt [20:30]' in text
    assert 'Sample 3: retained-0.txt [0:4]' in text
    assert 'retained-1.txt' not in text
    assert report['samples'] == retained + excluded
    assert 'Additional samples omitted.' in text


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_all_excluded_failure_shows_quality_report_and_retry_replaces_it(make_ui, monkeypatch, mode):
    import json

    h = make_ui(mode)
    previous = h.session.index
    attempts = []
    report = dict(policy='balanced', analyzed_spans=2, excluded_spans=2, excluded_windows=3,
                  flag_counts={'suspected_fragment': 2}, excluded_reasons={'suspected_fragment': 2},
                  samples=[dict(source='book.txt', start=5, end=15, text='fragment.',
                                flags=['suspected_fragment'], reasons=['suspected_fragment'])])

    class Encoder:
        def __init__(self, config):
            pass

        def load(self, progress):
            pass

        def release(self):
            pass

    class Index:
        def __init__(self, path):
            self.path = path

        def build(self, config, encoder, progress, force):
            attempts.append(config.quality_policy)
            if len(attempts) == 1:
                error = ValueError('No passages passed quality screening.')
                error.quality_report = report
                raise error
            return dict(files=1, candidates=1, occurrences=1,
                        quality=dict(report, analyzed_spans=3, excluded_spans=0, excluded_windows=0, samples=[]))

    monkeypatch.setattr(h.module, 'LateInteractionEncoder', Encoder)
    monkeypatch.setattr(h.module, 'CorpusIndex', Index)
    failed = list(h.callbacks['build'].fn(h.session, *h.config, True))
    assert_locked(h, failed[0], True)
    assert_locked(h, failed[-1], False)
    assert h.session.index is previous
    assert 'INDEX BUILD FAILED' in failed[-1][h.c['status']]
    assert 'excluded 2 sentence spans and 3 candidate windows' in failed[-1][h.c['status']]
    assert 'retry' in failed[-1][h.c['status']]
    assert 'Failed build — no new corpus was published.' in failed[-1][h.c['quality_report']]
    assert 'book.txt [5:15]' in failed[-1][h.c['quality_report']]
    retried = list(h.callbacks['build'].fn(h.session, *h.config, True))
    assert_locked(h, retried[-1], False)
    assert 'Corpus ready' in retried[-1][h.c['status']]
    assert 'Failed build' not in retried[-1][h.c['quality_report']]
    assert 'Sentence spans excluded: 0' in retried[-1][h.c['quality_report']]
    assert json.loads(h.module._settings_path(mode).read_text())['quality_policy'] == 'balanced'


def test_quality_policy_change_requires_rebuild(make_ui):
    h = make_ui()
    args = run_args(h)
    args[-1] = 'off'
    events = list(h.callbacks['rewrite'].fn(*args))
    assert 'Corpus settings changed' in events[-1][h.c['status']]
    assert not h.calls and h.session.pending is None
    assert_locked(h, events[-1], False)


def test_cleanup_preference_restoration(make_ui):
    import json

    h = make_ui()
    path = h.module._settings_path('notebook')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cleanup='scanned_book', join_hyphenated_lines=True, quality_policy='off')))
    defaults = h.module._defaults('notebook')
    assert defaults['cleanup'] == 'scanned_book' and defaults['join_hyphenated_lines'] is True
    assert defaults['quality_policy'] == 'off'


@pytest.mark.parametrize('available,expected', [(False, 'cpu'), (True, 'cuda')])
def test_embedding_device_auto_detects_torch_capability(make_ui, monkeypatch, available, expected):
    h = make_ui()
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert h.module._default_embedding_device() == expected
    assert h.module._defaults('notebook')['device'] == expected


def test_embedding_device_missing_torch_falls_back_to_cpu(make_ui, monkeypatch):
    h = make_ui()
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert h.module._default_embedding_device() == 'cpu'


def test_embedding_device_broken_gpu_probe_falls_back_to_cpu(make_ui, monkeypatch):
    h = make_ui()
    torch = ModuleType('torch')
    def unavailable():
        raise RuntimeError('GPU runtime unavailable')
    torch.cuda = SimpleNamespace(is_available=unavailable)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert h.module._default_embedding_device() == 'cpu'


@pytest.mark.parametrize('available,saved', [(True, 'cpu'), (False, 'cuda:1')])
def test_saved_embedding_device_overrides_auto_detection(make_ui, monkeypatch, available, saved):
    import json

    h = make_ui()
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    path = h.module._settings_path('notebook')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'device': saved}))
    assert h.module._defaults('notebook')['device'] == saved


@pytest.mark.parametrize('cuda_available,mps_available,expected',
                         [(False, True, 'mps'), (False, False, 'cpu'), (True, True, 'cuda')])
def test_embedding_device_prefers_cuda_then_mps(make_ui, monkeypatch, cuda_available, mps_available, expected):
    h = make_ui()
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert h.module._default_embedding_device() == expected
    assert any(choice[1] == 'mps' for choice in h.c['device'].choices)


def test_embedding_device_broken_mps_probe_falls_back_to_cpu(make_ui, monkeypatch):
    h = make_ui()
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    def unavailable():
        raise RuntimeError('MPS runtime unavailable')
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=unavailable))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert h.module._default_embedding_device() == 'cpu'


@pytest.mark.parametrize('callback,review', [('automatic_commit', False), ('commit', True)])
def test_applied_status_retains_context_trim_notice(make_ui, monkeypatch, callback, review):
    h = make_ui()
    def engine(*args, **kwargs):
        yield SimpleNamespace(text='Better sentence.', status='Complete', done=True,
                              plan=SimpleNamespace(context_trimmed=True))
    monkeypatch.setattr(h.module, 'generate_rewrite', engine)
    args = run_args(h, review=review)
    list(h.callbacks['rewrite'].fn(*args))
    assert h.session.pending['context_trimmed'] is True
    result = h.callbacks[callback].fn(h.session, args[1], args[2], 'draft', {}, '')
    assert result[h.c['status']].startswith('Rewrite applied.')
    assert 'Older notebook context was trimmed' in result[h.c['status']]
    assert 'every reference was retained' in result[h.c['status']]

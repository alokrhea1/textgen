"""Exercise actual Notebook event graphs and the normal autosave integration."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import gradio as gr
import pytest

from test_rewrite_auto_ui import auto_args, progress, ready


@pytest.fixture
def routes(monkeypatch, tmp_path):
    import modules

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(modules, name.rsplit('.', 1)[1], module, raising=False)
        return module

    shared = stub('modules.shared', args=SimpleNamespace(multi_user=False, extensions=[]),
                  settings={'show_two_notebook_columns': False, 'prompt-notebook': 'draft'},
                  gradio={}, input_elements=['native-setting'], user_data_dir=tmp_path,
                  model_name='fake-model', tokenizer=SimpleNamespace(name_or_path='fake-tokenizer'))

    def gradio_components(*keys):
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        return [shared.gradio[key] for key in keys]

    stub('modules.utils', gradio=gradio_components, sanitize_filename=lambda name: Path(name).name,
         get_available_prompts=lambda: ['draft'], current_time=lambda: 'new-prompt')
    ui = stub('modules.ui', gather_interface_values=lambda value: {'native-setting': value},
              create_refresh_button=lambda *args, **kwargs: gr.Button('Refresh', interactive=kwargs.get('interactive', True)),
              audio_notification_js='')
    stub('modules.logging_colors', logger=SimpleNamespace(exception=lambda *args: None))
    stub('modules.html_generator', generate_basic_html=lambda text: '<p>' + text + '</p>')
    stub('modules.prompts', count_tokens=lambda text: str(len(text)), load_prompt=lambda name: 'Loaded prompt.')
    stub('modules.logits', get_next_logits=lambda *args: ('', ''))
    stub('modules.text_generation', generate_reply_wrapper=lambda *args: iter(()),
         get_token_ids=lambda text: '', get_encoded_length=len, stop_everything_event=lambda: None)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'modules' / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(modules, name.rsplit('.', 1)[1], module, raising=False)
        spec.loader.exec_module(module)
        return module

    rewrite = load('modules.ui_sentence_rewrite', 'ui_sentence_rewrite.py')
    notebook = load('modules.ui_notebook', 'ui_notebook.py')
    default = load('modules.ui_default', 'ui_default.py')
    with gr.Blocks() as blocks:
        shared.gradio['native-setting'] = gr.Textbox()
        shared.gradio['interface_state'] = gr.State({})
        notebook.create_ui()
        default.create_ui()
        notebook.create_event_handlers()
        default.create_event_handlers()
    functions = blocks.fns.values() if isinstance(blocks.fns, dict) else blocks.fns
    by_id = {function._id: function for function in functions}
    return SimpleNamespace(shared=shared, blocks=blocks, notebook=notebook,
                           default=default, module=rewrite, ui=ui, by_id=by_id,
                           dependencies=blocks.config['dependencies'], tmp_path=tmp_path)


def event_chain(h, target, action):
    first = [edge for edge in h.dependencies if (target._id, action) in edge['targets']]
    assert len(first) == 1
    chain = [first[0]]
    while True:
        following = [edge for edge in h.dependencies if edge.get('trigger_after') == chain[-1]['id']]
        if not following:
            break
        assert len(following) == 1
        chain.append(following[0])
    return [h.by_id[edge['id']] for edge in chain]


def selected(h, mode):
    c, controls = h.shared.gradio['rewrite-ui-' + mode]
    config = ['references.txt', True, h.module.DEFAULT_MODEL, '', 'cpu', 256, 16,
              True, 3, 'conservative', False, 'balanced']
    session = h.module.RewriteSession()
    session.config, session.embedding_config = h.module._configs(config)
    session.encoder = SimpleNamespace(release=lambda: None)
    session.index = SimpleNamespace(path=h.tmp_path / 'index', search=lambda *args, **kwargs: [],
                                    assert_current=lambda *args, **kwargs: None)
    return SimpleNamespace(**vars(h), c=c, controls=controls, config=config,
                           session=session, mode=mode)


@pytest.mark.parametrize(('mode', 'target_name', 'action', 'source_name'), [
    ('notebook', 'Generate-notebook', 'click', 'textbox-notebook'),
    ('notebook', 'textbox-notebook', 'submit', 'textbox-notebook'),
    ('notebook', 'Regenerate-notebook', 'click', 'textbox-notebook'),
    ('default', 'Generate-default', 'click', 'textbox-default'),
    ('default', 'textbox-default', 'submit', 'textbox-default'),
    ('default', 'Continue-default', 'click', 'output_textbox'),
])
def test_six_native_routes_dispatch_then_commit_using_fresh_component_values(routes, mode, target_name, action, source_name):
    h = routes
    c, _ = h.shared.gradio['rewrite-ui-' + mode]
    chain = event_chain(h, h.shared.gradio[target_name], action)
    dispatches = [index for index, callback in enumerate(chain) if callback.fn is not None and callback.fn.__name__ == 'dispatch']
    assert len(dispatches) == 1
    index = dispatches[0]
    generation = chain[index]
    assert generation.inputs == [h.shared.gradio[source_name], *c['generation_inputs']]
    assert generation.outputs == c['generation_outputs']
    assert chain[index + 1].fn is c['generation_commit']
    expected_fresh = [c['session'], h.shared.gradio['textbox-notebook' if mode == 'notebook' else 'output_textbox'],
                      h.shared.gradio['textbox-notebook' if mode == 'notebook' else 'textbox-default'],
                      h.shared.gradio['prompt_menu-' + mode], h.shared.gradio['interface_state']]
    assert chain[index + 1].inputs == expected_fresh
    assert chain[index + 1].outputs == c['generation_commit_outputs']
    assert any(callback.fn is h.ui.gather_interface_values for callback in chain[:index])


def test_regenerate_restores_last_input_before_the_dispatch_snapshot(routes):
    h = routes
    chain = event_chain(h, h.shared.gradio['Regenerate-notebook'], 'click')
    assert chain[0].inputs == [h.shared.gradio['last_input-notebook']]
    assert chain[0].outputs == [h.shared.gradio['textbox-notebook']]
    assert chain[0].fn('Original prompt.') == 'Original prompt.'
    for target_name, action in [('Generate-notebook', 'click'), ('textbox-notebook', 'submit')]:
        start = event_chain(h, h.shared.gradio[target_name], action)[0]
        assert start.inputs == [h.shared.gradio['textbox-notebook']]
        assert start.outputs == [h.shared.gradio['last_input-notebook']]


@pytest.mark.parametrize('mode', ['notebook', 'default'])
def test_native_stop_registration_receives_current_rewrite_session(routes, mode):
    h = routes
    c, _ = h.shared.gradio['rewrite-ui-' + mode]
    callback = event_chain(h, h.shared.gradio['Stop-' + mode], 'click')[0]
    assert callback.inputs == [c['session']]
    assert callback.outputs == [c['status']]
    session = h.module.RewriteSession()
    session.busy, session.phase = True, 'automatic'
    update = callback.fn(session)
    assert isinstance(update, dict) and 'value' not in update
    assert session.cancel.is_set()
    # Processing a late native Stop response cannot replace a final stream
    # status such as "Stopped ... Applied 1 completed rewrite(s)".
    rendered = asyncio.run(h.blocks.postprocess_data(callback, update, None))
    assert len(rendered) == 1 and 'value' not in rendered[0]


def test_disabled_dispatch_retains_native_notebook_initial_periodic_and_final_autosaves(routes, monkeypatch):
    h = selected(routes, 'notebook')
    saves, native_calls = [], []

    def native(source, state):
        native_calls.append((source, state))
        yield source, '<p>' + source + '</p>'
        yield source + ' Draft output.', '<p>Draft output.</p>'

    monkeypatch.setattr(h.notebook, 'generate_reply_wrapper', native)
    monkeypatch.setattr(h.notebook, 'safe_autosave_prompt', lambda document, prompt: saves.append((document, prompt)))
    chain = event_chain(h, h.shared.gradio['Generate-notebook'], 'click')
    callback = next(callback for callback in chain if callback.fn is not None and callback.fn.__name__ == 'dispatch')
    state = {'temperature': .5}
    result = list(callback.fn(*auto_args(h, enabled=False, state=state)))
    output = h.shared.gradio['textbox-notebook']
    assert result[-1][output] == 'Existing sentence. Draft output.'
    assert native_calls == [('Existing sentence.', state)]
    assert native_calls[0][1] is state
    assert saves == [('Existing sentence.', 'draft'), ('Existing sentence.', 'draft'),
                     ('Existing sentence. Draft output.', 'draft')]


def test_automatic_staging_never_autosaves_provisional_text_final_commit_uses_existing_debounce(routes, monkeypatch):
    h = selected(routes, 'notebook')
    saves, timers = [], []

    def operation(source, state, retrieve, **kwargs):
        yield progress(source, status='Drafting', replacement='Unaccepted draft')
        yield progress(source + ' Accepted sentence.', completed=1, status='Accepted sentence 1')
        yield progress(source + ' Accepted sentence.', completed=1, status='Stopped', replacement='Incomplete next sentence')

    ready(h, monkeypatch, operation)
    monkeypatch.setattr(h.notebook, 'generate_reply_wrapper', lambda *args: (_ for _ in ()).throw(AssertionError('Native wrapper used')))
    monkeypatch.setattr(h.notebook, 'safe_autosave_prompt', lambda document, prompt: saves.append((document, prompt)))

    class Timer:
        def __init__(self, seconds, function):
            self.seconds, self.function, self.cancelled = seconds, function, False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(h.notebook.threading, 'Timer', Timer)
    chain = event_chain(h, h.shared.gradio['Generate-notebook'], 'click')
    callback = next(callback for callback in chain if callback.fn is not None and callback.fn.__name__ == 'dispatch')
    events = list(callback.fn(*auto_args(h)))
    assert saves == [] and timers == []
    assert any(event.get(h.c['auto_preview']) == 'Existing sentence. Accepted sentence.' for event in events)
    committed = h.c['generation_commit'](h.session, 'Existing sentence.', 'Existing sentence.', 'draft', {})
    document = committed[h.shared.gradio['textbox-notebook']]
    assert document == 'Existing sentence. Accepted sentence.'
    assert saves == [] and timers == []
    changed = [edge for edge in h.dependencies if (h.shared.gradio['textbox-notebook']._id, 'change') in edge['targets']]
    save_callback = next(h.by_id[edge['id']] for edge in changed
                         if h.by_id[edge['id']].fn is h.notebook.store_notebook_state_and_debounce)
    save_callback.fn(document, 'draft')
    assert len(timers) == 1 and timers[0].seconds == 1.0 and not timers[0].cancelled
    assert saves == []
    timers[0].function()
    assert saves == [('Existing sentence. Accepted sentence.', 'draft')]

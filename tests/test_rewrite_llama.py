"""Controlled llama.cpp request boundaries; no binary/model is needed."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def llama(monkeypatch):
    import modules

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if name.startswith('modules.') and name.count('.') == 1:
            monkeypatch.setattr(modules, name.split('.')[1], module, raising=False)
        return module

    shared = stub('modules.shared', stop_everything=False, is_multimodal=False,
                  args=SimpleNamespace(verbose=False))
    stub('modules.image_utils', convert_image_attachments_to_pil=None,
         convert_openai_messages_to_images=None, convert_pil_to_base64=None)
    stub('modules.logging_colors', logger=SimpleNamespace(error=lambda *a: None))
    stub('modules.utils', resolve_model_path=None)
    stub('modules.windows_subprocess', bind_to_parent_lifetime=None)
    spec = importlib.util.spec_from_file_location(
        '_rewrite_llama_under_test', Path(__file__).parents[1] / 'modules' / 'llama_cpp_server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = object.__new__(module.LlamaServer)
    model.process = None
    model.port, model.n_ctx = 8001, 20
    model.encode = lambda text, **kwargs: list(range(len(text) + int(kwargs['add_bos_token'])))
    model.prepare_payload = lambda state: {}
    posts = []

    class Response:
        status_code = 200
        status = 200
        closed = False
        def __init__(self):
            self.lines = iter(self.iter_lines())
        def raise_for_status(self):
            pass
        def iter_lines(self):
            for value in ['<think>', 'An internal sentence. ', '</think>', 'Replacement. ', 'Extra. ']:
                yield ('data: ' + json.dumps({'content': value})).encode()
        def close(self):
            self.closed = True
        def readline(self):
            return next(self.lines, b'')

    response = Response()
    model.session = SimpleNamespace(post=lambda *a, **kw: posts.append(kw['json']) or response)

    class Connection:
        def __init__(self, *args, **kwargs):
            self.sock = SimpleNamespace(shutdown=lambda how: None)
            self.closed = False
        def connect(self):
            pass
        def request(self, method, path, body, headers):
            posts.append(json.loads(body))
        def getresponse(self):
            return response
        def close(self):
            self.closed = True

    monkeypatch.setattr(module.http.client, 'HTTPConnection', Connection)
    return SimpleNamespace(model=model, shared=shared, posts=posts, response=response, module=module,
                           connection_class=Connection, monkeypatch=monkeypatch)


def state(**extra):
    result = dict(_rewrite_generation_guard=True, add_bos_token=True,
                  truncation_length=30, max_new_tokens=5, auto_max_new_tokens=False)
    result.update(extra)
    return result


def test_server_context_budget_uses_actual_bos_tokens(llama):
    with pytest.raises(ValueError, match='requires 16 tokens'):
        list(llama.model.generate_with_streaming('x' * 15, state()))
    assert not llama.posts


def test_auto_budget_respects_real_server_context(llama):
    list(llama.model.generate_with_streaming('x' * 10, state(auto_max_new_tokens=True)))
    assert llama.posts[0]['prompt'] == list(range(11))
    assert llama.posts[0]['n_predict'] == 9
    assert llama.response.closed


def test_reasoning_markers_survive_stream_and_close_early(llama):
    stream = llama.model.generate_with_streaming('prompt', state())
    assert next(stream) == '<think>'
    assert next(stream) == '<think>An internal sentence. '
    assert next(stream).endswith('</think>')
    stream.close()
    assert llama.response.closed


def test_cancelled_rewrite_does_not_open_request(llama):
    cancel = threading.Event()
    cancel.set()
    assert list(llama.model.generate_with_streaming('prompt', state(stop_event=cancel))) == []
    assert not llama.posts


def test_cancelled_stream_closes_response(llama):
    cancel = threading.Event()
    stream = llama.model.generate_with_streaming('prompt', state(stop_event=cancel))
    next(stream)
    cancel.set()
    assert list(stream) == []
    assert llama.response.closed


def test_ordinary_generation_retains_existing_context_behavior(llama):
    list(llama.model.generate_with_streaming('x' * 15, state(_rewrite_generation_guard=False)))
    assert llama.posts[0]['n_predict'] == 5


def test_rewrite_preserves_reasoning_tokens_only_when_opted_in(llama):
    # Supply all sampler keys, keeping this fixture independent of UI imports.
    source = Path(llama.module.__file__).read_text()
    import re
    keys = re.findall(r'state\[[\'\"]([^\'\"]+)[\'\"]\]', source.split('def prepare_payload', 1)[1].split('def _process_images', 1)[0])
    settings = dict.fromkeys(keys, 0)
    settings.update(dry_sequence_breakers='[]', sampler_priority=[], custom_token_bans='')
    ordinary = llama.module.LlamaServer.prepare_payload(llama.model, settings)
    assert 'preserved_tokens' not in ordinary
    settings['_rewrite_generation_guard'] = True
    payload = llama.module.LlamaServer.prepare_payload(llama.model, settings)
    assert {'<think>', '</think>', '<|channel>', '<channel|>', '<|channel|>', '<|message|>'} <= set(payload['preserved_tokens'])
    assert 'skip_special_tokens' not in payload


@pytest.mark.parametrize('phase', ['headers', 'tokens'])
def test_stop_interrupts_blocked_http_without_waiting_for_next_token(llama, phase):
    blocked = threading.Event()
    released = threading.Event()
    cleaned = threading.Event()
    cancel = threading.Event()

    class BlockingConnection(llama.connection_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sock = SimpleNamespace(shutdown=lambda how: released.set())
        def getresponse(self):
            if phase == 'headers':
                blocked.set()
                released.wait(2)
                raise OSError('connection shut down')
            def readline():
                blocked.set()
                released.wait(2)
                raise OSError('connection shut down')
            llama.response.readline = readline
            return llama.response
        def close(self):
            super().close()
            cleaned.set()

    llama.monkeypatch.setattr(llama.module.http.client, 'HTTPConnection', BlockingConnection)
    finished = threading.Event()
    failures = []
    def consume():
        try:
            assert list(llama.model.generate_with_streaming('prompt', state(stop_event=cancel))) == []
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert blocked.wait(1)
    cancel.set()
    assert finished.wait(1), 'Stop must release generation before another token arrives'
    assert released.is_set()
    assert cleaned.wait(1)
    assert not failures
    if phase == 'tokens':
        assert llama.response.closed


def test_generator_close_interrupts_socket_after_first_chunk(llama):
    blocked, released, cleaned = threading.Event(), threading.Event(), threading.Event()

    class Connection(llama.connection_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sock = SimpleNamespace(shutdown=lambda how: released.set())
        def getresponse(self):
            first = True
            def readline():
                nonlocal first
                if first:
                    first = False
                    return b'data: {"content": "Replacement. "}\n'
                blocked.set()
                released.wait(2)
                raise OSError('connection shut down')
            llama.response.readline = readline
            return llama.response
        def close(self):
            super().close()
            cleaned.set()

    llama.monkeypatch.setattr(llama.module.http.client, 'HTTPConnection', Connection)
    stream = llama.model.generate_with_streaming('prompt', state())
    assert next(stream) == 'Replacement. '
    assert blocked.wait(1)
    stream.close()
    assert released.is_set()
    assert cleaned.wait(1)
    assert llama.response.closed

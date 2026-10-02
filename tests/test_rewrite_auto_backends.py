"""Automatic Notebook backend guards exercise real threads, without models."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from test_rewrite_llama import llama


@pytest.fixture
def callbacks(monkeypatch):
    import modules

    shared = ModuleType('modules.shared')
    shared.stop_everything = False
    logging = ModuleType('modules.logging_colors')
    errors = []
    logging.logger = SimpleNamespace(exception=lambda message: errors.append(message))
    monkeypatch.setitem(sys.modules, 'modules.shared', shared)
    monkeypatch.setitem(sys.modules, 'modules.logging_colors', logging)
    monkeypatch.setattr(modules, 'shared', shared, raising=False)
    spec = importlib.util.spec_from_file_location(
        '_rewrite_auto_callbacks', Path(__file__).parents[1] / 'modules' / 'callbacks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, shared=shared, errors=errors)


@pytest.mark.parametrize('raise_exceptions', [False, True])
def test_callback_worker_error_follows_queued_partial_output(callbacks, raise_exceptions):
    failure = RuntimeError('generation failed after a complete-looking sentence')
    completions, cleaned = [], threading.Event()

    def worker(callback):
        try:
            callback('Partial sentence. ')
            raise failure
        finally:
            cleaned.set()

    with callbacks.module.Iteratorize(worker, callback=completions.append,
                                     raise_exceptions=raise_exceptions) as iterator:
        assert next(iterator) == 'Partial sentence. '
        if raise_exceptions:
            with pytest.raises(RuntimeError) as caught:
                next(iterator)
            assert caught.value is failure
        else:
            with pytest.raises(StopIteration):
                next(iterator)
    assert cleaned.is_set()
    assert not iterator.thread.is_alive()
    assert completions == [None], 'Worker errors must not leave its return value undefined'
    assert callbacks.errors == ['Failed in generation callback']


@pytest.mark.parametrize('raise_exceptions', [False, True])
def test_callback_cancellation_is_not_a_generation_error(callbacks, raise_exceptions):
    callbacks.shared.stop_everything = True
    completions = []

    def worker(callback):
        callback('Never accepted')
        raise AssertionError('Cancelled callback must stop the worker')

    with callbacks.module.Iteratorize(worker, callback=completions.append,
                                     raise_exceptions=raise_exceptions) as iterator:
        assert list(iterator) == []
    assert completions == [None]
    assert not iterator.thread.is_alive()
    assert callbacks.errors == []


def test_callback_early_consumer_close_waits_for_worker_cleanup(callbacks):
    released, cleaned = threading.Event(), threading.Event()

    def worker(callback):
        try:
            callback('A completed draft. ')
            assert released.wait(2)
            callback('Must be cancelled')
        finally:
            cleaned.set()

    with callbacks.module.Iteratorize(worker, raise_exceptions=True) as iterator:
        assert next(iterator) == 'A completed draft. '
        released.set()
    assert cleaned.is_set()
    assert not iterator.thread.is_alive()
    assert callbacks.errors == []


@pytest.mark.parametrize('exit_kind', ['context', 'generator'])
def test_callback_boundary_close_propagates_worker_failure(callbacks, exit_kind):
    failure = RuntimeError('Worker failed immediately after a sentence boundary')
    iterators = []

    def worker(callback):
        callback('A completed-looking sentence. ')
        raise failure

    def consume():
        with callbacks.module.Iteratorize(worker, raise_exceptions=True) as iterator:
            iterators.append(iterator)
            yield next(iterator)

    if exit_kind == 'context':
        with pytest.raises(RuntimeError) as caught:
            with callbacks.module.Iteratorize(worker, raise_exceptions=True) as iterator:
                iterators.append(iterator)
                assert next(iterator) == 'A completed-looking sentence. '
    else:
        stream = consume()
        assert next(stream) == 'A completed-looking sentence. '
        with pytest.raises(RuntimeError) as caught:
            stream.close()
    assert caught.value is failure
    assert iterators[0].worker_exception_raised
    assert not iterators[0].thread.is_alive()


def test_callback_worker_failure_does_not_mask_existing_consumer_error(callbacks):
    failure = RuntimeError('Worker failed')
    primary = ValueError('Consumer already failed')

    def worker(callback):
        callback('A completed-looking sentence. ')
        raise failure

    with pytest.raises(ValueError) as caught:
        with callbacks.module.Iteratorize(worker, raise_exceptions=True) as iterator:
            assert next(iterator) == 'A completed-looking sentence. '
            raise primary
    assert caught.value is primary
    assert not iterator.thread.is_alive()


def automatic_state(**extra):
    result = dict(_rewrite_generation_guard=False, _notebook_auto_generation=True,
                  add_bos_token=True, truncation_length=30, max_new_tokens=5,
                  auto_max_new_tokens=False)
    result.update(extra)
    return result


@pytest.mark.parametrize('cancel_kind', ['local', 'global'])
def test_cancelled_automatic_draft_does_not_tokenize_or_open_http(llama, cancel_kind):
    event = threading.Event()
    if cancel_kind == 'local':
        event.set()
    else:
        llama.shared.stop_everything = True

    def fail_encode(*args, **kwargs):
        raise AssertionError('Already cancelled operation must not tokenize')

    llama.model.encode = fail_encode
    assert list(llama.model.generate_with_streaming('prompt', automatic_state(stop_event=event))) == []
    assert not llama.posts


def test_cancel_during_automatic_tokenization_does_not_open_http(llama):
    event = threading.Event()

    def encode(*args, **kwargs):
        event.set()
        return [1, 2]

    llama.model.encode = encode
    assert list(llama.model.generate_with_streaming('prompt', automatic_state(stop_event=event))) == []
    assert not llama.posts


@pytest.mark.parametrize('phase', ['headers', 'tokens'])
def test_stop_interrupts_automatic_http_before_next_token(llama, phase):
    blocked, released, cleaned, finished = (threading.Event() for _ in range(4))
    event = threading.Event()
    failures = []

    class BlockingConnection(llama.connection_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sock = SimpleNamespace(shutdown=lambda how: released.set())

        def getresponse(self):
            if phase == 'headers':
                blocked.set()
                released.wait(2)
                raise OSError('socket shut down')

            def readline():
                blocked.set()
                released.wait(2)
                raise OSError('socket shut down')

            llama.response.readline = readline
            return llama.response

        def close(self):
            super().close()
            cleaned.set()

    llama.monkeypatch.setattr(llama.module.http.client, 'HTTPConnection', BlockingConnection)

    def consume():
        try:
            assert list(llama.model.generate_with_streaming(
                'prompt', automatic_state(stop_event=event))) == []
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert blocked.wait(1)
    event.set()
    assert finished.wait(1), 'Stop must finish without another generated token'
    consumer.join(timeout=1)
    assert released.is_set()
    assert cleaned.wait(1)
    assert not failures
    if phase == 'tokens':
        assert llama.response.closed


@pytest.mark.parametrize('failure_kind', ['transport', 'context', 'http'])
def test_automatic_http_failures_propagate_and_close_transport(llama, failure_kind):
    closed = threading.Event()
    original = OSError('connection failed before first token')
    if failure_kind == 'context':
        llama.response.status = 400
        llama.response.read = lambda: json.dumps({'error': {
            'type': 'exceed_context_size_error'}}).encode()
    elif failure_kind == 'http':
        llama.response.status = 503

    class FailingConnection(llama.connection_class):
        def getresponse(self):
            if failure_kind == 'transport':
                raise original
            return llama.response

        def close(self):
            super().close()
            closed.set()

    llama.monkeypatch.setattr(llama.module.http.client, 'HTTPConnection', FailingConnection)
    expected = {'transport': OSError, 'context': ValueError,
                'http': llama.module.requests.HTTPError}[failure_kind]
    with pytest.raises(expected) as caught:
        list(llama.model.generate_with_streaming('prompt', automatic_state()))
    assert closed.wait(1)
    if failure_kind == 'transport':
        assert caught.value is original
    else:
        assert llama.response.closed


def test_automatic_transport_error_after_partial_output_is_not_success(llama):
    failure = OSError('connection failed after provisional sentence')
    first = True

    def readline():
        nonlocal first
        if first:
            first = False
            return b'data: {"content": "A provisional sentence. "}\n'
        raise failure

    llama.response.readline = readline
    stream = llama.model.generate_with_streaming('prompt', automatic_state())
    assert next(stream) == 'A provisional sentence. '
    with pytest.raises(OSError) as caught:
        next(stream)
    assert caught.value is failure
    assert llama.response.closed


@pytest.mark.parametrize('automatic', [False, True])
def test_automatic_draft_preserves_native_prompt_budget_behavior(llama, automatic):
    # Drafting continues to use the ordinary generation context policy. The
    # stricter all-reference prompt budget applies only to an actual rewrite.
    list(llama.model.generate_with_streaming(
        'x' * 15, automatic_state(_notebook_auto_generation=automatic)))
    assert llama.posts[0]['prompt'] == list(range(16))
    assert llama.posts[0]['n_predict'] == 5
    assert llama.response.closed


def test_ordinary_context_http_error_retains_log_and_empty_output_behavior(llama):
    llama.response.status_code = 400
    llama.response.json = lambda: {'error': {'type': 'exceed_context_size_error'}}
    assert list(llama.model.generate_with_streaming(
        'prompt', automatic_state(_notebook_auto_generation=False))) == []
    assert llama.response.closed


@pytest.mark.parametrize('guard', [None, '_rewrite_generation_guard', '_notebook_auto_generation'])
def test_reasoning_marker_payload_is_scoped_to_notebook_operations(llama, guard):
    source = Path(llama.module.__file__).read_text()
    keys = re.findall(r'state\[[\'"]([^\'"]+)[\'"]\]',
                      source.split('def prepare_payload', 1)[1].split('def _process_images', 1)[0])
    settings = dict.fromkeys(keys, 0)
    settings.update(dry_sequence_breakers='[]', sampler_priority=[], custom_token_bans='')
    if guard:
        settings[guard] = True
    payload = llama.module.LlamaServer.prepare_payload(llama.model, settings)
    if guard:
        assert {'<think>', '</think>', '<|channel|>', '<|message|>', '<start_of_turn>'} <= set(
            payload['preserved_tokens'])
    else:
        assert 'preserved_tokens' not in payload
    assert 'skip_special_tokens' not in payload

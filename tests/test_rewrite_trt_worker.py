"""Socket lifecycle contracts using a fake native LLM; no GPU imports."""
import socket
import threading
from types import SimpleNamespace

import pytest

from modules.tensorrt_protocol import receive_message, send_message
from modules.tensorrt_worker import serve


@pytest.fixture
def worker():
    parent, child = socket.socketpair()
    submitted, aborted = threading.Event(), threading.Event()

    class Result:
        def __iter__(self):
            submitted.set()
            assert aborted.wait(timeout=2)
            return
            yield

        def abort(self):
            aborted.set()

    calls = []
    shutdown = threading.Event()
    llm = SimpleNamespace(generate_async=lambda prompt, **kwargs: calls.append((prompt, kwargs)) or Result(),
                          shutdown=shutdown.set)
    thread = threading.Thread(target=serve, args=(child, llm, lambda **kwargs: kwargs, 32), daemon=True)
    thread.start()
    assert receive_message(parent, 2)['type'] == 'ready'
    yield SimpleNamespace(socket=parent, calls=calls, submitted=submitted, aborted=aborted, shutdown=shutdown, thread=thread)
    try:
        if parent.fileno() >= 0:
            send_message(parent, dict(type='shutdown'))
    except (OSError, EOFError):
        pass
    parent.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert shutdown.is_set()


def test_worker_abort_interrupts_blocked_first_token(worker):
    send_message(worker.socket, dict(type='generate', request_id=1, prompt_ids=[1, 2], sampling_params=dict(max_tokens=4)))
    assert worker.submitted.wait(timeout=2)
    send_message(worker.socket, dict(type='abort', request_id=1))
    assert receive_message(worker.socket, 2) == dict(type='done', request_id=1, error=None)
    assert worker.calls[0][0] == [1, 2]
    assert worker.aborted.is_set()


def test_worker_rejects_budget_without_submission_and_can_recover(worker):
    send_message(worker.socket, dict(type='generate', request_id=1, prompt_ids=[1, 2], sampling_params=dict(max_tokens=32)))
    response = receive_message(worker.socket, 2)
    assert response['type'] == 'error' and 'will not be truncated' in response['error']
    assert worker.calls == []
    # An error is a single terminal frame, so the next request has no stale done.
    send_message(worker.socket, dict(type='generate', request_id=2, prompt_ids=[1, 2], sampling_params=dict(max_tokens=4)))
    assert worker.submitted.wait(timeout=2)
    send_message(worker.socket, dict(type='abort', request_id=2))
    assert receive_message(worker.socket, 2)['request_id'] == 2


def test_worker_eof_aborts_native_request(worker):
    send_message(worker.socket, dict(type='generate', request_id=1, prompt_ids=[1], sampling_params=dict(max_tokens=4)))
    assert worker.submitted.wait(timeout=2)
    worker.socket.close()
    assert worker.aborted.wait(timeout=2)
    worker.thread.join(timeout=2)
    assert worker.shutdown.is_set()


def test_transport_disconnect_and_timeout_are_explicit():
    parent, child = socket.socketpair()
    try:
        with pytest.raises(TimeoutError):
            receive_message(parent, .01)
        child.close()
        with pytest.raises(EOFError):
            receive_message(parent, 1)
    finally:
        parent.close()
        child.close()


def test_transport_closed_socket_reports_eof_for_send_and_receive():
    parent, child = socket.socketpair()
    parent.close()
    try:
        with pytest.raises(EOFError, match='socket is closed'):
            send_message(parent, dict(type='shutdown'))
        with pytest.raises(EOFError, match='socket is closed'):
            receive_message(parent, 1)
    finally:
        child.close()


def _proxy(worker):
    from modules.tensorrt_proxy import TensorRTWorker

    proxy = TensorRTWorker.__new__(TensorRTWorker)
    proxy.socket = worker.socket
    proxy.context_length = 32
    proxy.response_timeout = 2
    proxy._send_lock = threading.Lock()
    proxy._active = None
    proxy._next_id = 0
    proxy._closed = False
    proxy.process = None
    return proxy


def test_proxy_drains_cancelled_request_before_next_generation(worker):
    proxy = _proxy(worker)
    result = proxy.generate_async([1, 2], dict(max_tokens=4))
    assert worker.submitted.wait(timeout=2)
    result.abort()
    result.close()
    assert result.done and proxy._active is None
    second = proxy.generate_async([3], dict(max_tokens=4))
    second.abort()
    second.close()
    assert second.done and proxy._active is None
    assert len(worker.calls) == 2


def test_proxy_native_error_does_not_poison_next_request(worker):
    proxy = _proxy(worker)
    # Invalid prompt IDs are rejected by the worker rather than silently encoded.
    result = proxy.generate_async(['invalid'], dict(max_tokens=4))
    with pytest.raises(RuntimeError, match='integer token IDs'):
        next(result)
    assert result.done and proxy._active is None
    second = proxy.generate_async([3], dict(max_tokens=4))
    second.abort()
    second.close()
    assert second.done and len(worker.calls) == 1


def test_transport_send_has_deadline_when_worker_does_not_read():
    parent, child = socket.socketpair()
    parent.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    try:
        with pytest.raises(TimeoutError, match='sending'):
            send_message(parent, dict(payload='x' * 100000), timeout=.01)
    finally:
        parent.close()
        child.close()


def test_shutdown_signals_owned_group_after_direct_child_exits(monkeypatch):
    import signal
    import modules.tensorrt_proxy as module

    parent, child = socket.socketpair()
    proxy = module.TensorRTWorker.__new__(module.TensorRTWorker)
    proxy.socket = parent
    proxy._send_lock = threading.Lock()
    proxy._closed = False
    proxy.process = SimpleNamespace(pid=123456, wait=lambda **kwargs: 0, poll=lambda: 0)
    signals = []

    def killpg(pid, sig):
        assert pid == 123456
        signals.append(sig)
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(module.os, 'killpg', killpg)
    try:
        proxy.shutdown()
        assert signal.SIGTERM in signals
        assert proxy._closed
    finally:
        child.close()


def test_failure_cleanup_does_not_wait_for_send_lock(monkeypatch):
    import modules.tensorrt_proxy as module

    parent, child = socket.socketpair()
    proxy = module.TensorRTWorker.__new__(module.TensorRTWorker)
    proxy.socket = parent
    proxy._send_lock = threading.Lock()
    proxy._send_lock.acquire()
    proxy._closed = False
    proxy.process = None
    try:
        proxy.shutdown(graceful=False)
        assert proxy._closed and parent.fileno() == -1
    finally:
        proxy._send_lock.release()
        child.close()

"""Run TensorRT with its pinned dependencies in an isolated Python process."""
import os
from pathlib import Path
import socket
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

from modules.tensorrt_protocol import receive_message, send_message


class TensorRTWorker:
    def __init__(self, python, model, context_length, startup_timeout=1800, response_timeout=300, trust_remote_code=False):
        self.context_length = context_length
        self.response_timeout = response_timeout
        self._send_lock = threading.Lock()
        self._active = None
        self._next_id = 0
        self._closed = False
        self.process = None
        self.socket, child_socket = socket.socketpair()
        command = [str(python), '-u', str(Path(__file__).with_name('tensorrt_worker.py')),
                   '--fd', str(child_socket.fileno()), '--model', str(model),
                   '--context-length', str(context_length)]
        if trust_remote_code:
            command.append('--trust-remote-code')
        # A single visible device prevents any accidental multi-GPU engine build.
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        visible = env.get('CUDA_VISIBLE_DEVICES')
        if visible != '':
            env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[0] if visible else '0'
        try:
            self.process = subprocess.Popen(command, pass_fds=(child_socket.fileno(),), env=env, start_new_session=True)
            child_socket.close()
            ready = receive_message(self.socket, startup_timeout)
            if ready.get('type') != 'ready':
                raise RuntimeError(ready.get('error', 'TensorRT worker failed to initialize'))
        except BaseException:
            child_socket.close()
            self.shutdown()
            raise

    def _send(self, message, timeout=10):
        deadline = time.monotonic() + timeout
        if not self._send_lock.acquire(timeout=timeout):
            self.shutdown(graceful=False)
            raise TimeoutError('Timed out acquiring TensorRT worker send lock')
        try:
            if self._closed:
                raise RuntimeError('TensorRT worker is closed')
            send_message(self.socket, message, timeout=max(0, deadline - time.monotonic()))
        except (OSError, EOFError):
            self.shutdown(graceful=False)
            raise
        finally:
            self._send_lock.release()

    def generate_async(self, prompt, sampling_params, streaming=True):
        if self._active is not None:
            raise RuntimeError('TensorRT worker already has an active generation request')
        if sampling_params['max_tokens'] <= 0 or len(prompt) + sampling_params['max_tokens'] > self.context_length:
            raise ValueError(f'TensorRT request exceeds its {self.context_length}-token engine context; prompt will not be truncated')
        self._next_id += 1
        result = WorkerResult(self, self._next_id)
        self._send(dict(type='generate', request_id=result.request_id,
                        prompt_ids=prompt, sampling_params=sampling_params))
        self._active = result
        return result

    def shutdown(self, graceful=True):
        if self._closed:
            return
        self._closed = True
        if graceful and self._send_lock.acquire(timeout=0.2):
            try:
                send_message(self.socket, dict(type='shutdown'), timeout=0.2)
            except (OSError, EOFError):
                pass
            finally:
                self._send_lock.release()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        if self.process is not None:
            if graceful:
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            # MPI executors can outlive the direct Python child. Its private
            # process group belongs solely to this runtime, including startup
            # failures where the child exits before its descendants.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                self.process.poll()  # Reap the direct child before probing.
                try:
                    os.killpg(self.process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait(timeout=5)


class WorkerResult:
    def __init__(self, worker, request_id):
        self.worker, self.request_id = worker, request_id
        self.done = False
        self._aborted = False
        self._abort_deadline = None
        self._abort_lock = threading.Lock()

    def __iter__(self):
        return self

    def _receive(self, timeout):
        try:
            message = receive_message(self.worker.socket, timeout, interrupt_deadline=lambda: self._abort_deadline)
            if message.get('request_id') != self.request_id:
                raise RuntimeError('Unexpected TensorRT worker request ID')
            return message
        except (OSError, EOFError, TimeoutError, ValueError, RuntimeError):
            self.worker.shutdown(graceful=False)
            raise

    def __next__(self):
        if self.done:
            raise StopIteration
        message = self._receive(self.worker.response_timeout)
        kind = message.get('type')
        if kind in ('done', 'error'):
            self.done = True
            self.worker._active = None
            if kind == 'error' and not self._aborted:
                raise RuntimeError(message['error'])
            raise StopIteration
        if kind != 'output':
            self.worker.shutdown()
            raise RuntimeError('Invalid TensorRT worker response')
        return SimpleNamespace(outputs=[SimpleNamespace(token_ids=message['token_ids'], text_diff=message['text_diff'])])

    def abort(self):
        with self._abort_lock:
            if not self.done and not self._aborted and not self.worker._closed:
                self.worker._send(dict(type='abort', request_id=self.request_id))
                self._aborted = True
                self._abort_deadline = time.monotonic() + 10

    def close(self):
        self.abort()
        # Drain this request before allowing another; native abort must finish.
        while not self.done and not self.worker._closed:
            message = self._receive(10)
            if message.get('type') in ('done', 'error'):
                self.done = True
                self.worker._active = None

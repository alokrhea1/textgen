"""Length-framed JSON transport, separate from TensorRT's console output."""
import json
import select
import socket
import struct
import time

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _wait_socket(sock, timeout, writable=False):
    try:
        ready = select.select([] if writable else [sock], [sock] if writable else [], [], timeout)
    except ValueError as error:
        # Concurrent shutdown can close the descriptor between checking it
        # and entering select. Report the same lifecycle outcome as peer EOF.
        if sock.fileno() < 0:
            raise EOFError('TensorRT worker socket is closed') from error
        raise
    return bool(ready[1 if writable else 0])


def send_message(sock, message, timeout=10):
    data = json.dumps(message, ensure_ascii=False).encode('utf-8')
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError('TensorRT worker message is too large')
    frame = memoryview(struct.pack('!I', len(data)) + data)
    deadline = time.monotonic() + timeout
    while frame:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not _wait_socket(sock, remaining, writable=True):
            raise TimeoutError('Timed out sending to TensorRT worker')
        try:
            sent = sock.send(frame, socket.MSG_DONTWAIT)
        except BlockingIOError:
            continue
        if not sent:
            raise EOFError('TensorRT worker disconnected during send')
        frame = frame[sent:]


def receive_message(sock, timeout=None, interrupt_deadline=None):
    deadline = None if timeout is None else time.monotonic() + timeout

    def read(size):
        chunks = bytearray()
        while len(chunks) < size:
            effective_deadline = deadline
            if interrupt_deadline is not None:
                interrupted_at = interrupt_deadline()
                if interrupted_at is not None:
                    effective_deadline = interrupted_at if deadline is None else min(deadline, interrupted_at)
            remaining = None if effective_deadline is None else effective_deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError('Timed out waiting for TensorRT worker')
            poll = remaining
            if interrupt_deadline is not None:
                poll = min(0.1, remaining) if remaining is not None else 0.1
            if not _wait_socket(sock, poll):
                if interrupt_deadline is not None:
                    continue
                raise TimeoutError('Timed out waiting for TensorRT worker')
            data = sock.recv(size - len(chunks))
            if not data:
                raise EOFError('TensorRT worker disconnected')
            chunks.extend(data)
        return chunks

    size = struct.unpack('!I', read(4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError('TensorRT worker message is too large')
    message = json.loads(read(size))
    if not isinstance(message, dict):
        raise ValueError('Invalid TensorRT worker message')
    return message

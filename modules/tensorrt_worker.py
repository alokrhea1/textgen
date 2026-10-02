"""Isolated TensorRT-LLM 1.0 worker. Its socket carries JSON, never console logs."""
import argparse
import socket
import sys
import threading
import traceback
from pathlib import Path

# MPI executor processes re-import this main script with the parent's sys.path.
# Keep the repository root available, while excluding the directory containing
# the webui adapter named tensorrt_llm.py so it cannot shadow the real package.
_module_dir = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or '.').resolve() != _module_dir]
if str(_module_dir.parent) not in sys.path:
    sys.path.insert(0, str(_module_dir.parent))
from modules.tensorrt_protocol import receive_message, send_message


def serve(sock, llm, sampling_params_type, context_length):
    send_lock = threading.Lock()
    active_lock = threading.Lock()
    active = None
    closing = threading.Event()

    def send(message):
        with send_lock:
            send_message(sock, message)

    def abort(request):
        request['cancelled'].set()
        result = request.get('result')
        if result is not None:
            result.abort()

    def generate(request, command):
        nonlocal active
        result = None
        error = None
        try:
            prompt = command['prompt_ids']
            params = command['sampling_params']
            if not isinstance(prompt, list) or not all(isinstance(token, int) for token in prompt):
                raise ValueError('Prompt must contain integer token IDs')
            if params['max_tokens'] <= 0 or len(prompt) + params['max_tokens'] > context_length:
                raise ValueError('Request exceeds TensorRT engine context; prompt will not be truncated')
            result = llm.generate_async(prompt, sampling_params=sampling_params_type(**params), streaming=True)
            with active_lock:
                request['result'] = result
                cancelled = request['cancelled'].is_set()
            if cancelled:
                result.abort()
            else:
                for output in result:
                    if request['cancelled'].is_set() or closing.is_set():
                        break
                    completion = output.outputs[0]
                    send(dict(type='output', request_id=request['id'],
                              text_diff=completion.text_diff, token_ids=list(completion.token_ids)))
        except BaseException:
            if not request['cancelled'].is_set() and not closing.is_set():
                error = traceback.format_exc()
        finally:
            if result is not None:
                try:
                    result.abort()
                except Exception:
                    pass
            # Release active before sending done; the next request can arrive as
            # soon as the parent consumes that frame.
            with active_lock:
                if active is request:
                    active = None
            try:
                send(dict(type='error' if error else 'done', request_id=request['id'], error=error))
            except OSError:
                pass

    try:
        send(dict(type='ready'))
        while not closing.is_set():
            command = receive_message(sock)
            kind = command.get('type')
            if kind == 'shutdown':
                break
            with active_lock:
                if kind == 'abort':
                    if active is not None and command.get('request_id') == active['id']:
                        abort(active)
                elif kind == 'generate':
                    if active is not None:
                        send(dict(type='error', request_id=command['request_id'], error='Another generation is active'))
                        continue
                    active = dict(id=command['request_id'], cancelled=threading.Event(), result=None)
                    active['thread'] = threading.Thread(target=generate, args=(active, command), daemon=True)
                    active['thread'].start()
                else:
                    raise ValueError('Unknown TensorRT worker command')
    except (OSError, EOFError):
        pass
    finally:
        closing.set()
        with active_lock:
            request = active
            if request is not None:
                try:
                    abort(request)
                except Exception:
                    pass
        if request is not None:
            request['thread'].join(timeout=5)
        llm.shutdown()
        sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--context-length', type=int, default=8192)
    parser.add_argument('--trust-remote-code', action='store_true')
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    try:
        from tensorrt_llm._tensorrt_engine import LLM
        from tensorrt_llm.llmapi import SamplingParams

        llm = LLM(model=args.model, tensor_parallel_size=1, dtype='auto',
                  trust_remote_code=args.trust_remote_code,
                  skip_tokenizer_init=False, max_batch_size=1,
                  max_seq_len=args.context_length, max_input_len=args.context_length,
                  max_num_tokens=args.context_length)
    except BaseException:
        try:
            send_message(sock, dict(type='error', error=traceback.format_exc()))
        finally:
            sock.close()
        raise
    serve(sock, llm, SamplingParams, args.context_length)


if __name__ == '__main__':
    main()

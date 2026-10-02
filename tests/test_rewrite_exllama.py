"""Exercise native ExLlama boundaries without requiring CUDA extensions."""
import ast
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def backend():
    tree = ast.parse((Path(__file__).parents[1] / 'modules/exllamav3.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Exllamav3Model')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('generate_with_streaming', 'encode_rewrite_prompt')]
    namespace = dict(queue=queue, shared=SimpleNamespace(is_multimodal=False, stop_everything=False),
                     CustomSampler=lambda stages: stages, SS_Argmax=lambda: None,
                     get_max_prompt_length=lambda state: state['truncation_length'] - state['max_new_tokens'],
                     Job=lambda **kwargs: SimpleNamespace(**kwargs))

    def validate(prompt, state, input_ids=None):
        if input_ids.shape[-1] > namespace['get_max_prompt_length'](state):
            raise ValueError('References will not be silently truncated')

    namespace['validate_rewrite_prompt'] = validate
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<exllama methods>', 'exec'), namespace)
    calls, jobs, cancelled = [], [], []

    def encode(prompt, **kwargs):
        calls.append(kwargs)
        return np.arange(len(prompt) + int(kwargs['add_bos'])).reshape(1, -1)

    def submit(job):
        jobs.append(job)
        results = queue.Queue()
        for result in ({'text': '<think>reason</think>'}, {'text': 'Replacement.'}, {'eos': True}):
            results.put(result)
        return results

    model = SimpleNamespace(tokenizer=SimpleNamespace(encode=encode),
                            config=SimpleNamespace(eos_token_id_list=[2]),
                            parallel_generator=SimpleNamespace(submit=submit, cancel=cancelled.append))
    state = dict(temperature=0, add_bos_token=False, truncation_length=10, max_new_tokens=3,
                 auto_max_new_tokens=False, ban_eos_token=False, skip_special_tokens=False)
    return SimpleNamespace(namespace=namespace, model=model, state=state, calls=calls,
                           jobs=jobs, cancelled=cancelled)


def stream(backend, **state):
    return backend.namespace['generate_with_streaming'](
        backend.model, 'abcdefgh', dict(backend.state, **state))


def test_rewrite_rejects_actual_tokens_before_submission(backend):
    with pytest.raises(ValueError, match='silently truncated'):
        list(stream(backend, _rewrite_generation_guard=True))
    assert not backend.jobs
    assert backend.calls == [dict(add_bos=False, encode_special_tokens=True, embeddings=[])]


def test_ordinary_native_generation_still_truncates(backend):
    list(stream(backend))
    assert backend.jobs[0].input_ids.tolist() == [[1, 2, 3, 4, 5, 6, 7]]


def test_rewrite_retains_full_input_and_reasoning_markers(backend):
    result = list(stream(backend, _rewrite_generation_guard=True, truncation_length=12))
    assert backend.jobs[0].input_ids.tolist() == [list(range(8))]
    assert backend.jobs[0].decode_special_tokens is True
    assert result[-1] == '<think>reason</think>Replacement.'
    assert backend.cancelled == backend.jobs


def test_close_stream_cancels_native_job(backend):
    generated = stream(backend)
    next(generated)
    generated.close()
    assert backend.cancelled == backend.jobs


def test_pre_cancelled_request_does_not_submit(backend):
    event = threading.Event()
    event.set()
    assert list(stream(backend, stop_event=event)) == []
    assert not backend.jobs


def test_rewrite_preflight_uses_native_bos_and_special_tokens(backend):
    backend.namespace['encode_rewrite_prompt'](backend.model, 'prompt', backend.state)
    assert backend.calls == [dict(add_bos=False, encode_special_tokens=True)]

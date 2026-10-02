"""Lightweight TensorRT adapter contracts; no CUDA or TensorRT installation needed."""
import importlib.util
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def trt(monkeypatch):
    import sys

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    shared = stub('modules.shared', stop_everything=False)
    stub('modules.logging_colors', logger=SimpleNamespace(exception=lambda *args: None))
    stub('tensorrt_llm', __path__=[])
    stub('tensorrt_llm._tensorrt_engine', LLM=object)
    stub('tensorrt_llm.llmapi', SamplingParams=lambda **kwargs: kwargs)
    stub('modules.text_generation', get_max_prompt_length=lambda state: state['truncation_length'] - state['max_new_tokens'])
    # Import modules before attaching these stubs to avoid unrelated loaders.
    import modules
    monkeypatch.setattr(modules, 'shared', shared, raising=False)
    spec = importlib.util.spec_from_file_location('trt_contract', Path(__file__).parents[1] / 'modules/tensorrt_llm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    shared.tokenizer = SimpleNamespace(eos_token_id=2, encode=lambda prompt, add_special_tokens: ([1] if add_special_tokens else []) + [10, 11])
    model = module.TensorRTLLMModel()
    model.tokenizer = shared.tokenizer

    class Result:
        aborts = 0

        def __iter__(self):
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[12], text_diff='Sentence.')])

        def abort(self):
            self.aborts += 1

    result = Result()
    model.llm = SimpleNamespace(generate_async=lambda prompt, **kwargs: calls.append((prompt, kwargs)) or result)
    state = dict(add_bos_token=True, max_new_tokens=4, auto_max_new_tokens=False, truncation_length=8,
                 temperature=1, top_k=20, top_p=.9, min_p=0, repetition_penalty=1,
                 presence_penalty=0, frequency_penalty=0, no_repeat_ngram_size=0,
                 seed=0, ban_eos_token=False, skip_special_tokens=True, _rewrite_generation_guard=True)
    return SimpleNamespace(model=model, state=state, result=result, calls=calls, shared=shared)


@pytest.mark.parametrize('bos', [False, True])
def test_count_and_submit_exact_same_prompt_ids(trt, bos):
    trt.state['add_bos_token'] = bos
    assert list(trt.model.generate_with_streaming('template', trt.state)) == ['Sentence.']
    expected = ([1] if bos else []) + [10, 11]
    assert trt.calls[0][0] == expected
    assert trt.model.last_prompt_token_count == len(expected)
    assert trt.calls[0][1]['sampling_params']['add_special_tokens'] is bos
    assert trt.result.aborts == 1


def test_over_budget_never_submits_or_truncates(trt):
    trt.state['truncation_length'] = 6
    with pytest.raises(ValueError, match='will not be silently truncated'):
        list(trt.model.generate_with_streaming('template', trt.state))
    assert trt.calls == []


def test_already_cancelled_does_not_submit(trt):
    trt.state['stop_event'] = threading.Event()
    trt.state['stop_event'].set()
    assert list(trt.model.generate_with_streaming('template', trt.state)) == []
    assert trt.calls == []


def test_closing_stream_aborts_once(trt):
    stream = trt.model.generate_with_streaming('template', trt.state)
    assert next(stream) == 'Sentence.'
    stream.close()
    assert trt.result.aborts == 1


def test_local_stop_aborts_during_first_token_wait(trt):
    waiting, aborted = threading.Event(), threading.Event()
    event = trt.state['stop_event'] = threading.Event()

    class WaitingResult:
        aborts = 0

        def __iter__(self):
            waiting.set()
            assert aborted.wait(timeout=2), 'Native request was not aborted while waiting'
            return
            yield

        def abort(self):
            self.aborts += 1
            aborted.set()

    result = WaitingResult()
    trt.model.llm.generate_async = lambda *args, **kwargs: result
    replies, errors = [], []

    def request():
        try:
            replies.extend(trt.model.generate_with_streaming('template', trt.state))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    assert waiting.wait(timeout=2)
    event.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == [] and replies == []
    assert result.aborts == 1
    assert trt.shared.stop_everything is False

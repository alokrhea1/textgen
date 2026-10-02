import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from modules.sentence_rewrite.reranker import NuanceReranker, SEMANTIC_MODEL


@pytest.fixture
def classifier(monkeypatch):
    import torch

    class Tokenizer:
        model_max_length = 12

        def __init__(self):
            self.calls = []

        def __call__(self, first, second, **kwargs):
            self.calls.append(kwargs)
            assert kwargs['truncation'] is False

            def encode(a, b):
                lengths = [len(a.split()), len(b.split())]
                return lengths + [0] * (sum(lengths) + 1)

            if isinstance(first, str):
                return {'input_ids': encode(first, second)}
            encoded = [encode(a, b) for a, b in zip(first, second)]
            width = max(map(len, encoded))
            return {'input_ids': torch.tensor([row + [0] * (width - len(row)) for row in encoded])}

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(
                id2label={'0': 'contradiction', '1': 'neutral', '2': 'entailment'},
                label2id={'entailment': 2}, num_labels=3, max_position_embeddings=16,
            )
            self.batches = []
            self.evaluating = False

        def to(self, **kwargs):
            self.device_args = kwargs
            return self

        def eval(self):
            self.evaluating = True
            return self

        def __call__(self, input_ids):
            self.batches.append(len(input_ids))
            # Directional confidence differs, proving the reverse direction matters.
            logits = [[0.0, 0.0, float(row[0] - row[1])] for row in input_ids]
            return SimpleNamespace(logits=torch.tensor(logits))

    class SemanticModel(Model):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_labels=1, max_position_embeddings=16)

        def __call__(self, input_ids):
            self.batches.append(len(input_ids))
            return SimpleNamespace(logits=torch.tensor([[float(row[0] - row[1])] for row in input_ids]))

    tokenizer, model = Tokenizer(), Model()
    semantic_tokenizer, semantic_model = Tokenizer(), SemanticModel()
    tokenizer_loads, model_loads = [], []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            tokenizer_loads.append((name, kwargs))
            return semantic_tokenizer if name == SEMANTIC_MODEL else tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            model_loads.append((name, kwargs))
            return semantic_model if name == SEMANTIC_MODEL else model

    transformers = ModuleType('transformers')
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForSequenceClassification = AutoModel
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    return SimpleNamespace(tokenizer=tokenizer, model=model, model_loads=model_loads,
                           tokenizer_loads=tokenizer_loads, torch=torch,
                           semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer)


def test_lazy_safe_float32_load_and_release(classifier):
    reranker = NuanceReranker(device='cpu', local_files_only=True)
    assert not classifier.model_loads
    messages = []
    assert reranker.load(progress=messages.append) is reranker
    reranker.load()
    assert len(classifier.model_loads) == 2 and messages
    args = classifier.model_loads[0][1]
    assert args['trust_remote_code'] is False and args['use_safetensors'] is True
    assert args['local_files_only'] is True and args['torch_dtype'] == classifier.torch.float32
    assert classifier.model.evaluating and classifier.semantic_model.evaluating
    reranker.release()
    assert reranker._model is None and reranker._tokenizer is None
    assert reranker._semantic_model is None and reranker._semantic_tokenizer is None
    reranker.load()
    assert len(classifier.model_loads) == 4


def test_balanced_components_and_bounded_batches(classifier):
    reranker = NuanceReranker(device='cpu', batch_size=2)
    details = reranker.score_details('two words', ['one', 'three word candidate', 'another short example'])
    expected = classifier.torch.tensor([0.0, 0.0, -1.0]).softmax(0)[2].item()
    contradiction = classifier.torch.tensor([0.0, 0.0, -1.0]).softmax(0)[0].item()
    for item in details:
        assert item['semantic_score'] == pytest.approx(.5)
        assert item['entailment_score'] == pytest.approx(expected)
        assert item['contradiction_score'] == pytest.approx(contradiction)
        assert item['score'] == pytest.approx(.5 + .25 * expected - .25 * contradiction)
    assert reranker.score('two words', ['one']) == pytest.approx([details[0]['score']])
    assert classifier.model.batches[:3] == [2, 2, 2]
    assert classifier.semantic_model.batches[:3] == [2, 2, 2]
    assert all(call['truncation'] is False for call in classifier.tokenizer.calls)


def test_pair_limit_includes_special_tokens_and_rejects_without_inference(classifier):
    reranker = NuanceReranker(device='cpu')
    with pytest.raises(ValueError, match='Reduce the corpus window size or query length'):
        reranker.score('query words', ['word ' * 8])
    assert not classifier.model.batches
    assert not classifier.semantic_model.batches


def test_labels_are_discovered_instead_of_assumed(classifier):
    classifier.model.config.id2label = {0: 'ENTAILMENT', 1: 'neutral', 2: 'contradiction'}
    classifier.model.config.label2id = {'ENTAILMENT': 0}
    reranker = NuanceReranker(device='cpu')
    reranker.load()
    assert reranker._entailment_id == 0


@pytest.mark.parametrize('labels', [{0: 'LABEL_0', 1: 'LABEL_1'}, {0: 'entailment', 1: 'entails'}])
def test_unknown_or_ambiguous_entailment_label_fails(classifier, labels):
    classifier.model.config.id2label = labels
    classifier.model.config.label2id = {}
    with pytest.raises(ValueError, match='exactly one entailment label'):
        NuanceReranker(device='cpu').load()


def test_cancellation_before_load_and_empty_candidates(classifier):
    reranker = NuanceReranker(device='cpu')
    event = threading.Event()
    event.set()
    with pytest.raises(InterruptedError):
        reranker.score('query', ['candidate'], cancel_event=event)
    assert not classifier.model_loads
    assert reranker.score('query', []) == []
    assert not classifier.model_loads


@pytest.mark.parametrize(('query', 'texts'), [('', ['text']), ('query', ['']), ('query', [None])])
def test_empty_inputs_fail(classifier, query, texts):
    with pytest.raises(ValueError, match='nonempty'):
        NuanceReranker(device='cpu').score(query, texts)


def test_nonfinite_scores_fail(classifier):
    class InvalidModel:
        config = classifier.model.config

        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=classifier.torch.full((len(kwargs['input_ids']), 3), float('nan')))

    reranker = NuanceReranker(device='cpu')
    reranker.load()
    reranker._model = InvalidModel()
    with pytest.raises(ValueError, match='nonfinite'):
        reranker.score('query', ['candidate'])


def test_semantic_limit_checked_before_either_model_inference(classifier):
    classifier.semantic_tokenizer.model_max_length = 4
    with pytest.raises(ValueError, match='Semantic reranking'):
        NuanceReranker(device='cpu').score('two words', ['candidate'])
    assert not classifier.model.batches and not classifier.semantic_model.batches


def test_cancellation_between_batches_never_returns_partial_ranking(classifier):
    event = threading.Event()
    original = classifier.model.__class__.__call__
    model_class = classifier.model.__class__
    def interrupted(model, **kwargs):
        result = original(model, **kwargs)
        event.set()
        return result
    # An instance __call__ attribute is not consulted by Python's call operator.
    model_class.__call__ = interrupted
    try:
        with pytest.raises(InterruptedError):
            NuanceReranker(device='cpu', batch_size=1).score('query', ['candidate'], event)
        assert classifier.model.batches == [1]
        assert not classifier.semantic_model.batches
    finally:
        model_class.__call__ = original


def test_partial_load_failure_releases_both_models(classifier):
    classifier.semantic_model.config.num_labels = 3
    reranker = NuanceReranker(device='cpu')
    with pytest.raises(ValueError, match='one regression logit'):
        reranker.load()
    assert reranker._model is None and reranker._semantic_model is None

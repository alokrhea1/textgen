"""Balanced semantic similarity with bidirectional NLI nuance evidence.

The composite score is a ranking heuristic, not an equivalence probability.
"""

import gc
import threading


DEFAULT_MODEL = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
SEMANTIC_MODEL = 'cross-encoder/stsb-roberta-large'


class NuanceReranker:
    def __init__(self, device='cuda', local_files_only=False, model=DEFAULT_MODEL, batch_size=8,
                 semantic_model=SEMANTIC_MODEL):
        if not isinstance(model, str) or not model.strip():
            raise ValueError('Set a nonempty nuance reranker model.')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('Nuance reranker batch_size must be a positive integer.')
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self.model_name = model
        self.batch_size = batch_size
        if not isinstance(semantic_model, str) or not semantic_model.strip():
            raise ValueError('Set a nonempty semantic similarity model.')
        self.semantic_model_name = semantic_model
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._entailment_id = None
        self._max_tokens = None
        self._contradiction_id = None
        self._semantic_model = self._semantic_tokenizer = None
        self._semantic_max_tokens = None

    @staticmethod
    def _label(config, kind):
        ids = set()
        aliases = {'entailment': {'entailment', 'entail', 'entails'},
                   'contradiction': {'contradiction', 'contradict', 'contradicts'}}[kind]
        for index, label in (getattr(config, 'id2label', None) or {}).items():
            if str(label).strip().casefold() in aliases:
                ids.add(int(index))
        for label, index in (getattr(config, 'label2id', None) or {}).items():
            if str(label).strip().casefold() in aliases:
                ids.add(int(index))
        if len(ids) != 1:
            raise ValueError(f'Nuance model configuration must identify exactly one {kind} label; generic LABEL_0 labels are unsupported.')
        index = ids.pop()
        labels = getattr(config, 'num_labels', None)
        if index < 0 or not isinstance(labels, int) or index >= labels:
            raise ValueError(f'Nuance model configuration has an invalid {kind} label index.')
        return index

    @staticmethod
    def _entailment_label(config):
        return NuanceReranker._label(config, 'entailment')

    @staticmethod
    def _limit(model, tokenizer):
        limits = [getattr(model.config, 'max_position_embeddings', None),
                  getattr(tokenizer, 'model_max_length', None)]
        limits = [value for value in limits if isinstance(value, int) and 0 < value < 1_000_000]
        if not limits:
            raise ValueError('Reranker model does not declare a usable maximum input length.')
        return min(limits)

    def load(self, progress=None):
        with self._lock:
            if self._model is not None and self._semantic_model is not None:
                return self
            if progress:
                progress(f'Loading semantic and nuance rerankers on {self.device}.')
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError('Nuance reranker dependencies are missing. Install requirements/rewrite.txt in the webui environment.') from exc
            device = torch.device(self.device)
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA is unavailable for the nuance reranker. Choose CPU or install CUDA-enabled PyTorch.')
                if device.index is not None and device.index >= torch.cuda.device_count():
                    raise RuntimeError('The configured nuance reranker CUDA device does not exist.')
            common = dict(local_files_only=self.local_files_only, trust_remote_code=False)
            model = semantic_model = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, **common, use_safetensors=True, torch_dtype=torch.float32,
                )
                self._model, self._tokenizer = model, tokenizer
                self._entailment_id = self._label(model.config, 'entailment')
                self._contradiction_id = self._label(model.config, 'contradiction')
                if self._entailment_id == self._contradiction_id:
                    raise ValueError('Entailment and contradiction labels must be distinct.')
                self._max_tokens = self._limit(model, tokenizer)
                model.to(device=device, dtype=torch.float32).eval()
                semantic_tokenizer = AutoTokenizer.from_pretrained(self.semantic_model_name, **common)
                semantic_model = AutoModelForSequenceClassification.from_pretrained(
                    self.semantic_model_name, **common, use_safetensors=True, torch_dtype=torch.float32,
                )
                self._semantic_model, self._semantic_tokenizer = semantic_model, semantic_tokenizer
                if getattr(semantic_model.config, 'num_labels', None) != 1:
                    raise ValueError('Semantic similarity model must return one regression logit.')
                self._semantic_max_tokens = self._limit(semantic_model, semantic_tokenizer)
                semantic_model.to(device=device, dtype=torch.float32).eval()
            except Exception:
                self.release()
                model = semantic_model = None
                raise
            return self

    @staticmethod
    def _check_cancel(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError('Nuance reranking was interrupted.')

    def score(self, query, texts, cancel_event=None):
        """Return balanced semantic/NLI ranking scores, not probabilities."""
        return [item['score'] for item in self.score_details(query, texts, cancel_event)]

    def score_details(self, query, texts, cancel_event=None):
        """Return score and semantic, entailment, contradiction components."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Nuance reranking requires a nonempty query.')
        texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Nuance reranking candidates must be nonempty text.')
        self._check_cancel(cancel_event)
        if not texts:
            return []
        with self._lock:
            self._check_cancel(cancel_event)
            self.load()
            import torch

            pairs = [(query, text) for text in texts] + [(text, query) for text in texts]
            # Check both directions with their actual special tokens before any
            # inference. No partial ranking is returned if a pair is too large.
            for tokenizer, limit, label in [(self._tokenizer, self._max_tokens, 'Nuance'),
                                             (self._semantic_tokenizer, self._semantic_max_tokens, 'Semantic')]:
                for index, (premise, hypothesis) in enumerate(pairs):
                    self._check_cancel(cancel_event)
                    tokens = tokenizer(premise, hypothesis, add_special_tokens=True,
                                       truncation=False, padding=False)['input_ids']
                    if len(tokens) > limit:
                        raise ValueError(
                            f'{label} reranking candidate {index % len(texts) + 1} needs '
                            f'{len(tokens)} paired tokens; model limit is {limit}. '
                            'Reduce the corpus window size or query length; inputs are never silently truncated.'
                        )
            entailments, contradictions, semantic_scores = [], [], []
            with torch.inference_mode():
                for start in range(0, len(pairs), self.batch_size):
                    self._check_cancel(cancel_event)
                    batch = pairs[start:start + self.batch_size]
                    inputs = self._tokenizer(
                        [pair[0] for pair in batch], [pair[1] for pair in batch],
                        add_special_tokens=True, truncation=False, padding=True,
                        return_tensors='pt',
                    )
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    logits = self._model(**inputs).logits.float()
                    if (logits.ndim != 2 or logits.shape[0] != len(batch)
                            or logits.shape[1] <= max(self._entailment_id, self._contradiction_id)):
                        raise ValueError('Nuance model returned an invalid classifier output shape.')
                    if not torch.isfinite(logits).all().item():
                        raise ValueError('Nuance model returned nonfinite scores.')
                    probabilities = logits.softmax(dim=-1)
                    entailments.extend(probabilities[:, self._entailment_id].cpu().tolist())
                    contradictions.extend(probabilities[:, self._contradiction_id].cpu().tolist())
                    self._check_cancel(cancel_event)
                for start in range(0, len(pairs), self.batch_size):
                    self._check_cancel(cancel_event)
                    batch = pairs[start:start + self.batch_size]
                    inputs = self._semantic_tokenizer(
                        [pair[0] for pair in batch], [pair[1] for pair in batch],
                        add_special_tokens=True, truncation=False, padding=True, return_tensors='pt',
                    )
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    logits = self._semantic_model(**inputs).logits.float()
                    if logits.ndim != 2 or logits.shape != (len(batch), 1):
                        raise ValueError('Semantic model returned an invalid classifier output shape.')
                    if not torch.isfinite(logits).all().item():
                        raise ValueError('Semantic model returned nonfinite scores.')
                    semantic_scores.extend(logits.sigmoid()[:, 0].cpu().tolist())
                    self._check_cancel(cancel_event)
            count = len(texts)
            details = []
            for index in range(count):
                semantic = (semantic_scores[index] + semantic_scores[index + count]) / 2
                entailment = min(entailments[index], entailments[index + count])
                contradiction = max(contradictions[index], contradictions[index + count])
                details.append(dict(score=float(semantic + .25 * entailment - .25 * contradiction),
                                    semantic_score=float(semantic), entailment_score=float(entailment),
                                    contradiction_score=float(contradiction)))
            return details

    def release(self):
        with self._lock:
            self._model = self._tokenizer = None
            self._entailment_id = self._max_tokens = None
            self._contradiction_id = None
            self._semantic_model = self._semantic_tokenizer = None
            self._semantic_max_tokens = None
            gc.collect()

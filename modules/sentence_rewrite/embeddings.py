"""Trained ColBERT token embeddings and exact late-interaction scoring.

Mean MaxSim is directional retrieval relevance. Symmetric Mean MaxSim is an
optional sentence-similarity heuristic, not a calibrated equivalence score.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
from typing import Callable

import numpy as np

DEFAULT_MODEL = "lightonai/LateOn"
BASELINE_MODEL = "lightonai/GTE-ModernColBERT-v1"
MLATE_MODEL = "lightonai/mLateOn"
FAST_MODEL = "answerdotai/answerai-colbert-small-v1"
_SCORE_BLOCK = 512
_SIMILARITY_ELEMENTS = 4 * 1024 * 1024
_PADDED_DOCUMENT_TOKENS = 16 * 1024


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_MODEL
    revision: str = ""
    device: str = "cpu"
    max_tokens: int = 256
    batch_size: int = 16
    local_files_only: bool = False

    def __post_init__(self):
        if not self.model.strip() or self.max_tokens < 8 or self.batch_size < 1:
            raise ValueError("Set a model, max_tokens >= 8, and batch_size >= 1.")


def _matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("Expected a nonempty token-by-dimension embedding matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding contains nonfinite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding contains zero-length token vectors.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """Mean of each query token's best cosine match in the document."""
    query, document = _matrix(query), _matrix(document)
    if query.shape[1] != document.shape[1]:
        raise ValueError("Embedding dimensions differ.")
    forward, _ = _token_maxima(query, document, reverse=False)
    return float(forward.mean())


def symmetric_maxsim(sentence: np.ndarray, document: np.ndarray) -> float:
    """Bidirectional mean MaxSim; both inputs must use document encoding."""
    sentence, document = _matrix(sentence), _matrix(document)
    if sentence.shape[1] != document.shape[1]:
        raise ValueError("Embedding dimensions differ.")
    forward, reverse = _token_maxima(sentence, document, reverse=True)
    return float((forward.mean() + reverse.mean()) / 2)


def _token_maxima(query, document, *, reverse):
    """Exact maxima with at most a 512-by-512 float32 similarity tile."""
    forward = np.full(len(query), -np.inf, dtype=np.float32)
    backward = np.full(len(document), -np.inf, dtype=np.float32) if reverse else None
    for qstart in range(0, len(query), _SCORE_BLOCK):
        qend = min(qstart + _SCORE_BLOCK, len(query))
        for dstart in range(0, len(document), _SCORE_BLOCK):
            dend = min(dstart + _SCORE_BLOCK, len(document))
            tile = query[qstart:qend] @ document[dstart:dend].T
            np.clip(tile, -1, 1, out=tile)
            np.maximum(forward[qstart:qend], tile.max(axis=1), out=forward[qstart:qend])
            if reverse:
                np.maximum(backward[dstart:dend], tile.max(axis=0), out=backward[dstart:dend])
    return forward, backward


def score_many(query: np.ndarray, documents: list[np.ndarray],
               scoring: str = "symmetric", device: str = "cpu") -> list[float]:
    """Exact token matching on CPU or bounded float32 CUDA batches.

    Symmetric inputs must both use document-role encoding. Padding never
    participates in maxima, including when every genuine match is negative.
    """
    if scoring not in {"symmetric", "directional"}:
        raise ValueError("scoring must be 'symmetric' or 'directional'.")
    normalized_query = _matrix(query)
    matrices = [_matrix(document) for document in documents]
    if any(matrix.shape[1] != normalized_query.shape[1] for matrix in matrices):
        raise ValueError("Embedding dimensions differ.")
    if not matrices:
        return []
    reverse = scoring == "symmetric"
    if device == "cpu":
        results = []
        for document in matrices:
            forward, backward = _token_maxima(normalized_query, document, reverse=reverse)
            results.append(float((forward.mean() + backward.mean()) / 2) if reverse else float(forward.mean()))
        return results
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install sentence rewrite dependencies with python -m pip install -r requirements/rewrite.txt.") from exc
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("Exact batched scoring supports 'cpu' or a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for retrieval scoring. Choose device='cpu'.")
    if target.index is not None and not 0 <= target.index < torch.cuda.device_count():
        raise RuntimeError("The configured CUDA scoring device does not exist. Choose device='cpu' or an available device.")
    results = []
    with torch.inference_mode():
        q = torch.as_tensor(normalized_query, device=target)
        position = 0
        while position < len(matrices):
            first = matrices[position]
            if len(q) * len(first) > _SIMILARITY_ELEMENTS:
                # Long pairs retain exact scoring with a bounded tile.
                d = torch.as_tensor(first, device=target)
                forward = torch.full((len(q),), -torch.inf, device=target)
                backward = torch.full((len(d),), -torch.inf, device=target) if reverse else None
                for qs in range(0, len(q), _SCORE_BLOCK):
                    qe = min(qs + _SCORE_BLOCK, len(q))
                    for ds in range(0, len(d), _SCORE_BLOCK):
                        de = min(ds + _SCORE_BLOCK, len(d))
                        tile = torch.mm(q[qs:qe], d[ds:de].T).clamp_(-1, 1)
                        forward[qs:qe] = torch.maximum(forward[qs:qe], tile.amax(dim=1))
                        if reverse:
                            backward[ds:de] = torch.maximum(backward[ds:de], tile.amax(dim=0))
                score = (forward.mean() + backward.mean()) / 2 if reverse else forward.mean()
                results.append(float(score.cpu()))
                position += 1
                continue
            count, width = 1, len(first)
            while position + count < len(matrices) and count < 128:
                candidate_width = max(width, len(matrices[position + count]))
                if ((count + 1) * candidate_width * len(q) > _SIMILARITY_ELEMENTS
                        or (count + 1) * candidate_width > _PADDED_DOCUMENT_TOKENS):
                    break
                count += 1
                width = candidate_width
            group = matrices[position:position + count]
            padded = torch.zeros((count, width, q.shape[1]), dtype=torch.float32, device=target)
            lengths = torch.tensor([len(document) for document in group], device=target)
            for index, document in enumerate(group):
                padded[index, :len(document)] = torch.as_tensor(document, device=target)
            valid = torch.arange(width, device=target)[None, :] < lengths[:, None]
            similarities = torch.matmul(q[None, :, :], padded.transpose(1, 2)).clamp_(-1, 1)
            similarities.masked_fill_(~valid[:, None, :], -torch.inf)
            scores = similarities.amax(dim=2).mean(dim=1)
            if reverse:
                backward = similarities.amax(dim=1).masked_fill_(~valid, 0)
                scores = (scores + backward.sum(dim=1) / lengths) / 2
            results.extend(scores.cpu().tolist())
            position += count
    return results


class LateInteractionEncoder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._lock = threading.RLock()
        self._identity = None
        self._local_stats = None

    def _checkpoint_stats(self):
        root = Path(self.config.model).expanduser()
        if not root.is_dir():
            return None
        files = {}
        for target in sorted(root.rglob("*")):
            if target.is_file() and target.suffix in {".json", ".safetensors", ".txt", ".model", ".vocab", ".tiktoken"}:
                stat = target.stat()
                files[str(target.relative_to(root))] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
        return files

    def _local_fingerprint(self, stats):
        if stats is None:
            return None
        root = Path(self.config.model).expanduser()
        hashes = {}
        for name in stats:
            digest = hashlib.sha256()
            with (root / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
        return {"sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                "files": hashes}

    def _check_local_unchanged(self):
        if self._local_stats is not None and self._checkpoint_stats() != self._local_stats:
            raise RuntimeError("Local embedding checkpoint changed after loading. Release/reload the encoder and rebuild the corpus index.")

    def _metadata(self, filename: str):
        root = Path(self.config.model).expanduser()
        if root.is_dir():
            target = root / filename
            return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
        try:
            target = hf_hub_download(self.config.model, filename,
                                     revision=self.config.revision or None,
                                     local_files_only=self.config.local_files_only)
        except (EntryNotFoundError, LocalEntryNotFoundError):
            return None
        return json.loads(Path(target).read_text(encoding="utf-8"))

    def _validate_checkpoint(self):
        config = self._metadata("config.json") or {}
        if "HF_ColBERT" in config.get("architectures", []):
            return
        modules = self._metadata("modules.json") or []
        for module in modules:
            name = str(module.get("type", ""))
            if not (name.startswith("sentence_transformers.") or name == "pylate.models.Dense.Dense"):
                raise ValueError(f"Unsupported custom checkpoint module {name!r}; use a native trained ColBERT model.")
        saved = self._metadata("config_sentence_transformers.json") or {}
        projection = any(str(module.get("type", "")).endswith("Dense") for module in modules)
        late_interaction = str(saved.get("similarity_fn_name", "")).lower() in {"maxsim", "meanmaxsim"}
        native = any("MultiVectorMask" in str(module.get("type", "")) for module in modules)
        if not projection or not (late_interaction or native):
            raise ValueError("Use a trained ColBERT checkpoint with a saved token projection; generic sentence encoders are unsupported.")

    def load(self, progress: Callable[[str], None] | None = None):
        with self._lock:
            if self._model is not None:
                self._check_local_unchanged()
                return self
            if progress:
                progress(f"Loading token embedding model {self.config.model} on {self.config.device}.")
            try:
                import torch
                from sentence_transformers import MultiVectorEncoder
                from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
                import huggingface_hub
            except ImportError as exc:
                raise ImportError("Sentence rewrite embedding dependencies are missing or incompatible. Run python -m pip install -r requirements/rewrite.txt in the webui environment.") from exc
            if self.config.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable for sentence embeddings. Choose device='cpu' or install a CUDA-enabled PyTorch build.")
                if ":" in self.config.device and int(self.config.device.split(":", 1)[1]) >= torch.cuda.device_count():
                    raise RuntimeError("The configured CUDA device does not exist. Choose an available CUDA device or device='cpu'.")
            self._validate_checkpoint()
            local_stats = self._checkpoint_stats()
            fingerprint = self._local_fingerprint(local_stats)
            model_path = str(Path(self.config.model).expanduser()) if local_stats is not None else self.config.model
            model = MultiVectorEncoder(model_path, device=self.config.device,
                                       revision=self.config.revision or None,
                                       local_files_only=self.config.local_files_only,
                                       trust_remote_code=False,
                                       model_kwargs={"use_safetensors": True},
                                       similarity_fn_name="meanmaxsim")
            backbone = model[0]
            positions = getattr(backbone.auto_model.config, "max_position_embeddings", None)
            if positions and self.config.max_tokens > positions:
                raise ValueError(f"max_tokens exceeds this model's {positions}-token architecture limit.")
            backbone.query_length = self.config.max_tokens
            backbone.document_length = self.config.max_tokens
            backbone.max_seq_length = self.config.max_tokens
            if backbone.query_expansion is not None:
                # Preserve the trained expansion floor for short inputs while
                # allowing longer sentences through without 32-token truncation.
                expansion = dict(backbone.query_expansion)
                expansion["strategy"] = "min"
                expansion["length"] = min(expansion["length"], self.config.max_tokens)
                backbone.query_expansion = expansion
            for module in model:
                if isinstance(module, MultiVectorMask):
                    module.skiplist_words = []
                    module.resolve_with_tokenizer(model.tokenizer)
            model.eval()
            if local_stats is not None and self._checkpoint_stats() != local_stats:
                raise RuntimeError("Local embedding checkpoint changed while loading. Retry after the checkpoint files stop changing.")
            self._model = model
            self._local_stats = local_stats
            self._identity = {
                "model": self.config.model, "revision": self.config.revision,
                "resolved_revision": getattr(backbone.auto_model.config, "_commit_hash", None),
                "device": self.config.device, "max_tokens": self.config.max_tokens,
                "prompts": dict(model.prompts), "query_expansion": backbone.query_expansion,
                "preserve_punctuation": True, "normalize_token_vectors": True,
                "library": "sentence-transformers", "library_version": importlib.metadata.version("sentence-transformers"),
                "transformers_version": importlib.metadata.version("transformers"),
                "torch_version": importlib.metadata.version("torch"),
                "backend": "torch", "trust_remote_code": False,
                "local_checkpoint": fingerprint,
            }
            return self

    @property
    def identity(self) -> dict:
        with self._lock:
            self.load()
            return json.loads(json.dumps(self._identity))

    def _count(self, text: str, role: str) -> int:
        prefix = self._model.prompts.get(role) or ""
        return len(self._model.tokenizer(prefix + text, add_special_tokens=True, truncation=False)["input_ids"])

    def token_count(self, text: str) -> int:
        """Count document-role tokens including its trained marker and specials."""
        with self._lock:
            self.load()
            return self._count(text, "document")

    def _encode(self, texts: list[str], role: str) -> list[np.ndarray]:
        with self._lock:
            self.load()
            if not texts:
                return []
            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Embedding inputs must be nonempty text.")
                count = self._count(text, role)
                if count > self.config.max_tokens:
                    raise ValueError(f"Input has {count} embedding tokens; limit is {self.config.max_tokens}. Split the text before embedding.")
            method = self._model.encode_query if role == "query" else self._model.encode_document
            embeddings = method(texts, batch_size=self.config.batch_size,
                                show_progress_bar=False, convert_to_numpy=True,
                                normalize_embeddings=True)
            if len(embeddings) != len(texts):
                raise ValueError("Embedding backend returned an unexpected number of inputs.")
            return [_matrix(embedding) for embedding in embeddings]

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self._encode(list(texts), "document")

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], "query")[0]

    def encode_sentence(self, text: str) -> np.ndarray:
        return self._encode([text], "document")[0]

    def release(self):
        with self._lock:
            self._model = None
            self._identity = None
            self._local_stats = None
            gc.collect()

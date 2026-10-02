"""Transactional local corpus storage and exact late-interaction retrieval."""
import hashlib
import heapq
import json
import os
import re
import sqlite3
import stat
import threading
import uuid
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path
from .cleanup import CLEANUP_VERSION, MODES, clean_text

import numpy as np

SCHEMA_VERSION = 1
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_CORPUS_BYTES = 512 * 1024 * 1024
MAX_WINDOWS = 200000
MAX_FILES = 10000
MAX_INDEX_BYTES = 8 * 1024 ** 3
_LOCK_REGISTRY = {}
_LOCK_REGISTRY_LOCK = threading.Lock()


@dataclass(frozen=True)
class CorpusConfig:
    paths: str
    recursive: bool = True
    max_sentences: int = 3
    cleanup: str = 'conservative'
    join_hyphenated_lines: bool = False


@dataclass(frozen=True)
class RetrievalHit:
    text: str
    source: str
    start: int
    end: int
    sentence_count: int
    token_count: int
    score: float
    provenance: tuple = ()
    late_score: float | None = None
    nuance_score: float | None = None
    semantic_score: float | None = None
    entailment_score: float | None = None
    contradiction_score: float | None = None


def _matrix(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not all(value.shape) or not np.isfinite(value).all():
        raise ValueError('Embedding must be a nonempty finite token matrix.')
    if np.any(np.linalg.norm(value, axis=1) == 0):
        raise ValueError('Embedding contains a zero token vector.')
    return np.ascontiguousarray(value)


def _identity(encoder):
    return json.loads(json.dumps(encoder.identity, sort_keys=True, default=str))


def _sources(config, progress, cancel_event=None):
    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError('Corpus retrieval cancelled.')
    if not 1 <= config.max_sentences <= 20:
        raise ValueError('Maximum sentence window must be between 1 and 20.')
    if config.cleanup not in MODES:
        raise ValueError('Invalid corpus cleanup mode.')
    paths = [Path(x.strip()).expanduser().absolute() for x in config.paths.splitlines() if x.strip()]
    if not paths:
        raise ValueError('Enter at least one local text file or directory.')
    files = set()
    def walk_error(error):
        raise ValueError(f'Cannot read corpus directory: {error.filename}: {error}') from error
    for root in paths:
        check_cancelled()
        if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
            raise ValueError(f'Symlink corpus paths are not supported: {root}')
        if not root.exists():
            raise ValueError(f'Corpus path does not exist: {root}')
        if root.is_dir():
            found = 0
            for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
                check_cancelled()
                dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink()) if config.recursive else []
                for name in sorted(names):
                    path = Path(directory) / name
                    if path.suffix.lower() != '.txt':
                        continue
                    if path.is_symlink():
                        raise ValueError(f'Symlink text files are not supported: {path}')
                    files.add(path.resolve())
                    if len(files) > MAX_FILES:
                        raise ValueError(f'Corpus exceeds {MAX_FILES} source files.')
                    found += 1
            if not found:
                raise ValueError(f'No .txt files found in directory: {root}')
        elif root.suffix.lower() == '.txt':
            files.add(root.resolve())
        else:
            raise ValueError(f'Corpus file must have a .txt extension: {root}')
        if len(files) > MAX_FILES:
            raise ValueError(f'Corpus exceeds {MAX_FILES} source files.')
    manifest = []
    total = 0
    for i, path in enumerate(sorted(files), 1):
        progress(f'Hashing file {i}/{len(files)}: {path}')
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f'Corpus source is not a regular file: {path}')
        if before.st_size > MAX_FILE_BYTES:
            raise ValueError(f'Corpus file exceeds {MAX_FILE_BYTES} bytes: {path}')
        total += before.st_size
        if total > MAX_CORPUS_BYTES:
            raise ValueError('Corpus exceeds the total size limit.')
        digest = hashlib.sha256()
        bytes_read = 0
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                check_cancelled()
                bytes_read += len(chunk)
                if bytes_read > MAX_FILE_BYTES:
                    raise ValueError(f'Corpus source grew beyond the file size limit: {path}')
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Corpus source changed while hashing: {path}; retry indexing.')
        manifest.append({'path': str(path), 'size': before.st_size, 'sha256': digest.hexdigest()})
    return manifest


class CorpusIndex:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / 'corpus.sqlite3'
        self._token_lengths = {}
        self._token_length_key = None
        with _LOCK_REGISTRY_LOCK:
            self._lock = _LOCK_REGISTRY.setdefault(str(self.path.resolve()), threading.RLock())

    @property
    def manifest(self):
        if not self.path.exists():
            return None
        try:
            with sqlite3.connect(self.path) as db:
                result = json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])
                signature = result['signature']
                if not isinstance(signature['files'], list) or not isinstance(signature['encoder'], dict):
                    raise ValueError('Invalid cache signature')
                CorpusConfig(**signature['config'])
                if signature['schema'] != SCHEMA_VERSION or signature['splitter'] != 'period-spans-v1':
                    raise ValueError('Unsupported cache schema')
                return result
        except (sqlite3.Error, ValueError, TypeError, KeyError) as exc:
            raise ValueError('Corpus cache is corrupt; rebuild the index.') from exc

    @property
    def ready(self):
        return self.manifest is not None

    def _signature(self, config, encoder, progress, cancel_event=None):
        return {'schema': SCHEMA_VERSION, 'splitter': 'period-spans-v1',
                'cleanup_version': CLEANUP_VERSION,
                'config': asdict(config), 'encoder': _identity(encoder), 'files': _sources(config, progress, cancel_event)}

    def assert_current(self, config, encoder, progress=None, cancel_event=None):
        with self._lock:
            current = self.manifest
            if current is None:
                raise ValueError('Build the corpus index before retrieving replacements.')
            if current['signature'] != self._signature(config, encoder, progress or (lambda _: None), cancel_event):
                raise ValueError('Corpus files, paths, settings, or embedding model changed; rebuild the index.')

    def clear(self):
        with self._lock:
            self.path.unlink(missing_ok=True)
            self._token_lengths.clear()
            self._token_length_key = None

    def build(self, config, encoder, progress=lambda _: None, force=False):
        from .sentences import iter_sentence_spans
        progress = progress or (lambda _: None)
        with self._lock:
            encoder.load(progress)
            signature = self._signature(config, encoder, progress)
            try:
                previous = self.manifest
            except ValueError:
                if not force:
                    raise
                previous = None
            if not force and previous and previous['signature'] == signature:
                progress('Reusing unchanged corpus index.')
                return previous
            self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            staging = self.cache_dir / f'.building-{uuid.uuid4().hex}.sqlite3'
            db = None
            try:
                db = sqlite3.connect(staging)
                os.chmod(staging, 0o600)
                db.executescript('CREATE TABLE metadata(value TEXT); CREATE TABLE candidates(id INTEGER PRIMARY KEY, text TEXT UNIQUE, sentences INTEGER, tokens INTEGER, rows INTEGER, dims INTEGER, matrix BLOB); CREATE TABLE occurrences(candidate INTEGER, source TEXT, start INTEGER, end INTEGER); CREATE INDEX occurrence_candidate ON occurrences(candidate);')
                occurrence_count = 0
                unique_count = 0
                max_tokens = encoder.config.max_tokens
                batch_size = encoder.config.batch_size
                pending = []
                embedding_dimension = None
                matrix_bytes = 0
                cleanup_reports = []

                def flush():
                    nonlocal unique_count, embedding_dimension, matrix_bytes
                    if not pending:
                        return
                    progress(f'Embedding candidates {unique_count + 1}–{unique_count + len(pending)}')
                    matrices = encoder.encode_documents([item[0] for item in pending])
                    if len(matrices) != len(pending):
                        raise ValueError('Embedding backend returned an unexpected candidate count.')
                    for item, matrix in zip(pending, matrices):
                        matrix = _matrix(matrix)
                        if embedding_dimension is None:
                            embedding_dimension = matrix.shape[1]
                        elif matrix.shape[1] != embedding_dimension:
                            raise ValueError('Embedding backend returned inconsistent matrix dimensions; index was not published.')
                        matrix_bytes += matrix.nbytes
                        if matrix_bytes > MAX_INDEX_BYTES:
                            raise ValueError(f'Corpus matrices exceed the {MAX_INDEX_BYTES} byte index budget; reduce corpus or sentence windows.')
                        text, count, tokens, occurrences = item
                        cursor = db.execute('INSERT INTO candidates(text,sentences,tokens,rows,dims,matrix) VALUES(?,?,?,?,?,?)', (text, count, tokens, *matrix.shape, matrix.tobytes()))
                        db.executemany('INSERT INTO occurrences VALUES(?,?,?,?)', [(cursor.lastrowid, *o) for o in occurrences])
                    unique_count += len(pending)
                    db.commit()
                    pending.clear()

                for number, source in enumerate(signature['files'], 1):
                    progress(f'Reading file {number}/{len(signature["files"])}: {source["path"]}')
                    with Path(source['path']).open('rb') as stream:
                        raw = stream.read(MAX_FILE_BYTES + 1)
                    if len(raw) > MAX_FILE_BYTES or hashlib.sha256(raw).hexdigest() != source['sha256']:
                        raise ValueError(f'Corpus source changed while indexing: {source["path"]}; retry.')
                    try:
                        text = raw.decode('utf-8')
                    except UnicodeDecodeError as exc:
                        raise ValueError(f'Corpus file is not valid UTF-8: {source["path"]}') from exc
                    cleaned = clean_text(text, config.cleanup, config.join_hyphenated_lines)
                    text = cleaned.text
                    cleanup_reports.append({'source': source['path'], **cleaned.report})
                    progress(f'Cleanup {source["path"]}: {cleaned.report["changes"]} changes ({config.cleanup}).' + (' ' + cleaned.report['warning'] if cleaned.report['warning'] else ''))
                    spans = iter(iter_sentence_spans(text))
                    lookahead = deque()
                    def fill():
                        while len(lookahead) < config.max_sentences:
                            next_span = next(spans, None)
                            if next_span is None:
                                break
                            lookahead.append(next_span)
                    fill()
                    while lookahead:
                        span = lookahead[0]
                        if encoder.token_count(span.text) > max_tokens:
                            raw_start, _ = cleaned.source_span(span.start, span.end)
                            raise ValueError(f'Sentence at offset {raw_start} in {source["path"]} exceeds embedding token limit {max_tokens}.')
                        for count, ending in enumerate(lookahead, 1):
                            end = ending.end
                            candidate = text[span.start:end]
                            tokens = encoder.token_count(candidate)
                            if tokens > max_tokens:
                                break
                            occurrence_count += 1
                            if occurrence_count > MAX_WINDOWS:
                                raise ValueError(f'Corpus exceeds {MAX_WINDOWS} candidate occurrences.')
                            raw_start, raw_end = cleaned.source_span(span.start, end)
                            occurrence = (source['path'], raw_start, raw_end)
                            existing = db.execute('SELECT id FROM candidates WHERE text=?', (candidate,)).fetchone()
                            if existing:
                                db.execute('INSERT INTO occurrences VALUES(?,?,?,?)', (existing[0], *occurrence))
                            else:
                                match = next((item for item in pending if item[0] == candidate), None)
                                if match:
                                    match[3].append(occurrence)
                                else:
                                    pending.append((candidate, count, tokens, [occurrence]))
                                if len(pending) >= batch_size:
                                    flush()
                        lookahead.popleft()
                        fill()
                    db.commit()
                flush()
                if not unique_count:
                    raise ValueError('Corpus contains no complete period-terminated sentences.')
                if self._signature(config, encoder, progress) != signature:
                    raise ValueError('Corpus files or embedding identity changed during indexing; retry.')
                result = {'signature': signature, 'files': len(signature['files']), 'candidates': unique_count, 'occurrences': occurrence_count, 'matrix_bytes': matrix_bytes, 'generation': uuid.uuid4().hex}
                cleanup_totals = {}
                cleanup_samples = {}
                for report in cleanup_reports:
                    for kind, count in report['counts'].items():
                        cleanup_totals[kind] = cleanup_totals.get(kind, 0) + count
                    for snippet in report['snippets']:
                        cleanup_samples.setdefault(snippet['kind'], {'source': report['source'], **snippet})
                sample_priority = ('furniture_lines', 'ocr_wraps', 'hyphenated_wraps', 'soft_hyphen_wraps',
                                   'soft_hyphens', 'line_wraps', 'horizontal_whitespace', 'paragraph_whitespace',
                                   'paragraph_separators', 'line_separators', 'line_endings', 'byte_order_mark')
                result['cleanup'] = {'version': CLEANUP_VERSION, 'mode': config.cleanup,
                                     'join_hyphenated_lines': config.join_hyphenated_lines,
                                     'changes': sum(cleanup_totals.values()), 'counts': cleanup_totals,
                                     'files': cleanup_reports,
                                     'files_changed': sum(bool(report['changes']) for report in cleanup_reports),
                                     'samples': [cleanup_samples[kind] for kind in sample_priority if kind in cleanup_samples][:10],
                                     'warning': cleanup_reports[0]['warning'] if cleanup_reports else ''}
                db.execute('INSERT INTO metadata VALUES(?)', (json.dumps(result, sort_keys=True),))
                db.commit()
                db.close()
                db = None
                os.replace(staging, self.path)
                self._token_lengths.clear()
                self._token_length_key = None
                progress(f'Index ready: {unique_count} unique candidates from {len(signature["files"])} files.')
                return result
            finally:
                if db is not None:
                    db.close()
                staging.unlink(missing_ok=True)

    def search(self, query, encoder, top_k=5, length_mode='sentences', sentence_count=1,
               token_tolerance=0.15, token_counter=None, scoring='symmetric', diversity=0.0,
               exclude_exact=True, progress=None, reranker=None, rerank_pool=50, cancel_event=None,
               token_counter_key=None):
        """Scan all eligible matrices; optionally rerank the exact top pool.

        Hit.score is raw relevance before diversity (nuance score when enabled), while
        late_score and nuance_score retain the unmodified backend scores.
        """
        from .embeddings import score_many
        progress = progress or (lambda _: None)
        if scoring == 'directional':
            scoring = 'query'
        if not 1 <= top_k <= 50 or length_mode not in ('sentences', 'tokens') or scoring not in ('symmetric', 'query'):
            raise ValueError('Invalid retrieval options.')
        if not isinstance(rerank_pool, int) or not 1 <= rerank_pool <= 500:
            raise ValueError('Nuance reranking pool must be between 1 and 500.')
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise ValueError('Corpus retrieval cancelled.')
        check_cancelled()
        if not 0 <= token_tolerance <= 1 or not 0 <= diversity <= 1:
            raise ValueError('Tolerance and diversity must be between zero and one.')
        with self._lock:
            manifest = self.manifest
            if manifest is None:
                raise ValueError('Build the corpus index before retrieving replacements.')
            encoder.load(progress)
            signature = manifest['signature']
            config = CorpusConfig(**signature['config'])
            self.assert_current(config, encoder, progress, cancel_event)
            check_cancelled()
            if length_mode == 'tokens' and token_counter is None:
                raise ValueError('Token-length retrieval requires the generator token counter.')
            target_tokens = token_counter(query) if length_mode == 'tokens' else None
            query_matrix = _matrix(encoder.encode_query(query) if scoring == 'query' else encoder.encode_sentence(query))
            key = (manifest.get('generation'), token_counter_key)
            if token_counter_key is None or key != self._token_length_key:
                self._token_lengths.clear()
                self._token_length_key = key
            pool_size = max(top_k, rerank_pool) if reranker is not None else (top_k * 8 if diversity else top_k)
            heap = []
            scoring_batch = []
            scoring_tokens = 0
            def flush_scores():
                nonlocal scoring_tokens
                if not scoring_batch:
                    return
                check_cancelled()
                scores = score_many(query_matrix, [item[1] for item in scoring_batch],
                                    scoring='directional' if scoring == 'query' else 'symmetric',
                                    device=encoder.config.device)
                if len(scores) != len(scoring_batch):
                    raise ValueError('Scoring backend returned an unexpected result count.')
                for (metadata, _), score in zip(scoring_batch, scores):
                    score = float(score)
                    if not np.isfinite(score):
                        raise ValueError('Retrieval score is not finite.')
                    identifier, text, count, tokens = metadata
                    entry = (score, -identifier, identifier, text, count, tokens)
                    if len(heap) < pool_size:
                        heapq.heappush(heap, entry)
                    elif entry[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, entry)
                scoring_batch.clear()
                scoring_tokens = 0
            with sqlite3.connect(self.path) as db:
                sql = 'SELECT id,text,sentences,tokens,rows,dims FROM candidates'
                parameters = ()
                if length_mode == 'sentences':
                    sql += ' WHERE sentences=?'
                    parameters = (sentence_count,)
                for index, row in enumerate(db.execute(sql, parameters), 1):
                    check_cancelled()
                    identifier, text, count, tokens, rows, dims = row
                    if exclude_exact and text.strip() == query.strip():
                        continue
                    if length_mode == 'sentences' and count != sentence_count:
                        continue
                    if length_mode == 'tokens':
                        if identifier not in self._token_lengths:
                            self._token_lengths[identifier] = token_counter(text)
                        tokens = self._token_lengths[identifier]
                        if abs(tokens - target_tokens) > target_tokens * token_tolerance:
                            continue
                    blob = db.execute('SELECT matrix FROM candidates WHERE id=?', (identifier,)).fetchone()[0]
                    if rows < 1 or dims < 1 or len(blob) != rows * dims * 4:
                        raise ValueError('Invalid cached embedding; rebuild the corpus index.')
                    matrix = _matrix(np.frombuffer(blob, dtype=np.float32).reshape(rows, dims))
                    if dims != query_matrix.shape[1]:
                        raise ValueError('Cached embedding dimensions changed; rebuild the index.')
                    if scoring_batch and (len(scoring_batch) >= 64 or scoring_tokens + rows > 16384):
                        flush_scores()
                    scoring_batch.append(((identifier, text, count, tokens), matrix))
                    scoring_tokens += rows
                    if index % 100 == 0:
                        progress(f'Scoring candidate {index}/{manifest["candidates"]}')
                flush_scores()
                if not heap:
                    raise ValueError('No corpus candidates match the requested length and exclusion rules; adjust length settings or corpus.')
                ranked = sorted(heap, reverse=True)
                late_scores = {item[2]: item[0] for item in ranked}
                nuance_scores = {}
                nuance_details = {}
                if reranker is not None:
                    check_cancelled()
                    progress(f'Reranking {len(ranked)} exact late-interaction candidates for nuance.')
                    texts = [item[3] for item in ranked]
                    details_method = getattr(reranker, 'score_details', None)
                    if callable(details_method):
                        details = details_method(query, texts, cancel_event=cancel_event)
                        fields = ('score', 'semantic_score', 'entailment_score', 'contradiction_score')
                        try:
                            if len(details) != len(ranked):
                                raise ValueError('Unexpected nuance detail count')
                            if not all(isinstance(detail, dict) and all(np.isfinite(float(detail[field])) for field in fields) for detail in details):
                                raise ValueError('Nonfinite nuance details')
                            scores = [float(detail['score']) for detail in details]
                            nuance_details = {item[2]: {field: float(detail[field]) for field in fields} for item, detail in zip(ranked, details)}
                        except (KeyError, TypeError, ValueError, OverflowError) as exc:
                            raise ValueError('Nuance reranker returned invalid score components.') from exc
                    else:
                        scores = reranker.score(query, texts, cancel_event=cancel_event)
                    check_cancelled()
                    if len(scores) != len(ranked) or not np.isfinite(np.asarray(scores, dtype=float)).all():
                        raise ValueError('Nuance reranker returned invalid candidate scores.')
                    nuance_scores = {item[2]: float(score) for item, score in zip(ranked, scores)}
                    # Python's stable sort preserves late-score ordering on ties.
                    ranked.sort(key=lambda item: nuance_scores[item[2]], reverse=True)
                selected = []
                while ranked and len(selected) < top_k:
                    if selected and diversity:
                        def adjusted(item):
                            words = set(re.findall(r'\w+', item[3].casefold()))
                            overlap = max(len(words & set(re.findall(r'\w+', other[3].casefold()))) / max(1, len(words | set(re.findall(r'\w+', other[3].casefold())))) for other in selected)
                            return nuance_scores.get(item[2], item[0]) - diversity * overlap
                        chosen = max(ranked, key=adjusted)
                        ranked.remove(chosen)
                    else:
                        chosen = ranked.pop(0)
                    selected.append(chosen)
                hits = []
                for score, _, identifier, text, count, tokens in selected:
                    origins = tuple(db.execute('SELECT source,start,end FROM occurrences WHERE candidate=? ORDER BY source,start', (identifier,)))
                    source, start, end = origins[0]
                    components = nuance_details.get(identifier, {})
                    hits.append(RetrievalHit(text, source, start, end, count, tokens,
                                             nuance_scores.get(identifier, score), origins,
                                             late_scores[identifier], nuance_scores.get(identifier),
                                             components.get('semantic_score'), components.get('entailment_score'),
                                             components.get('contradiction_score')))
                warning = f' Only {len(hits)} matching candidates available for requested {top_k}.' if len(hits) < top_k else ''
                progress('Retrieval complete.' + warning + (' Diversity uses textual word overlap.' if diversity else ''))
                return hits

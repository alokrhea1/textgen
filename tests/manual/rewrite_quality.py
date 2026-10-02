#!/usr/bin/env python3
"""Opt-in, real-model ingestion/retrieval diagnostics with inspectable evidence.

Example (run only by the lead agent, with private output outside the repository):
  python tests/manual/rewrite_quality.py --device cuda:1 --offline \
      --output-dir /workspace/rewrite-validation/quality-after
  python tests/manual/rewrite_quality.py --device cuda:1 --offline \
      --corpus /workspace/literary-corpus/nabokov.txt --cleanup scanned_book \
      --output-dir /workspace/rewrite-validation/nabokov-quality

Use the same fixture/settings before and after a change, then pass --compare to
report the factual differences. Fixtures are diagnostics, not calibrated quality
scores. Private-corpus results require human assessment; fixture labels do not
apply to an arbitrary book. This script never changes sources or comparison.html.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def options(values, parser, allowed, label):
    result = {}
    for value in values:
        key, separator, raw = value.partition('=')
        if not separator or key not in allowed:
            parser.error(f'Unsupported {label} option {key!r}; available: {", ".join(sorted(allowed))}')
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError as exc:
            parser.error(f'{label} option {key!r} must contain a JSON value: {exc}')
    return result


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def code_identity():
    names = ['modules/sentence_rewrite', 'tests/manual/rewrite_quality.py',
             'tests/fixtures/rewrite_quality_cases.json']
    files = []
    for name in names:
        path = ROOT / name
        files.extend(sorted(path.glob('*.py')) if path.is_dir() else [path])
    result = {'sha256': {str(path.relative_to(ROOT)): sha256(path) for path in files}}
    for key, command in [('commit', ['rev-parse', 'HEAD']),
                         ('working_tree_status', ['status', '--porcelain', '--', *names])]:
        completed = subprocess.run(['git', '-C', str(ROOT), *command], capture_output=True, text=True)
        result[key] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def reranker_identity(reranker):
    if reranker is None:
        return None
    result = {'class': type(reranker).__name__, 'device': reranker.device,
              'batch_size': reranker.batch_size, 'local_files_only': reranker.local_files_only}
    for name, attribute, model_name in [('nuance', '_model', 'model_name'),
                                         ('semantic', '_semantic_model', 'semantic_model_name')]:
        model = getattr(reranker, attribute, None)
        config = getattr(model, 'config', None)
        config_dict = config.to_dict() if config is not None else None
        result[name] = {
            'requested_model': getattr(reranker, model_name),
            'loaded': model is not None,
            'resolved_commit': getattr(config, '_commit_hash', None),
            'name_or_path': getattr(config, '_name_or_path', None),
            'config_sha256': hashlib.sha256(json.dumps(config_dict, sort_keys=True, default=str).encode()).hexdigest()
                if config_dict is not None else None,
        }
    return result


def indexed_fixture(index, source_ids, suite, config, requested_quality_policy='balanced'):
    retained = {identifier: [] for identifier in source_ids.values()}
    with sqlite3.connect(index.path) as database:
        for text, source, start, end in database.execute(
                'SELECT c.text,o.source,o.start,o.end FROM candidates c '
                'JOIN occurrences o ON c.id=o.candidate ORDER BY o.source,o.start,c.id'):
            if source in source_ids:
                retained[source_ids[source]].append({'text': text, 'start': start, 'end': end})
    # Old checkouts lack a quality_policy field. Evaluate them against the
    # explicitly requested policy, while exposing that the feature was absent.
    # This permits a baseline comparison without pretending the policy ran.
    policy = {'cleanup': config.get('cleanup'),
              'quality_policy': config.get('quality_policy', requested_quality_policy)}
    probes = []
    for probe in suite.get('retention_probes', []):
        actual = any(item['text'] == probe['text'] for item in retained.get(probe['document'], []))
        applicable = all(policy.get(key.removeprefix('required_')) == value
                         for key, value in probe.items() if key.startswith('required_'))
        probes.append({**probe, 'retained': actual, 'applicable': applicable,
                       'expectation_met': actual == probe['expected_retained'] if applicable else None})
    return {'documents': retained, 'retention_probes': probes,
            'probe_evaluation_policy': policy,
            'quality_policy_supported': 'quality_policy' in config,
            'policy_note': 'Use the actual manifest configuration; when a legacy checkout lacks quality_policy, '
                           'compare its output against the requested policy without claiming that it implemented it.'}


def diagnostic(case, references, fixture):
    if not fixture:
        return {'assessment': 'human_review_required',
                'note': 'Fixture relevance labels do not apply to this corpus override.'}
    relevant = set(case.get('relevant_documents', []))
    discouraged = set(case.get('discouraged_documents', []))
    good_ranks = [rank for rank, hit in enumerate(references, 1) if relevant.intersection(hit['fixture_documents'])]
    bad_ranks = [rank for rank, hit in enumerate(references, 1) if discouraged.intersection(hit['fixture_documents'])]
    no_good_match = bool(case.get('expected_no_good_match'))
    return {
        'assessment': 'human_review_required',
        'first_labeled_relevant_rank': min(good_ranks) if good_ranks else None,
        'labeled_relevant_ranks': good_ranks, 'labeled_discouraged_ranks': bad_ranks,
        'expected_no_good_match': no_good_match,
        'references_returned_without_labeled_good_match': bool(references) and no_good_match,
        'note': 'Labels are narrow diagnostic expectations; all unlabeled results need inspection. '
                'Raw similarity and reranking scores are not relevance probabilities.',
    }


def comparison(previous, current):
    earlier = {item['id']: item for item in previous.get('queries', [])}
    deltas = []
    for item in current['queries']:
        old = earlier.get(item['id'])
        if old is None:
            continue
        deltas.append({
            'id': item['id'], 'same_query': old.get('query') == item['query'],
            'before_status': old.get('status'), 'after_status': item['status'],
            'before_texts': [hit['text'] for hit in old.get('references', [])],
            'after_texts': [hit['text'] for hit in item.get('references', [])],
            'before_diagnostic': old.get('diagnostic'), 'after_diagnostic': item.get('diagnostic'),
        })
    def sources(payload):
        return sorted((source['sha256'], source['size'])
                      for source in payload.get('manifest', {}).get('signature', {}).get('files', []))
    return {'same_settings': previous.get('settings') == current['settings'],
            'same_source_content': sources(previous) == sources(current),
            'same_fixture': previous.get('fixture_sha256') == current['fixture_sha256'],
            'queries': deltas,
            'note': 'These are observed differences, not an automatic quality verdict. Inspect model identities and settings.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/rewrite_quality_cases.json')
    parser.add_argument('--corpus', type=Path, action='append', help='Local file/directory; repeat for multiple paths.')
    parser.add_argument('--case', action='append', help='Run only these fixture case IDs; repeat to select several.')
    parser.add_argument('--compare', type=Path, help='An earlier results.json; differences do not imply improvement.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--model', default='lightonai/LateOn')
    parser.add_argument('--revision', default='')
    parser.add_argument('--max-tokens', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--rerank-pool', type=int, default=200)
    parser.add_argument('--diversity', type=float, default=0.0)
    parser.add_argument('--scoring', choices=('symmetric', 'query'), default='symmetric')
    parser.add_argument('--cleanup', choices=('none', 'conservative', 'scanned_book'), default='conservative')
    parser.add_argument('--max-sentences', type=int, default=1)
    parser.add_argument('--quality-policy', choices=('balanced', 'off'), default='balanced')
    parser.add_argument('--min-length-ratio', type=float, default=0.5)
    parser.add_argument('--min-semantic-score', type=float, default=0.3)
    parser.add_argument('--max-contradiction-score', type=float, default=0.8,
                        help='Reranker contradiction ceiling; 1 disables this filter.')
    parser.add_argument('--no-reranker', '--no-nuance', action='store_true')
    parser.add_argument('--corpus-option', action='append', default=[], metavar='NAME=JSON',
                        help='Additional CorpusConfig field for comparing new ingestion settings.')
    parser.add_argument('--search-option', action='append', default=[], metavar='NAME=JSON',
                        help='Additional search option for comparing new quality controls.')
    args = parser.parse_args()
    suite = json.loads(args.fixture.read_text(encoding='utf-8'))
    if suite.get('schema_version') != 1:
        parser.error('Unsupported fixture schema.')
    cases = [case for case in suite['cases'] if not args.case or case['id'] in args.case]
    if not cases or set(args.case or []) - {case['id'] for case in cases}:
        parser.error('Select at least one existing case ID.')
    # Keep each run independent: do not silently reuse stale results or fixtures.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory for each run.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.output_dir.resolve()
    sys.path.insert(0, str(ROOT))
    from modules.sentence_rewrite.corpus import CorpusConfig, CorpusIndex
    from modules.sentence_rewrite.embeddings import EmbeddingConfig, LateInteractionEncoder
    from modules.sentence_rewrite.reranker import NuanceReranker
    import torch

    compatibility_notes = []
    corpus_fields = set(CorpusConfig.__dataclass_fields__)
    search_fields = set(inspect.signature(CorpusIndex.search).parameters)
    corpus_settings = dict(recursive=True, max_sentences=args.max_sentences, cleanup=args.cleanup)
    if 'quality_policy' in corpus_fields:
        corpus_settings['quality_policy'] = args.quality_policy
    else:
        compatibility_notes.append('This checkout has no ingestion quality_policy; its original policy applies.')
    corpus_settings.update(options(args.corpus_option, parser,
                                   set(CorpusConfig.__dataclass_fields__) - {'paths'}, 'corpus'))
    search_settings = dict(top_k=args.top_k, length_mode='sentences', sentence_count=1,
                           scoring=args.scoring, diversity=args.diversity, exclude_exact=True,
                           rerank_pool=args.rerank_pool)
    for name, value in [('min_length_ratio', args.min_length_ratio),
                        ('min_semantic_score', 0.0 if args.no_reranker else args.min_semantic_score),
                        ('max_contradiction_score', 1.0 if args.no_reranker else args.max_contradiction_score)]:
        if name in search_fields:
            search_settings[name] = value
        else:
            compatibility_notes.append(f'This checkout has no {name}; no such filter applies.')
    # Generator-native token matching belongs in a loaded-backend/browser check.
    reserved = {'self', 'query', 'encoder', 'token_counter', 'token_counter_key',
                'progress', 'reranker', 'cancel_event', 'length_mode'}
    search_settings.update(options(args.search_option, parser,
                                   set(inspect.signature(CorpusIndex.search).parameters) - reserved, 'search'))
    if args.no_reranker and search_settings.get('min_semantic_score', 0) != 0:
        parser.error('--no-nuance requires min_semantic_score=0; a semantic cutoff needs the reranker.')
    if args.no_reranker and search_settings.get('max_contradiction_score', 1) != 1:
        parser.error('--no-nuance requires max_contradiction_score=1; a contradiction cutoff needs the reranker.')
    source_ids = {}
    if args.corpus:
        paths = '\n'.join(str(path.expanduser().resolve()) for path in args.corpus)
    else:
        source_dir = args.output_dir / 'fixture-corpus'
        source_dir.mkdir()
        for document in suite['documents']:
            identifier = document['id']
            if not identifier or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for character in identifier):
                parser.error(f'Invalid fixture document ID: {identifier!r}')
            source = source_dir / (identifier + '.txt')
            if source.exists():
                parser.error(f'Duplicate fixture document ID: {identifier!r}')
            source.write_text(document['text'] + '\n', encoding='utf-8')
            source_ids[str(source)] = identifier
        paths = str(source_dir)
    config = CorpusConfig(paths=paths, **corpus_settings)
    encoder_config = EmbeddingConfig(model=args.model, revision=args.revision, device=args.device,
                                     max_tokens=args.max_tokens, batch_size=args.batch_size,
                                     local_files_only=args.offline)
    encoder = LateInteractionEncoder(encoder_config)
    reranker = None if args.no_reranker else NuanceReranker(device=args.device, local_files_only=args.offline)
    index = CorpusIndex(args.output_dir / 'index')
    versions = {}
    for package in ('torch', 'transformers', 'sentence-transformers', 'numpy', 'huggingface-hub'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'fixture_sha256': sha256(args.fixture), 'fixture_authorship': suite.get('authorship'),
        'fixture_labels_apply': not bool(args.corpus), 'limitations': suite.get('limitations', []),
        'compatibility_notes': compatibility_notes,
        'code': code_identity(), 'python': sys.version, 'versions': versions,
        'torch_cuda': torch.version.cuda,
        'settings': {'corpus': corpus_settings, 'encoder': asdict(encoder_config),
                     'search': search_settings, 'reranker_enabled': reranker is not None},
        'requested_evaluation_policy': {'quality_policy': args.quality_policy},
        'corpus_paths': paths.splitlines(),
        'queries': [], 'progress': [], 'status': 'running',
    }
    output = args.output_dir / 'results.json'

    def progress(message):
        result['progress'].append(str(message))
        print(message, flush=True)

    started = time.monotonic()
    try:
        for note in compatibility_notes:
            progress(note)
        result['manifest'] = index.build(config, encoder, progress=progress, force=True)
        result['encoder_identity'] = encoder.identity
        source_hashes = {item['path']: item['sha256'] for item in result['manifest']['signature']['files']}
        source_texts = {}
        if source_ids:
            result['ingestion'] = indexed_fixture(index, source_ids, suite,
                result['manifest']['signature']['config'], requested_quality_policy=args.quality_policy)
        for case in cases:
            progress('Query: ' + case['id'])
            query_started = time.monotonic()
            query_result = {'id': case['id'], 'query': case['query'], 'focus': case.get('focus'),
                            'references': [], 'status': 'running'}
            try:
                hits = index.search(case['query'], encoder, reranker=reranker,
                                    progress=progress, **search_settings)
                for rank, hit in enumerate(hits, 1):
                    reference = asdict(hit)
                    reference['rank'] = rank
                    reference['fixture_documents'] = sorted({source_ids[origin[0]]
                        for origin in hit.provenance if origin[0] in source_ids})
                    reference['raw_source_spans'] = []
                    for source, start, end in hit.provenance:
                        if source not in source_texts:
                            # Do not translate CRLF: offsets address decoded original bytes.
                            raw = Path(source).read_bytes()
                            if hashlib.sha256(raw).hexdigest() != source_hashes[source]:
                                raise RuntimeError(f'Source changed while saving evaluation evidence: {source}')
                            source_texts[source] = raw.decode('utf-8')
                        reference['raw_source_spans'].append({
                            'source': source, 'start': start, 'end': end,
                            'source_sha256': source_hashes[source],
                            'text': source_texts[source][start:end],
                        })
                    query_result['references'].append(reference)
                query_result['status'] = 'returned' if hits else 'no_references'
            except ValueError as exc:
                # Keep the exact reason; a refusal is not automatically a success.
                query_result.update(status='retrieval_rejected', error=str(exc), error_type=type(exc).__name__)
                if not (str(exc).startswith('No ') and ('candidate' in str(exc) or 'reference' in str(exc))):
                    result['queries'].append(query_result)
                    raise
            query_result['elapsed_seconds'] = time.monotonic() - query_started
            query_result['diagnostic'] = diagnostic(case, query_result['references'], not bool(args.corpus))
            result['queries'].append(query_result)
            result['reranker_identity'] = reranker_identity(reranker)
            write_json(output, result)
        result['status'] = 'complete'
        if args.compare:
            previous = json.loads(args.compare.read_text(encoding='utf-8'))
            result['comparison'] = comparison(previous, result)
    except Exception as exc:
        result.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        result['reranker_identity'] = reranker_identity(reranker)
        write_json(output, result)
        encoder.release()
        if reranker is not None:
            reranker.release()
    print(f'Evidence saved to {output}. Inspect the references; completion is not a quality verdict.', flush=True)


if __name__ == '__main__':
    main()

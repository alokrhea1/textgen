"""Session-scoped retrieval rewriting for both Notebook layouts."""
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import threading
import uuid
import weakref

import gradio as gr

from modules import shared, ui
from modules.logging_colors import logger
from modules.sentence_rewrite.corpus import CorpusConfig, CorpusIndex
from modules.sentence_rewrite.embeddings import DEFAULT_MODEL, FAST_MODEL, EmbeddingConfig, LateInteractionEncoder
from modules.sentence_rewrite.engine import generate_rewrite
from modules.sentence_rewrite.quality import QUALITY_REASONS
from modules.sentence_rewrite.reranker import NuanceReranker
from modules.sentence_rewrite.sentences import last_sentence, replace_sentence


_sessions = weakref.WeakSet()
_sessions_lock = threading.Lock()


class RewriteSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.index = self.encoder = self.config = self.embedding_config = None
        self.reranker = self.reranker_config = self.pending = None
        self.history = []
        self.busy = False
        self.phase = ''
        self.build_path = None
        self.closed = False
        with _sessions_lock:
            _sessions.add(self)

    def __deepcopy__(self, memo):
        # Gradio deep-copies the initial State for each browser session.
        return RewriteSession()

    def dispose(self):
        self.closed = True
        self.cancel.set()
        if self.lock.acquire(blocking=False):
            try:
                for resource in (self.encoder, self.reranker):
                    if resource is not None:
                        resource.release()
                self.encoder = self.reranker = self.index = self.pending = None
            finally:
                self.lock.release()


def _guard():
    if shared.args.multi_user:
        raise ValueError('Local server-file retrieval is disabled in multi-user mode.')


def stop_notebook_generation(session):
    """Cancel this session's whole automatic operation, including retrieval."""
    if session.busy and session.phase == 'automatic':
        session.cancel.set()
    # The stream owns status updates. An asynchronous Stop response can arrive
    # after its final status, so this callback must not overwrite that result.
    return gr.update()


def _settings_path(mode):
    return shared.user_data_dir / 'retrieval_indexes' / f'{mode}-settings.json'


def _default_embedding_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        mps = getattr(getattr(torch, 'backends', None), 'mps', None)
        return 'mps' if mps is not None and mps.is_available() else 'cpu'
    except Exception:
        # Missing or incompatible optional GPU libraries must not prevent UI creation.
        return 'cpu'


def _defaults(mode):
    values = dict(paths='', recursive=True, model=DEFAULT_MODEL, revision='', device=_default_embedding_device(),
                  max_tokens=256, batch_size=16, local_files_only=False, max_sentences=3, cleanup='conservative',
                  join_hyphenated_lines=False, quality_policy='balanced')
    if not shared.args.multi_user:
        try:
            saved = json.loads(_settings_path(mode).read_text(encoding='utf-8'))
            values.update({k: v for k, v in saved.items() if k in values})
        except (OSError, ValueError, TypeError):
            pass
    return values


def _configs(values):
    paths, recursive, model, revision, device, max_tokens, batch_size, offline, max_sentences, cleanup, join_hyphenated_lines, quality_policy = values
    return (CorpusConfig(paths=paths, recursive=recursive, max_sentences=int(max_sentences),
                         cleanup=cleanup, join_hyphenated_lines=bool(join_hyphenated_lines), quality_policy=quality_policy),
            EmbeddingConfig(model=model, revision=revision, device=device, max_tokens=int(max_tokens),
                            batch_size=int(batch_size), local_files_only=offline))


def _source(mode, notebook, input_text):
    return notebook if mode == 'notebook' or notebook else input_text


def _seeded(source, seed):
    span = last_sentence(source)
    if seed.strip():
        source = replace_sentence(source, span, seed)
        span = last_sentence(source)
    return source, span


def _evidence(hits, length_mode='sentences'):
    def score(hit):
        if getattr(hit, 'nuance_score', None) is not None:
            parts = [f'Nuance score {hit.nuance_score:.4f}', f'Late interaction {hit.late_score:.4f}']
            for field, label in [('semantic_score', 'Semantic similarity'),
                                 ('entailment_score', 'Entailment'), ('contradiction_score', 'Contradiction')]:
                value = getattr(hit, field, None)
                if value is not None:
                    parts.append(f'{label} {value:.4f}')
            return ' | '.join(parts)
        return f'Late interaction {hit.score:.4f}'

    def quality(hit):
        flags = getattr(hit, 'quality_flags', ()) or ()
        if not flags:
            return ''
        lines = []
        for flag in flags[:8]:
            description = str(QUALITY_REASONS.get(flag, str(flag).replace('_', ' ')))
            description = description[:240] + ('…' if len(description) > 240 else '')
            if flag == 'short_reference':
                label = 'Quality note'
            elif flag in QUALITY_REASONS:
                label = 'Quality warning (suspected damage; inspect source)'
            else:
                label = 'Quality flag'
            lines.append(f'{label}: {description}')
        if len(flags) > 8:
            lines.append('Additional quality flags omitted.')
        return '\n' + '\n'.join(lines)

    return '\n\n'.join(
        f'{i}. {score(hit)} | {hit.sentence_count} sentence(s) | '
        f'{hit.token_count} {"generation" if length_mode == "tokens" else "embedding"} tokens\n'
        f'{hit.source} [{hit.start}:{hit.end}]{quality(hit)}\n{hit.text}'
        for i, hit in enumerate(hits, 1)
    )


def _cleanup_report(report):
    if not report:
        return 'No cleanup report is available for this corpus.'
    lines = [f"Cleanup: {report.get('mode', 'unknown')}",
             f"Join hyphenated lines: {bool(report.get('join_hyphenated_lines', False))}",
             'Source locations refer to original decoded characters.']
    if report.get('warning'):
        lines.append(f"Warning: {report['warning']}")
    if 'changes' in report:
        lines.append(f"Total changes: {report['changes']}")
    if 'files_changed' in report:
        lines.append(f"Files changed: {report['files_changed']}")
    for name, count in sorted(report.get('counts', {}).items()):
        lines.append(f'{name.replace("_", " ")}: {count}')
    samples = report.get('samples', [])
    for i, sample in enumerate(samples[:3], 1):
        def bounded(value):
            value = str(value)
            return value[:240] + ('…' if len(value) > 240 else '')
        lines.extend([f"Sample {i}: {bounded(sample.get('source', ''))}",
                      f"Before: {bounded(sample.get('before', ''))}",
                      f"After: {bounded(sample.get('after', ''))}"])
    if len(samples) > 3:
        lines.append('Additional samples omitted.')
    return '\n'.join(lines)


def _quality_report(report):
    if not report:
        return 'No ingestion quality report is available for this corpus.'

    def bounded(value):
        value = str(value)
        return value[:240] + ('…' if len(value) > 240 else '')

    lines = [f"Quality policy: {report.get('policy', 'unknown')}",
             f"Sentence spans analyzed: {report.get('analyzed_spans', 0)}",
             f"Sentence spans excluded: {report.get('excluded_spans', 0)}",
             f"Candidate windows excluded: {report.get('excluded_windows', 0)}",
             'Flags identify possible problems; source text is preserved. Locations refer to original decoded characters.']
    if report.get('policy') == 'off':
        lines.append('Quality exclusions are disabled. Flagged passages may remain in the corpus.')
    for name, count in sorted(report.get('flag_counts', {}).items()):
        lines.append(f"Flag — {name.replace('_', ' ')}: {count}")
    for name, count in sorted(report.get('excluded_reasons', {}).items()):
        lines.append(f"Excluded — {name.replace('_', ' ')}: {count}")
    samples = sorted(report.get('samples', []), key=lambda sample: not bool(sample.get('reasons')))
    for i, sample in enumerate(samples[:3], 1):
        lines.extend([f"Sample {i}: {bounded(sample.get('source', ''))} [{sample.get('start', '?')}:{sample.get('end', '?')}]",
                      f"Flags: {bounded(', '.join(sample.get('flags', [])))}",
                      f"Exclusion reasons: {bounded(', '.join(sample.get('reasons', []))) or 'none'}",
                      f"Text: {bounded(sample.get('text', ''))}"])
        for field, label in [('context_before', 'Before'), ('context_after', 'After')]:
            if sample.get(field):
                lines.append(f"{label}: {bounded(sample[field])}")
    if len(samples) > 3:
        lines.append('Additional samples omitted.')
    return '\n'.join(lines)


def create_ui(mode='notebook'):
    """Called inside the existing Notebook's subtab group."""
    defaults = _defaults(mode)
    c = {}
    controls = []

    def control(name, component):
        c[name] = component
        controls.append(component)
        return component

    with gr.Tab('Rewrite'):
        gr.Markdown('Build a local text corpus, then rewrite the last complete sentence using matching examples. '
                    'Enable automatic rewriting to retrieve and rewrite each newly completed sentence before generation continues. '
                    'Editing text never starts generation.')
        if shared.args.multi_user:
            gr.Markdown('Local file retrieval is disabled in multi-user mode.')
        c['session'] = gr.State(RewriteSession(), delete_callback=lambda state: state.dispose())
        control('auto', gr.Checkbox(label='Automatically retrieve and rewrite generated sentences', value=False,
                                   info='Applies to Notebook Generate, Continue, and Regenerate. Requires a ready corpus and a loaded generation model. Drafting shares the current Max new tokens allowance across the run; each rewrite has its own allowance. This setting is session-only.'))
        control('auto_max_sentences', gr.Slider(1, 100, step=1, value=5,
                                               label='Maximum sentences per automatic generation',
                                               info='Each newly completed sentence is rewritten before the next sentence is generated. Stop keeps completed rewrites and discards unfinished drafts.'))
        relative_example = Path('user_data') / 'corpus.txt'
        absolute_example = Path.cwd() / relative_example
        control('paths', gr.Textbox(label='Local .txt files or directories — one path per line', lines=3,
                                   value=defaults['paths'],
                                   placeholder=f'{relative_example}\n{absolute_example}',
                                   info=f'Paths refer to files on this server. Relative paths start from {Path.cwd()}. '
                                        f'Relative example: {relative_example}. Absolute example: {absolute_example}. '
                                        'On Linux/macOS, start an absolute path with /.'))
        with gr.Row():
            control('recursive', gr.Checkbox(label='Include subdirectories', value=defaults['recursive']))
            control('force', gr.Checkbox(label='Force rebuild', value=False))
            control('build', gr.Button('Build corpus / retry', variant='primary'))
            control('clear', gr.Button('Clear this corpus'))
        with gr.Accordion('Corpus and embedding settings', open=False):
            control('model', gr.Dropdown(label='Trained multi-vector embedding model',
                                        choices=list(dict.fromkeys([DEFAULT_MODEL, FAST_MODEL, 'lightonai/LateOn',
                                                                   'lightonai/mLateOn', 'lightonai/GTE-ModernColBERT-v1'])),
                                        value=defaults['model'], allow_custom_value=True,
                                        info='A Hugging Face ColBERT ID or a local trained checkpoint.'))
            with gr.Row():
                control('device', gr.Dropdown(label='Embedding device', choices=['cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu'],
                                             value=defaults['device'], allow_custom_value=True))
                control('revision', gr.Textbox(label='Model revision (optional commit)', value=defaults['revision']))
                control('local_files_only', gr.Checkbox(label='Offline: cached/local models only', value=defaults['local_files_only']))
            with gr.Row():
                control('max_tokens', gr.Slider(32, 8192, step=16, value=defaults['max_tokens'], label='Embedding token limit'))
                control('batch_size', gr.Slider(1, 128, step=1, value=defaults['batch_size'], label='Embedding batch size'))
                control('max_sentences', gr.Slider(1, 8, step=1, value=defaults['max_sentences'], label='Index windows up to this many sentences'))
            control('cleanup', gr.Dropdown(label='Cleanup applied to every corpus file',
                                          choices=['none', 'conservative', 'scanned_book'], value=defaults['cleanup'],
                                          info='Conservative normalizes whitespace and soft hyphens. Scanned book also removes standalone number lines, which may remove intended numbers.'))
            control('join_hyphenated_lines', gr.Checkbox(label='Join words hyphenated across lines',
                                                       value=defaults['join_hyphenated_lines'],
                                                       info='Optional and off by default to protect intentional compound words. Source locations refer to original decoded characters.'))
            control('quality_policy', gr.Dropdown(label='Ingestion quality screening',
                                                 choices=['balanced', 'off'], value=defaults['quality_policy'],
                                                 info='Balanced excludes passages with suspected damage before embedding. Off keeps flagged passages. Reports retain original source locations; changing this setting requires rebuilding.'))
        with gr.Row():
            control('top_k', gr.Slider(1, 50, step=1, value=5, label='Top K references'))
            control('length_mode', gr.Radio(['sentences', 'tokens'], value='sentences', label='Match length by'))
            control('sentence_count', gr.Slider(1, 8, step=1, value=1, label='Sentences per reference'))
            control('token_tolerance', gr.Slider(0, 1, step=0.05, value=0.15,
                                                label='Generation-token length tolerance', info='0 = exactly the query token count.'))
        control('min_length_ratio', gr.Slider(0, 1, step=0.05, value=0.5,
                                            label='Minimum reference length relative to target',
                                            info='Sentence mode only: compare embedding content-token counts. 0.5 requires at least half the target length; 0 disables this filter. Token mode uses the generation-token tolerance.'))
        with gr.Row():
            control('scoring', gr.Radio(['symmetric', 'directional'], value='symmetric', label='Late-interaction score',
                                        info='Symmetric compares whole sentences; directional uses trained query/document roles.'))
            control('diversity', gr.Slider(0, 1, step=0.05, value=0.0, label='Reference diversity',
                                          info='0 = closest matches. Increase to reduce similar wording among references.'))
            control('exclude_exact', gr.Checkbox(label='Exclude exact copies of the query', value=True))
        with gr.Row():
            control('nuance', gr.Checkbox(label='Rerank for meaning and nuance', value=True,
                                         info='Combine pairwise semantic similarity (STS) with NLI entailment and contradiction checks. Useful for negation and who did what to whom.'))
            control('rerank_pool', gr.Slider(5, 500, step=5, value=200, label='Late-interaction candidates to rerank',
                                            info='A larger pool gives the nuance model more alternatives.'))
        control('min_semantic_score', gr.Slider(0, 1, step=0.05, value=0.3,
                                              label='Minimum semantic similarity',
                                              info='Requires meaning and nuance reranking. English STS scores are heuristic, not confidence percentages. Lower the minimum for indirect style matches or multilingual text; 0 disables this filter. May return fewer than K references or fail if none qualify.'))
        control('max_contradiction_score', gr.Slider(0, 1, step=0.05, value=0.8,
                                                   label='Maximum contradiction score',
                                                   info='Requires meaning and nuance reranking. English NLI scores are heuristic. Reject references that strongly contradict the target; increase for indirect style matches or multilingual text. 1 disables this filter.'))
        control('seed', gr.Textbox(label='Optional seed sentence', lines=2,
                                  info='Manual Rewrite only. Leave blank to use the notebook’s last period-completed sentence. A seed guides its replacement.'))
        control('guidance', gr.Textbox(label='Optional writing guidance', lines=2,
                                      placeholder='For example: vary the syntax and use concrete, vivid verbs.'))
        with gr.Row():
            control('template', gr.Checkbox(label='Use the selected instruction template', value=True))
            control('thinking', gr.Checkbox(label='Enable model thinking for this rewrite', value=False))
            control('review', gr.Checkbox(label='Review before applying', value=False,
                                         info='Manual Rewrite only. Automatic generation applies accepted sentences together after the run.'))
        with gr.Row():
            control('search', gr.Button('Preview references'))
            control('rewrite', gr.Button('Rewrite', variant='primary'))
            control('stop', gr.Button('Stop generation', interactive=False))
            control('apply', gr.Button('Apply rewrite', interactive=False))
            control('undo', gr.Button('Undo rewrite', interactive=False))
        c['status'] = gr.Textbox(label='Status', value='Choose local text files, then build the corpus.', interactive=False)
        c['cleanup_report'] = gr.Textbox(label='Cleanup report', interactive=False, lines=7)
        c['quality_report'] = gr.Textbox(label='Ingestion quality report', interactive=False, lines=10)
        c['target'] = gr.Textbox(label='Sentence used for retrieval', interactive=False, lines=2)
        c['result'] = gr.Textbox(label='Proposed replacement', interactive=False, lines=3)
        c['references'] = gr.Textbox(label='Retrieved references and sources', interactive=False, lines=10)
        c['auto_preview'] = gr.Textbox(label='Automatic generation preview', interactive=False, lines=10,
                                       info='Staged document containing completed rewrites. Notebook text updates once the run finishes or stops; unfinished drafts are discarded.')
    if shared.args.multi_user:
        for component in controls:
            component.interactive = False
    shared.gradio[f'rewrite-ui-{mode}'] = (c, controls)


def create_event_handlers(mode='notebook'):
    c, controls = shared.gradio[f'rewrite-ui-{mode}']
    session = c['session']
    config_names = ['paths', 'recursive', 'model', 'revision', 'device', 'max_tokens', 'batch_size', 'local_files_only', 'max_sentences', 'cleanup', 'join_hyphenated_lines', 'quality_policy']
    config_inputs = [c[x] for x in config_names]
    outputs = [session, c['status'], c['target'], c['result'], c['references'], c['cleanup_report'], c['quality_report'], *controls]
    notebook = shared.gradio['textbox-notebook' if mode == 'notebook' else 'output_textbox']
    input_text = shared.gradio['textbox-notebook' if mode == 'notebook' else 'textbox-default']
    prompt = shared.gradio[f'prompt_menu-{mode}']
    html_output = shared.gradio[f'html-{mode}']
    interface = shared.gradio['interface_state']
    queue_options = dict(api_name=False, concurrency_id=f'rewrite-{mode}', concurrency_limit=1, show_progress='full')

    def updates(s, message, locked=False):
        enabled = not locked and not shared.args.multi_user
        value = {session: s, c['status']: message}
        value.update({component: gr.update(interactive=enabled) for component in controls})
        value[c['stop']] = gr.update(interactive=locked and s.phase in ('rewrite', 'automatic'))
        value[c['apply']] = gr.update(interactive=enabled and s.pending is not None and s.pending.get('kind') != 'automatic')
        value[c['undo']] = gr.update(interactive=enabled and bool(s.history))
        return value

    def retrieval(s, settings, top_k, length_mode, sentence_count, token_tolerance,
                  scoring, diversity, exclude_exact, nuance, rerank_pool,
                  min_length_ratio, min_semantic_score, max_contradiction_score, progress):
        """Validate once and share the exact retrieval setup across both strategies."""
        _guard()
        if s.index is None or s.encoder is None:
            raise ValueError('Build the corpus first.')
        corpus_config, embedding_config = _configs(settings)
        if corpus_config != s.config or embedding_config != s.embedding_config:
            raise ValueError('Corpus settings changed. Build the corpus before retrieving.')
        token_counter = None
        token_counter_key = None
        if length_mode == 'tokens':
            from modules.text_generation import get_encoded_length
            from modules.models import load_model_if_idle_unloaded
            load_model_if_idle_unloaded()
            tokenizer = shared.tokenizer
            model_name = shared.model_name

            def token_counter(value):
                if shared.tokenizer is not tokenizer or shared.model_name != model_name:
                    raise ValueError('Generation tokenizer changed during retrieval. Retry with the current model.')
                return get_encoded_length(value)
            if not shared.args.extensions:
                token_counter_key = (model_name, id(tokenizer))
        reranker = None
        if nuance:
            reranker_config = (embedding_config.device, embedding_config.local_files_only)
            if s.reranker is None or s.reranker_config != reranker_config:
                if s.reranker is not None:
                    s.reranker.release()
                s.reranker = NuanceReranker(device=embedding_config.device, local_files_only=embedding_config.local_files_only)
                s.reranker_config = reranker_config
            reranker = s.reranker
            reranker.load(lambda msg: progress(None, desc=msg))

        def retrieve(query):
            if s.cancel.is_set():
                raise InterruptedError('Stopped before retrieval.')
            hits = s.index.search(query, s.encoder, top_k=int(top_k), length_mode=length_mode,
                                  sentence_count=int(sentence_count), token_tolerance=float(token_tolerance),
                                  token_counter=token_counter, scoring=scoring, diversity=float(diversity),
                                  exclude_exact=exclude_exact, progress=lambda msg: progress(None, desc=msg),
                                  reranker=reranker, rerank_pool=int(rerank_pool), cancel_event=s.cancel,
                                  token_counter_key=token_counter_key, min_length_ratio=float(min_length_ratio),
                                  min_semantic_score=float(min_semantic_score) if nuance else 0.0,
                                  max_contradiction_score=float(max_contradiction_score) if nuance else 1.0)
            if s.cancel.is_set():
                raise InterruptedError('Stopped before generation.')
            return hits

        return retrieve

    def build(s, paths, recursive, model, revision, device, max_tokens, batch_size,
              offline, max_sentences, cleanup, join_hyphenated_lines, quality_policy, force, progress=gr.Progress()):
        if not s.lock.acquire(blocking=False):
            yield {c['status']: 'Another corpus operation is already running.'}
            return
        s.busy, s.phase, s.pending = True, 'build', None
        encoder = None
        committed = False
        cleanup_report = None
        quality_report = None
        try:
            yield updates(s, 'Building corpus. Controls are locked until indexing succeeds or fails.', locked=True)
            _guard()
            corpus_config, embedding_config = _configs([paths, recursive, model, revision, device,
                                                        max_tokens, batch_size, offline, max_sentences, cleanup, join_hyphenated_lines, quality_policy])
            key = hashlib.sha256(json.dumps([asdict(corpus_config), asdict(embedding_config)], sort_keys=True).encode()).hexdigest()
            index = CorpusIndex(shared.user_data_dir / 'retrieval_indexes' / key)
            s.build_path = index.path
            if not force and s.encoder is not None and s.embedding_config == embedding_config:
                encoder = s.encoder
            else:
                if s.encoder is not None:
                    s.encoder.release()
                encoder = LateInteractionEncoder(embedding_config)
            report = lambda message: progress(None, desc=message)
            encoder.load(report)
            manifest = index.build(corpus_config, encoder, report, force=bool(force))
            s.index, s.encoder = index, encoder
            s.config, s.embedding_config = corpus_config, embedding_config
            committed = True
            cleanup_report = _cleanup_report(manifest.get('cleanup'))
            quality_report = _quality_report(manifest.get('quality'))
            saved = {**asdict(corpus_config), **asdict(embedding_config)}
            path = _settings_path(mode)
            tmp = path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
            preference_warning = ''
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(saved, indent=2), encoding='utf-8')
                tmp.replace(path)
            except OSError as exc:
                preference_warning = f' Corpus settings could not be saved: {exc}'
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError as exc:
                    preference_warning += f' Temporary settings file could not be removed: {exc}'
            message = (f'Corpus ready: {manifest["files"]} files, {manifest["candidates"]} unique passages, '
                       f'{manifest["occurrences"]} occurrences. Model: {embedding_config.model}.{preference_warning}')
        except Exception as exc:
            logger.exception('Notebook corpus indexing failed')
            message = f'INDEX BUILD FAILED: {exc}\nCorrect the path/settings and click Build corpus / retry.'
            rejected_report = getattr(exc, 'quality_report', None)
            if rejected_report is not None:
                quality_report = 'Failed build — no new corpus was published.\n' + _quality_report(rejected_report)
                message += (f'\nQuality screening excluded {rejected_report.get("excluded_spans", 0)} sentence spans '
                            f'and {rejected_report.get("excluded_windows", 0)} candidate windows. See the ingestion quality report.')
            if s.index is not None:
                message += '\nThe previous completed corpus is retained; matching settings are required to use it.'
        finally:
            if encoder is not None and not committed and encoder is not s.encoder:
                encoder.release()
            s.build_path = None
            s.busy, s.phase = False, ''
            s.lock.release()
            if s.closed:
                s.dispose()
        completed = updates(s, message)
        if cleanup_report is not None:
            completed[c['cleanup_report']] = cleanup_report
        if quality_report is not None:
            completed[c['quality_report']] = quality_report
        yield completed

    def run(s, text, left, prompt_name, state, seed, guidance, use_template, review, thinking,
            top_k, length_mode, sentence_count, token_tolerance, scoring, diversity, exclude_exact, nuance, rerank_pool,
            min_length_ratio, min_semantic_score, max_contradiction_score,
            *settings, progress=gr.Progress(), generate=False):
        if not s.lock.acquire(blocking=False):
            yield {c['status']: 'Another corpus operation is already running.'}
            return
        s.busy, s.phase, s.pending = True, 'rewrite' if generate else 'search', None
        s.cancel.clear()
        try:
            initial = updates(s, 'Retrieving references…', locked=True)
            initial.update({c['result']: '', c['references']: '', c['target']: ''})
            yield initial
            retrieve = retrieval(s, settings, top_k, length_mode, sentence_count, token_tolerance,
                                 scoring, diversity, exclude_exact, nuance, rerank_pool,
                                 min_length_ratio, min_semantic_score, max_contradiction_score, progress)
            original = _source(mode, text, left)
            working, span = _seeded(original, seed)
            hits = retrieve(span.text)
            evidence = _evidence(hits, length_mode)
            retrieved = f'Retrieved {len(hits)} of requested {int(top_k)} references.'
            yield {c['target']: span.text, c['references']: evidence, c['result']: '', c['status']: retrieved}
            if not generate:
                message = f'{retrieved} Click Rewrite to generate with the current settings.'
            else:
                final = None
                generation_state = state.copy()
                generation_state['enable_thinking'] = bool(thinking)
                for event in generate_rewrite(working, hits, generation_state, guidance=guidance,
                                              use_template=use_template, cancel_event=s.cancel):
                    yield {c['result']: event.text, c['status']: event.status}
                    if event.done:
                        final = event
                if final is None or s.cancel.is_set():
                    raise InterruptedError('No completed rewrite was produced. Notebook text is unchanged.')
                after = replace_sentence(working, span, final.text)
                s.pending = dict(before=original, after=after, text=text, left=left,
                                 prompt=prompt_name, review=review, replacement=final.text, seed=seed,
                                 context_trimmed=bool(final.plan and final.plan.context_trimmed))
                message = f'Rewrite ready using {len(hits)} references.'
                if final.plan and final.plan.context_trimmed:
                    message += ' Older notebook context was trimmed to fit the model; every reference was retained.'
                if review:
                    message += ' Review the result, then click Apply rewrite.'
        except Exception as exc:
            if isinstance(exc, InterruptedError) or s.cancel.is_set():
                message = 'Stopped. Notebook text is unchanged; click Rewrite to try again.'
            else:
                logger.exception('Notebook retrieval/rewrite failed')
                message = f'REWRITE FAILED: {exc}' if generate else f'RETRIEVAL FAILED: {exc}'
            s.pending = None
        finally:
            s.busy, s.phase = False, ''
            s.lock.release()
            if s.closed:
                s.dispose()
        yield updates(s, message)

    def search(s, text, left, prompt_name, state, seed, guidance, use_template, review, thinking,
               top_k, length_mode, sentence_count, token_tolerance, scoring, diversity, exclude_exact,
               nuance, rerank_pool, min_length_ratio, min_semantic_score, max_contradiction_score,
               paths, recursive, model, revision, device, max_tokens, batch_size,
               offline, max_sentences, cleanup, join_hyphenated_lines, quality_policy, progress=gr.Progress()):
        yield from run(s, text, left, prompt_name, state, seed, guidance, use_template, review, thinking,
                       top_k, length_mode, sentence_count, token_tolerance, scoring, diversity, exclude_exact,
                       nuance, rerank_pool, min_length_ratio, min_semantic_score, max_contradiction_score,
                       paths, recursive, model, revision, device, max_tokens, batch_size,
                       offline, max_sentences, cleanup, join_hyphenated_lines, quality_policy, progress=progress, generate=False)

    def rewrite(s, text, left, prompt_name, state, seed, guidance, use_template, review, thinking,
                top_k, length_mode, sentence_count, token_tolerance, scoring, diversity, exclude_exact,
                nuance, rerank_pool, min_length_ratio, min_semantic_score, max_contradiction_score,
                paths, recursive, model, revision, device, max_tokens, batch_size,
                offline, max_sentences, cleanup, join_hyphenated_lines, quality_policy, progress=gr.Progress()):
        yield from run(s, text, left, prompt_name, state, seed, guidance, use_template, review, thinking,
                       top_k, length_mode, sentence_count, token_tolerance, scoring, diversity, exclude_exact,
                       nuance, rerank_pool, min_length_ratio, min_semantic_score, max_contradiction_score,
                       paths, recursive, model, revision, device, max_tokens, batch_size,
                       offline, max_sentences, cleanup, join_hyphenated_lines, quality_policy, progress=progress, generate=True)

    native_names = [f'prompt_menu-{mode}', f'Generate-{mode}', f'Stop-{mode}',
                    f'get_logits-{mode}', f'refresh_prompt-{mode}',
                    *[f'{action}_prompt-{mode}' for action in ('new', 'rename', 'delete')],
                    *[f'{action}_prompt-{suffix}-{mode}' for action in ('rename', 'delete')
                      for suffix in ('cancel', 'confirm')], f'rename_prompt_to-{mode}']
    native_names += ['Regenerate-notebook', 'Undo'] if mode == 'notebook' else ['Continue-default']
    native_controls = list(dict.fromkeys([notebook, input_text, prompt,
                                         *[shared.gradio[name] for name in native_names if name in shared.gradio]]))
    native_interactive = {component: component.interactive is not False for component in native_controls}
    native_stop = shared.gradio.get(f'Stop-{mode}')

    def automatic_updates(s, message, locked=False):
        result = updates(s, message, locked=locked)
        result.update({component: gr.update(interactive=False if locked else native_interactive[component])
                       for component in native_controls})
        if native_stop is not None:
            result[native_stop] = gr.update(interactive=True if locked else native_interactive[native_stop])
        return result

    def generation_dispatch(native_wrapper):
        """Return the opt-in route; disabled generation calls the native wrapper unchanged."""
        def dispatch(source, s, enabled, text, left, prompt_name, state, guidance, use_template, thinking,
                     automatic_max_sentences, top_k, length_mode, sentence_count, token_tolerance,
                     scoring, diversity, exclude_exact, nuance, rerank_pool,
                     min_length_ratio, min_semantic_score, max_contradiction_score,
                     *settings, progress=gr.Progress()):
            if not enabled:
                native = native_wrapper(source, state, prompt_name) if mode == 'notebook' else native_wrapper(source, state)
                try:
                    for document, rendered in native:
                        yield {notebook: document, html_output: rendered}
                finally:
                    native.close()
                return
            if not s.lock.acquire(blocking=False):
                yield {c['status']: 'Another corpus operation is already running.'}
                return
            s.busy, s.phase, s.pending = True, 'automatic', None
            s.cancel.clear()
            accepted, completed, context_trimmed = source, 0, False
            operation = None
            abandoned = False
            message = 'Automatic generation finished without a completed rewrite.'
            try:
                allowance = state.get('max_new_tokens', 'unset')
                initial = automatic_updates(s, 'Preparing automatic sentence generation… '
                                              f'Completed rewrites: 0 of {int(automatic_max_sentences)} maximum. '
                                              f'Draft allowance: {allowance} tokens shared across the run; '
                                              f'each rewrite has a separate allowance of up to {allowance} tokens.', locked=True)
                initial.update({c['auto_preview']: source, c['target']: '', c['result']: '', c['references']: ''})
                yield initial
                _guard()
                if s.index is None or s.encoder is None:
                    raise ValueError('Build the corpus first.')
                corpus_config, embedding_config = _configs(settings)
                if corpus_config != s.config or embedding_config != s.embedding_config:
                    raise ValueError('Corpus settings changed. Build the corpus before retrieving.')
                s.index.assert_current(s.config, s.encoder, lambda msg: progress(None, desc=msg), s.cancel)
                if s.cancel.is_set():
                    raise InterruptedError('Stopped while checking the corpus.')
                from modules import models
                from modules.utils import check_model_loaded
                models.load_model_if_idle_unloaded()
                loaded, error = check_model_loaded()
                if not loaded:
                    raise ValueError(error or 'No model is loaded.')
                tokenizer, model_name = shared.tokenizer, shared.model_name
                retrieve_base = retrieval(s, settings, top_k, length_mode, sentence_count, token_tolerance,
                                          scoring, diversity, exclude_exact, nuance, rerank_pool,
                                          min_length_ratio, min_semantic_score, max_contradiction_score, progress)

                def retrieve(query):
                    if shared.tokenizer is not tokenizer or shared.model_name != model_name:
                        raise ValueError('Generation model changed during automatic generation. Retry with the current model.')
                    return retrieve_base(query)

                from modules.sentence_rewrite.automatic import generate_automatic
                generation_state = dict(state)
                generation_state['enable_thinking'] = bool(thinking)
                operation = generate_automatic(source, generation_state, retrieve, guidance=guidance,
                                               use_template=use_template, cancel_event=s.cancel,
                                               max_sentences=int(automatic_max_sentences))
                for event in operation:
                    accepted, completed = event.document, event.completed
                    context_trimmed = context_trimmed or bool(getattr(event, 'context_trimmed', False))
                    message = f'{event.status} | Completed rewrites: {completed} of {int(automatic_max_sentences)} maximum.'
                    update = {c['auto_preview']: accepted, c['status']: message}
                    if event.target:
                        update[c['target']] = event.target
                    if event.phase == 'retrieving':
                        update[c['references']] = ''
                        update[c['result']] = ''
                    if event.hits is not None:
                        update[c['references']] = _evidence(event.hits, length_mode)
                    if event.phase == 'rewriting' or event.replacement:
                        update[c['result']] = event.replacement
                    yield update
            except GeneratorExit:
                abandoned = True
                s.cancel.set()
                raise
            except Exception as exc:
                if isinstance(exc, InterruptedError) or s.cancel.is_set():
                    message = 'Stopped automatic generation.'
                else:
                    logger.exception('Automatic Notebook generation failed')
                    message = f'AUTOMATIC GENERATION FAILED: {exc}'
            finally:
                try:
                    if operation is not None:
                        operation.close()
                except Exception as exc:
                    logger.exception('Automatic Notebook generation cleanup failed')
                    message += f' Generation cleanup failed: {exc}'
                finally:
                    if completed > 0 and not s.closed and not abandoned:
                        s.pending = dict(kind='automatic', before=text, after=accepted, text=text, left=left,
                                         prompt=prompt_name, review=False, replacement='', seed=None,
                                         context_trimmed=context_trimmed, automatic_status=message,
                                         completed=completed)
                    s.busy, s.phase = False, ''
                    s.lock.release()
                    if s.closed:
                        s.dispose()
            if completed > 0:
                message += f' {completed} completed rewrite(s) ready to apply; unfinished drafts were discarded.'
            else:
                message += ' Notebook text is unchanged; correct the problem or click Generate to retry.'
            yield automatic_updates(s, message)

        return dispatch

    commit_outputs = [session, notebook, html_output, interface, c['status'], c['apply'], c['undo'], c['seed']]
    if mode == 'notebook':
        commit_outputs.append(shared.gradio['last_input-notebook'])

    def commit(s, text, left, prompt_name, state, seed=None, automatic=False):
        _guard()
        pending = s.pending
        if pending is None or (automatic and pending['review']):
            return {session: s}
        with s.lock:
            if text != pending['text'] or left != pending['left'] or prompt_name != pending['prompt']:
                s.pending = None
                return {session: s, c['apply']: gr.update(interactive=False),
                        c['status']: ('Notebook text or prompt changed. Automatic generation was not applied. Click Generate again.'
                                      if pending.get('kind') == 'automatic' else
                                      'Notebook text or prompt changed. The proposed rewrite was not applied. Click Rewrite again.')}
            before, after = pending['before'], pending['after']
            s.history.append(dict(before=pending['text'], after=after, prompt=prompt_name, left=left))
            s.history[:] = s.history[-20:]
            s.pending = None
            state = dict(state)
            state['textbox-notebook' if mode == 'notebook' else 'output_textbox'] = after
            from modules.html_generator import generate_basic_html
            result = {session: s, notebook: after, html_output: generate_basic_html(html.escape(after)), interface: state,
                      c['status']: 'Rewrite applied. Edit the text or click Rewrite again to continue.',
                      c['apply']: gr.update(interactive=False), c['undo']: gr.update(interactive=True)}
            if pending.get('kind') == 'automatic':
                result[c['status']] = (f'{pending["automatic_status"]} '
                                       f'Applied {pending["completed"]} completed rewrite(s); unfinished drafts were discarded.')
            if pending.get('context_trimmed'):
                result[c['status']] += ' Older notebook context was trimmed to fit the model; every reference was retained.'
            if pending.get('kind') != 'automatic' and seed == pending['seed']:
                result[c['seed']] = ''
            if mode == 'notebook':
                result[shared.gradio['last_input-notebook']] = before
            # Single-column Notebook changes use its normal debounced autosave.
            # Two-column output follows the native output box's persistence behavior.
            return result

    def automatic_commit(*args):
        return commit(*args, automatic=True)

    def generation_commit(s, *args):
        # Ordinary generation may coexist with a reviewed manual proposal.
        # Its completion must neither apply that proposal nor run file guards.
        if s.pending is None or s.pending.get('kind') != 'automatic':
            return {session: s}
        return commit(s, *args, automatic=True)

    def undo(s, text, left, prompt_name, state):
        _guard()
        with s.lock:
            if not s.history:
                return {c['status']: 'There is no rewrite to undo.'}
            previous = s.history[-1]
            if text != previous['after'] or prompt_name != previous['prompt'] or (mode != 'notebook' and left != previous['left']):
                return {c['status']: 'The notebook changed since this rewrite. Undo was not applied.'}
            s.history.pop()
            s.pending = None
            state = dict(state)
            state['textbox-notebook' if mode == 'notebook' else 'output_textbox'] = previous['before']
            from modules.html_generator import generate_basic_html
            return {session: s, notebook: previous['before'], html_output: generate_basic_html(html.escape(previous['before'])),
                    interface: state, c['status']: 'Rewrite undone.', c['apply']: gr.update(interactive=False),
                    c['undo']: gr.update(interactive=bool(s.history))}

    def stop(s):
        _guard()
        if s.busy and s.phase == 'automatic':
            return stop_notebook_generation(s)
        if s.busy and s.phase == 'rewrite':
            s.cancel.set()
            return 'Stopping generation. The notebook will keep its previous text.'
        return gr.update()

    def clear(s):
        _guard()
        with s.lock:
            retained = False
            if s.index is not None:
                with _sessions_lock:
                    retained = any(other is not s and (
                        (other.index is not None and other.index.path == s.index.path)
                        or other.build_path == s.index.path) for other in _sessions)
                    if not retained:
                        s.index.clear()
            if s.encoder is not None:
                s.encoder.release()
            if s.reranker is not None:
                s.reranker.release()
                s.reranker = None
            s.index = s.encoder = s.pending = None
        message = 'Corpus cleared. Source .txt files were preserved.'
        if retained:
            message += ' The disk cache remains available to another open Rewrite tab that is using it.'
        result = updates(s, message)
        result[c['cleanup_report']] = ''
        result[c['quality_report']] = ''
        return result

    undo_inputs = [session, notebook, input_text, prompt, interface]
    commit_inputs = [*undo_inputs, c['seed']]
    run_inputs = [session, notebook, input_text, prompt, interface, c['seed'], c['guidance'], c['template'], c['review'], c['thinking'],
                  c['top_k'], c['length_mode'], c['sentence_count'], c['token_tolerance'], c['scoring'], c['diversity'], c['exclude_exact'],
                  c['nuance'], c['rerank_pool'], c['min_length_ratio'], c['min_semantic_score'],
                  c['max_contradiction_score'], *config_inputs]
    c['generation_inputs'] = [session, c['auto'], notebook, input_text, prompt, interface,
                              c['guidance'], c['template'], c['thinking'], c['auto_max_sentences'],
                              c['top_k'], c['length_mode'], c['sentence_count'], c['token_tolerance'],
                              c['scoring'], c['diversity'], c['exclude_exact'], c['nuance'], c['rerank_pool'],
                              c['min_length_ratio'], c['min_semantic_score'], c['max_contradiction_score'],
                              *config_inputs]
    c['generation_outputs'] = list(dict.fromkeys([notebook, html_output, session, c['status'], c['target'],
                                                 c['result'], c['references'], c['auto_preview'], *controls,
                                                 *native_controls]))
    c['generation_dispatch'] = generation_dispatch
    c['generation_commit'] = generation_commit
    c['generation_commit_inputs'] = undo_inputs
    c['generation_commit_outputs'] = commit_outputs
    c['build'].click(build, [session, *config_inputs, c['force']], outputs, **queue_options)
    c['clear'].click(clear, session, outputs, **queue_options)
    c['search'].click(search, run_inputs, outputs, **queue_options)
    c['rewrite'].click(ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], interface,
                       **queue_options).then(rewrite, run_inputs, outputs, **queue_options).then(
        automatic_commit, commit_inputs, commit_outputs, **queue_options)
    c['apply'].click(commit, commit_inputs, commit_outputs, **queue_options)
    c['undo'].click(undo, undo_inputs, commit_outputs, **queue_options)
    c['stop'].click(stop, session, c['status'], queue=False, api_name=False)

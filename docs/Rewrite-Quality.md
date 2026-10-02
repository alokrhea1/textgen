# Corpus quality and reference acceptance

The earlier literary comparison exposed two separate failures: OCR damage created apparently complete reference sentences, and retrieval could fill K with short or weak matches. This change screens candidate occurrences before embedding, then applies target-dependent length and semantic acceptance when searching. It preserves the existing period-based Notebook selection, trained token embeddings, native generation path, rewrite prompt, and review/apply/undo workflow.

## Plan and design decisions

1. Run every corpus through shared formatting cleanup, period segmentation, and inspectable quality screening. Keep source files unchanged and map retained/excluded occurrences to original decoded character offsets.
2. Exclude narrowly recognized damage before embedding, with reasons and neighboring context. Keep legitimate short text; its suitability depends on the target. Exclude windows containing rejected spans without joining across them.
3. Require useful reference substance relative to the target in sentence mode. Retain exact generation-token matching and tolerance in token mode.
4. Permit fewer than K references and an explicit no-reference outcome. Expose all model score components and make acceptance thresholds adjustable.
5. Compare old-style and new settings on original diagnostic text and separately on local literary corpora. Inspect both removed damage and incorrectly excluded legitimate writing. Run regression checks and independent source/workflow reviews before publishing.

The resulting pipeline is:

```text
UTF-8 source → shared formatting cleanup → period spans + neighboring context
            → occurrence quality flags / exclusions → eligible sentence windows
            → trained token matrices in SQLite

target → length eligibility → exact late-interaction scan → STS/NLI reranking
       → reference acceptance → up to K references → existing rewrite prompt
```

### Research that informed the choices

Unicode describes the ambiguity of periods and the need to tailor sentence boundaries to an application. Our design keeps the user's period-based Notebook behavior and adds a separate corpus-quality decision; it does not claim Unicode sentence-boundary conformance. [Unicode UAX #29, sentence boundaries](https://unicode.org/reports/tr29/#Sentence_Boundaries)

OCR detection and correction are separable tasks. Schaefer and Neudecker report that detecting erroneous sequences before correcting them reduces false changes to correct characters. That supports the decision to identify suspect source passages first. The implementation here uses narrow, inspectable rules and excludes suspect occurrences; it does not implement their learned corrector or infer missing prose. [A Two-Step Approach for Automatic OCR Post-Correction](https://aclanthology.org/2020.latechclfl-1.6/)

Sentence Transformers' multi-vector interface retains token vectors and uses MaxSim, with model-specific query/document roles. This remains the retrieval foundation. Replacing trained token representations with average sentence vectors would not address malformed source candidates. [Multi-vector encoder usage](https://sbert.net/docs/multi_vector_encoder/usage/usage.html)

LightOn reports strongly anisotropic token spaces in several ColBERT models, including LateOn. That is one reason raw cosine-derived scores should not be presented as calibrated relevance. Their investigation concerns efficient approximate retrieval; our exact scan does not use those approximations, and the report alone does not establish the cause of any particular literary ranking. No unvalidated centering or embedding transformation is applied here. [LightOn's late-interaction regularization investigation](https://huggingface.co/blog/lightonai/lateon-regularization)

Research on syntactic-template retrieval explicitly optimizes the usefulness of templates for generated paraphrases. That is a different objective from semantic similarity. We therefore do not label STS/NLI ranking a literary-style optimizer or add an arbitrary “interestingness” bonus. A learned structure/style retriever would need its own data and evaluation of resulting rewrites. [A Quality-based Syntactic Template Retriever for Syntactically-Controlled Paraphrase Generation](https://aclanthology.org/2023.emnlp-main.604/)

### Ingestion policies and provenance

`quality.py` runs for every corpus, independently of whether formatting cleanup is `none`, `conservative`, or `scanned_book`. `balanced` is the default policy; `off` records flags but excludes nothing on quality grounds. Changing policy requires a rebuild. Cleanup changes remain governed by the existing cleanup setting.

Balanced screening excludes replacement/control-character damage and punctuation-only spans. Quality version 2 applies additional rules only with `scanned_book`: concentrated separator noise, some suspicious adjacent period boundaries, passages ending in English `if`/`in`/`of`/`to` before a lowercase continuation, and narrowly defined digit-for-quote/contraction patterns. The function-word check also applies to longer passages with uppercase letters; it is not restricted to a standalone word. The quotation pattern recognizes one ASCII digit, while the contraction pattern remains restricted to `5` in its recognized English context. A detached single leading digit followed by a paragraph break and prose is separately flagged. Such a digit can also be an intentional numbered heading, so inspect exclusions.

Both sides of a suspect sentence boundary are flagged. Paragraph, quotation, and ellipsis boundaries are handled conservatively. These rules cannot detect all OCR damage and can still reject deliberate literary constructions. `off` is the escape hatch when inspection shows unwanted exclusions.

Short sentences remain indexable. The `short_reference` flag is informational, not a rejection reason. Letter/run counts are Unicode character statistics, not linguistic word segmentation; they do not establish sentence quality. Lowercase writing and non-Latin scripts are not rejected merely for being lowercase, unfamiliar, or unspaced. The scan-specific rules include English assumptions and are not a multilingual OCR detector.

Each occurrence retains source path, original start/end character offsets, quality flags, and up to 200 characters of cleaned context on either side. Excluded spans additionally retain their text and exclusion reasons in `excluded_spans`. Identical accepted passages still share an embedding while keeping occurrence provenance. An exclusion at one location does not exclude an intact identical passage elsewhere. Windows containing a rejected span are blocked, including windows that begin before it.

Schema 2 adds these audit fields and embedding content-token counts. Cleanup version, quality version/policy, model identity, source hashes, and other build settings participate in the cache signature. Normal **Build corpus / retry** rebuilds older-schema caches; publication remains transactional. A failed build preserves the previous completed cache. When all complete spans are rejected, the UI keeps an actionable quality report and permits retry rather than publishing an empty index. That failed build's staged database is removed, so only the bounded report remains available from it.

The UI report intentionally shows aggregate counts and a few samples, not the whole source corpus. On a completed cache, maintainers can inspect every rejected occurrence read-only. Supply the exact cache path rather than guessing which database belongs to a current tab:

```sh
python - /absolute/path/to/completed-corpus.sqlite <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve(strict=True)
with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
    db.execute('PRAGMA query_only=ON')
    db.row_factory = sqlite3.Row
    rows = db.execute('''
        SELECT source, start, end, text, flags, reasons,
               context_before, context_after
        FROM excluded_spans ORDER BY source, start, end
    ''')
    for row in rows:
        record = dict(row)
        for field in ('flags', 'reasons'):
            record[field] = json.loads(record[field])
        print(json.dumps(record, ensure_ascii=True))
PY
```

The output contains corpus text and server-local paths. Keep it with the private validation artifacts rather than committing it. For retained occurrences, join `occurrences.candidate` to `candidates.id`; `occurrences.quality_flags` and its context fields describe that particular location. Displayed spans and context are cleaned text; `start`/`end` address the original decoded Unicode text, including a BOM if present, not bytes or the cleaned string.

### Query-dependent acceptance

The UI's sentence-mode **Minimum reference length relative to target** defaults to 0.5. It compares embedding content-token counts, computed as token count minus the empty document's marker/special-token overhead. It is only a lower bound: a reference can be longer than the target. Zero disables it. Generation-token mode continues to use the current backend's exact token count and the selected tolerance; this floor does not apply there. Lowering the floor can be useful for deliberately terse references.

Top K remains 5 and the reranking pool remains 200 by default. Exact late interaction considers all eligible candidates, but STS/NLI sees only that retained pool. Raising the pool can recover a useful candidate ranked below the cutoff; it also increases inference cost. These defaults are not established as optimal for literary corpora.

With nuance reranking enabled, the UI defaults to STS similarity at least 0.3 and contradiction at most 0.8. The existing bidirectional composite score orders survivors. Minimum 0 and maximum 1 respectively disable those filters; disabling nuance makes both inactive. Fewer qualifying candidates means fewer references, and no qualifying candidate means a visible failure before generation. There is no fallback that substitutes rejected references to fill K.

These scores are raw model outputs transformed as described in the feature guide, not calibrated confidence or guarantees of equivalent meaning. The English STS/NLI checkpoints can reject good indirect stylistic analogies and perform poorly for other languages. Acceptance is configurable, and preview remains essential. There is no automatic corpus-specific threshold calibration and no explicit syntax/style ranker in this change.

## Reproducible evaluation

`tests/fixtures/rewrite_quality_cases.json` contains repository-original passages and diagnostic labels: contrasting composition, role reversal, negation, quantities, legitimate short replies, lowercase and non-Latin text, damaged period boundaries, uncertain broken spelling, and an unrelated query. These labels are small inspection probes, not a calibrated literary-quality benchmark or exhaustive judgments of all acceptable references.

The lead agent runs `tests/manual/rewrite_quality.py` with real models. Each run requires an empty output directory and writes `results.json`, model/source/settings identities, query references and raw source spans, retention probes for the synthetic fixture, timing, errors, and optional factual differences from an earlier run. A completed script or improved label count does not establish a general quality improvement. It evaluates retrieval; it does not run or grade sentence generation.

Retention probes declare any required cleanup/policy settings. `applicable` is false and `expectation_met` is JSON `null` when a probe's requirements do not match the evaluated configuration; this is neither a pass nor a failure. Inspect `probe_evaluation_policy` and the actual manifest configuration. For a legacy checkout that lacks `quality_policy`, the requested policy identifies the intended evaluation target only: it does not mean that checkout applied screening. The evaluator records that distinction rather than crediting a nonexistent feature.

For example, in the activated textgen environment with all three retrieval models cached:

```sh
python tests/manual/rewrite_quality.py --device cuda:1 --offline \
  --cleanup scanned_book --quality-policy off --min-length-ratio 0 \
  --min-semantic-score 0 --max-contradiction-score 1 \
  --output-dir /workspace/rewrite-validation/quality-controls-off

python tests/manual/rewrite_quality.py --device cuda:1 --offline \
  --cleanup scanned_book \
  --compare /workspace/rewrite-validation/quality-controls-off/results.json \
  --output-dir /workspace/rewrite-validation/quality-balanced
```

The first run approximates the old acceptance behavior on the new code; an actual old-checkout comparison must additionally record its revision and differences. Neither run should reuse a prior output directory. Use `--corpus /absolute/path/to/corpus.txt` (repeatable) to inspect local literary text separately. The fixture's relevance labels do not transfer to an external corpus. Keep private source passages and generated evidence outside the repository. To assess reranking-pool recall, repeat otherwise identical runs with `--rerank-pool 500` or another supported value and inspect changes rather than assuming a larger pool wins.

Maintainers should examine exclusions as well as results, and compare model identity/settings before attributing a change to ingestion. At minimum, verify damaged known boundaries stay out of embeddings, valid short/cased/uncased text remains usable, raw offsets are correct, duplicates retain occurrence-specific decisions, no window bridges an exclusion, and a weak-match refusal prevents generation without losing notebook text. Regression checks must cover normal token matching, both Notebook layouts, prompt budgeting, failure/retry, cancellation, review/apply/undo, and ordinary generation.

### Current validation record

The full Nabokov reranking-pool comparison inspected pools of 200, 500, 1,000, and 2,000 candidates. It did not show a consistent retrieval benefit from enlarging the pool, so the default remains 200. This does not establish that 200 is optimal for other queries or corpora, nor that reference style is adequately ranked.

The lead agent completed the quality-version-2 regression and retrieval checks below. The JSON artifacts record source hashes, code hashes, model identities, library versions, settings, and individual results. Private corpus passages remain outside the repository.

| Evidence | Observed result |
| --- | --- |
| Automated regression suite, with `REWRITE_TEST_CUDA=1` | **339 passed and 17 subtests**, with no skipped CUDA check. |
| Original nine-query fixture: ingestion | All **8 legitimate retention probes** stayed indexed; all **4 damaged-boundary probes** were excluded: **12/12 applicable probes** met their expectations. |
| Original nine-query fixture: ranking | First labeled relevant reference remained **rank 1 for all 8 queries** with a labeled relevant passage. Labeled discouraged references across those queries fell from **9 to 0**. |
| Original nine-query fixture: unrelated query | The baseline returned 5 references; the new configuration explicitly rejected retrieval with no qualifying references. |
| Nabokov, one-sentence windows, quality version 2 | **4,518 sentence spans assessed**, **155 excluded**, **4,363 retained occurrences**, and **4,336 unique candidates**. Exclusion-reason counts overlap when a span has several flags. |
| Nabokov, eight retrieval queries | **3/8 queries** returned an explicit no-reference result: expectation, institutional irony, and number. The degree query selected a useful 31-word reference first; the memory query still accepted a weak generic garden reference. |
| Both Notebook layouts and ordinary-generation regression | Passed with Gemma 4 12B IT in BF16 through Transformers: indexing locks and failure/retry, preview, weak-reference refusal without editing, review/apply/undo, stale-edit protection, seed/repeat, Stop/retry, exact generation-token matching, and ordinary Generate. Both Notebook layouts passed Rewrite/Undo. |
| Independent source and domain/workflow reviews | Three independent source, UI/workflow, and domain reviews approved after fixes, including the final quality-version-2 natural-corpus audit. |

The synthetic comparison used the same fixture hash, `59d2048d501a9e88dd32c6f562725538dcf7857bb30e176c53b566a82bca24ed`, in the baseline and new runs. Both used LateOn with real STS/NLI rerankers on `cuda:1`, a 384-token embedding limit, batch size 16, symmetric scoring, top K 5, reranking pool 200, one-sentence windows, and `scanned_book` cleanup. The baseline ran the pre-change retrieval code at `003ccab527a104e0f1d4c4a7217b910709d12497` with the evaluator added. The new run enabled balanced screening, a 0.5 content-token length floor, minimum STS 0.3, and maximum contradiction 0.8. The environment recorded PyTorch 2.8.0+cu128, Transformers 5.10.4, and Sentence Transformers 6.1.0.

Local evidence paths are `/workspace/rewrite-validation/quality-baseline-final/results.json`, `/workspace/rewrite-validation/quality-verified/results.json`, and `/workspace/rewrite-validation/quality-nabokov-final/results.json`. The private `/workspace/literary-corpus/retrieval-quality-comparison.html`, linked from the historical `comparison.html`, presents the before/after references. These files contain source passages and should not be committed or redistributed with the public code.

The browser workflow record is `/workspace/rewrite-validation/quality-browser-verified/browser-results.json`, with a screenshot in the same directory. It used isolated user data, Gemma on `cuda:0`, and retrieval on `cuda:1`. The browser fixture includes several valid alternatives so repeated rewrites can respect exact-copy exclusion and the new acceptance thresholds.

The lowercase diagnostic query still accepted some loosely related references, despite retaining the correct reference first. The Nabokov results likewise show that excluding damaged spans and refusing weak matches does not solve literary-style ranking. These are small diagnostic observations, not a style benchmark or a measured false-exclusion rate over the whole corpus.

The older 210-check/backend/installation record in [Notebook-Rewrite.md](Notebook-Rewrite.md) predates this change and does not validate these new heuristics or thresholds. No claim of literary-style superiority, universal OCR repair, or semantic perfection is made.

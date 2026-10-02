# Notebook sentence rewriting

The **Rewrite** subtab is available in both Notebook layouts. It retrieves examples from local text files and uses the currently loaded generation model to replace the last complete sentence. Retrieval and generation run only when requested; editing text never starts rewriting.

## Install and first use

Activate your existing textgen Python environment, then install the optional dependencies:

```sh
python -m pip install -r requirements/rewrite.txt
```

This adds Sentence Transformers and its dependencies. Model weights are separate downloads, and embeddings plus the optional reranker models can require substantial memory alongside your generation model. The UI defaults to CUDA; select **cpu** if CUDA is unavailable, or another available CUDA device to separate workloads.

For Gemma 4 12B Unified, use a separate environment with the normal webui dependencies installed, then install `requirements/rewrite-gemma4.txt`. This optional file includes the Rewrite dependencies and pins `transformers==5.10.4`, overriding the normal webui pin. The Transformers loader selects the multimodal loading path specifically when the model configuration declares `gemma4_unified`.

1. Load your generation model and open Notebook → **Rewrite**.
2. Enter local `.txt` files or directories, one path per line. These are server paths; relative paths start in the application directory. Files must be UTF-8 (a UTF-8 BOM is accepted). **Include subdirectories** defaults to enabled. Explicit symlink paths and symlink text files are rejected; symlink subdirectories are not traversed.
3. Click **Build corpus / retry**. Status and progress describe loading, hashing, reading, and embedding. Controls lock during operations and unlock on success or failure.
4. Optionally click **Preview references** to inspect text, scores, source paths, and character offsets. Then click **Rewrite**.
5. By default a successful proposal applies automatically. Enable **Review before applying** to inspect it and click **Apply rewrite** yourself. Application is refused if the notebook text, input, or selected prompt changed while the proposal was being prepared.

**Undo rewrite** retains up to 20 applied rewrites in the browser session and refuses to overwrite subsequent edits. Single-column Notebook uses its normal autosave; two-column output follows its existing persistence behavior. In the two-column layout, retrieval uses the output when nonempty, otherwise the input; the resulting document is written to output.

## Which sentence changes

Segmentation is deliberately period-based rather than linguistic. A terminating period, optionally followed by closing quotes or brackets, must be followed by whitespace or end of text. Question and exclamation marks alone do not complete a sentence. Ellipses can complete one; common abbreviations and single-letter initials are conservatively treated as incomplete, even at end of text. Dotted words remain intact. This can miss genuine endings such as `and so on etc.` or a sentence ending in `p.m.`; an earlier completed span may therefore remain the rewrite target.

Only the last completed span is replaced. Every character outside it—including an unfinished trailing fragment, whitespace, and following punctuation—is preserved exactly. The replacement must contain exactly one complete period-terminated sentence. During streaming the system waits for confirmation beyond the period, or validates completion when generation ends.

An **Optional seed sentence** must itself be exactly one complete sentence. It temporarily substitutes for the target in the working document and becomes the retrieval query and generation target. It does not change the notebook until a successful rewrite is applied. The seed is cleared after application only if it has not changed since the rewrite began. **Optional writing guidance** steers composition; references are examples rather than instructions.

## Models and retrieval settings

The default trained multi-vector checkpoint is [lightonai/LateOn](https://huggingface.co/lightonai/LateOn). The menu also offers [answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1), [mLateOn](https://huggingface.co/lightonai/mLateOn), and [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1). See their model cards for training, languages, and intended use. A custom Hugging Face ID or local trained ColBERT checkpoint is accepted; generic pooled sentence encoders are unsupported. Loading uses the native Sentence Transformers multi-vector encoder, trained projections, safetensors, and no remote code execution.

Use **Model revision** to pin a Hugging Face commit. **Offline: cached/local models only** prevents model downloads, including both reranker models; all required files must already be available. Local checkpoint fingerprints, model identity, library version, file hashes, and settings participate in cache validation. Changed corpus files or build settings require rebuilding. An unchanged index can be reused by clicking Build; **Force rebuild** bypasses reuse and can recover a corrupt cache. Failed builds retain the previously completed database. Correct the error and retry; matching current settings and unchanged sources are required to use that prior corpus. **Clear this corpus** releases this session's models and corpus, preserving source files. It deletes the selected cached database only when another open Rewrite tab is not using or building that shared cache; status reports retention.

| Setting | Default and meaning |
| --- | --- |
| Embedding token limit | 256; counts the embedding tokenizer's document marker and special tokens. Inputs are never silently truncated. |
| Embedding batch size | 16; lower it if indexing exhausts memory. |
| Index sentence windows | Up to 3 consecutive completed sentences, within each file; UI range 1–8. |
| Top K references | 5; UI range 1–50. Fewer are returned if fewer eligible candidates exist. |
| Match length: sentences | Exactly the selected sentence count, default 1, from already indexed windows. |
| Match length: tokens | Uses the current generation tokenizer, not the embedding tokenizer. Tolerance defaults to 0.15: absolute token-count difference must be at most 15% of the query count. Zero requires an exact count. |
| Late-interaction score | Symmetric by default: average of the two directional mean token MaxSim scores, using document-role encoding on both sides. Directional mode uses trained query/document roles. |
| Exclude exact copies | Enabled; excludes identical text after stripping outer whitespace, not paraphrases or case variations. |
| Reference diversity | 0; increasing it penalizes word-set overlap among selected references. This is textual diversity, not a semantic guarantee. |

Token-mode matching still searches only the sentence windows built into the index. It does not construct arbitrary token slices. A single sentence over the embedding limit fails indexing; longer multi-sentence windows over that limit are omitted. Incomplete fragments are not candidates. Change the window or token limits and rebuild when these restrictions leave too few matches.

Every corpus file passes through the shared cleanup implementation in `modules/sentence_rewrite/cleanup.py` (version 2). **Cleanup applied to every corpus file** defaults to `conservative`: it removes a leading BOM and soft hyphens, normalizes line endings and horizontal whitespace, joins single line wraps with spaces, and preserves blank-line paragraph boundaries. Unicode U+2028 becomes a line break; U+2029 and form feed become paragraph breaks. Choose `none` to retain decoded source formatting. Opt-in `scanned_book` additionally removes standalone decimal-number lines and `page`/`chapter` labels followed by decimal or Roman numerals, and joins OCR `¬` word wraps. This can remove meaningful numbers or chapter labels; inspect the read-only **Cleanup report**, which shows counts and before/after examples.

**Join words hyphenated across lines** defaults to disabled. When enabled with an active cleanup mode, it joins letter-to-letter hard-hyphen wraps; this can alter intended compounds. Cleanup handles formatting and recognized wrap artifacts, not OCR word correction. Source character offsets refer to the original decoded Unicode text, including a leading BOM, rather than the cleaned text or byte positions. A reported span can include removed interior formatting. Changes to cleanup mode, hyphen joining, or cleanup implementation version invalidate the index cache and require a rebuild.

Retrieval scans every eligible cached token matrix and computes exact late-interaction scores; there is no approximate vector index. CUDA scoring uses bounded float32 batches with padding excluded from token maxima; long pairs use bounded tiles. CPU scoring also uses exact tiled matching. This bounds scoring intermediates rather than loading the entire index onto the GPU. Punctuation is retained and token vectors are normalized. Repeated identical passages share an embedding while retaining occurrence provenance. Generation-token lengths are cached in memory by index generation and generation-model/tokenizer identity, then invalidated when either changes. With extensions enabled the UI avoids cross-search length reuse because extensions can alter tokenization. Changing the tokenizer during retrieval fails the operation rather than mixing counts. Length filtering, full scans, source hashing, and optional reranking can still be expensive on large corpora.

## Meaning and nuance

**Rerank for meaning and nuance** is enabled by default and loads both an [STS cross-encoder](https://huggingface.co/cross-encoder/stsb-roberta-large) and a [DeBERTa NLI checkpoint](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) on the selected embedding device. It reranks the top late-interaction pool, default 200 (at least K), using both query→candidate and candidate→query. The composite score is `average(sigmoid(STS logits)) + 0.25 × min(entailment probabilities) − 0.25 × max(contradiction probabilities)`, with each average/minimum/maximum taken across the two directions. This balances semantic similarity with evidence about entailment, negation, and participant roles. The score is a ranking heuristic, not a calibrated equivalence probability. Candidates outside the pool cannot be recovered by reranking. Both models check paired input lengths, including special tokens, in both directions before inference; exceeding either model's limit causes an error rather than truncation.

The preview exposes the composite nuance score, semantic similarity, entailment, contradiction, and late-interaction score. Neither retrieval nor generation guarantees preservation of every number, name, tense, implication, or qualifier. Review generated text for your use case. Small diagnostic checks of real models are limited evidence, not a general quality benchmark or a promise of perfect semantics.

## Generation, context, and stopping

Rewriting uses the application's native generation path, current sampling settings and token allowance, and optionally its selected instruction template. The default prompt explicitly preserves participants, actions, facts, polarity, degree, and numbers, and permits reference wording only when compatible with the target's meaning. Every retrieved reference and the target sentence remain in the prompt. Nearby notebook context is retained where possible; older context can be trimmed for the prompt without changing the saved document. If all references plus the target cannot fit, rewriting fails with instructions to reduce K/window size or increase available context. References are never silently dropped.

**Enable model thinking for this rewrite** defaults to disabled. It sets the thinking option for this Rewrite generation request; enable it when you want the loaded model's supported thinking behavior. Gemma testing exhausted both 256- and 1,024-token allowances with thinking enabled and produced no usable final sentence; leave it disabled for short rewrites unless you deliberately allocate a larger budget.

Generation stops at the first confirmed complete sentence. **Stop generation** sets this Rewrite session's cancellation event and preserves the previous notebook text. Cancellation is checked during retrieval source hashing, candidate scoring, nuance reranking, and native generation; model loading or an in-flight inference batch can delay its effect. The button does not set the global stop flag for unrelated generation. Custom stopping strings, exhausted token allowance, or model output that ends before a usable complete sentence can also produce an error; no partial replacement is applied. Native integration is implemented across loaders through shared generation and token counting. End-to-end browser validation passed with **Mistral-Nemo-Instruct-2407 and Gemma 4 12B IT using the Transformers loader**, including both layouts, Stop/retry, repeated seed rewriting, exact token matching, custom stopping strings, incomplete-output recovery, and subsequent ordinary Notebook generation. Other loaders have source/adapter checks but have not been runtime-validated.

## Storage, limits, and developer notes

Caches and per-layout build settings live under `user_data/retrieval_indexes` (or the configured user-data directory). They contain corpus text, local paths, and embedding matrices. New cache directories and SQLite files request private permissions (`0700`/`0600`); protect the containing user-data directory as well. Session proposals/history are not persisted. Local server-file retrieval is disabled in multi-user mode, including callbacks.

Current hard limits are 64 MiB per file, 512 MiB total source bytes, 10,000 source files, 200,000 candidate occurrences, and 8 GiB of unique embedding-matrix payload per index. The payload budget excludes SQLite metadata, text, provenance, and database overhead. A transactional rebuild temporarily needs space for both the previous completed database and its staged replacement; caches for other configurations also remain on disk. These are rejection bounds, not supported performance targets: multi-vector storage and inference can consume much more memory or disk than raw text. There is no directory watcher or automatic background rebuild.

Implementation is divided between `modules/ui_sentence_rewrite.py` (session state, controls, locking and guarded application) and `modules/sentence_rewrite/`: `sentences.py` (spans), `embeddings.py` (trained encoder and scores), `corpus.py` (transactional SQLite build/search), `reranker.py` (bidirectional STS and NLI), and `engine.py` (prompt planning and native generation). Builds stage a database and atomically replace the completed cache only after success. Per-session and shared-cache locks prevent overlapping mutations.

## Local literary corpus

An optional preparation script curates three English Project Gutenberg editions: Samuel Butler **1835–1902**, [Erewhon (#1906)](https://www.gutenberg.org/ebooks/1906) and [The Way of All Flesh (#2084)](https://www.gutenberg.org/ebooks/2084), plus Henry James, [The Ambassadors (#432)](https://www.gutenberg.org/ebooks/432). Their catalog entries identify them as public domain in the USA. Obtain the plain-text files as `pg1906.txt`, `pg2084.txt`, and `pg432.txt` in a source directory, then run:

```sh
python tests/manual/prepare_literary_corpus.py --source-dir /workspace/literary-corpus/sources --output-dir /workspace/literary-corpus/prepared
```

The original downloads, including their full source and license notices, remain unchanged in `/workspace/literary-corpus/sources`. Curated texts are written under `samuel_butler` and `henry_james` in the output directory; `manifest.json` records URLs, attribution, hashes, and removals outside those author text directories. The script validates the inspected edition markers before writing. It removes front matter/prefaces, headings, illustration labels, Erewhon's final footnotes and reference numbers, and James's editorial chapter-order note. It excludes chapters IV–V of *The Way of All Flesh*, which the editor says he reconstructed. Original prose wording and wrapping are preserved; line endings and blank spacing are normalized. Consult the [Project Gutenberg license](https://www.gutenberg.org/license) for distribution terms.

Enter `/workspace/literary-corpus/prepared/samuel_butler` and `/workspace/literary-corpus/prepared/henry_james` as separate server-local corpus paths. Keep full sources separate from prepared retrieval paths so headers, licenses, and editorial material do not become references. The default **Embedding token limit** for this literary sample is **384**; preparation itself applies no token-length filter, and indexing still rejects any single sentence over the configured limit. This sample setting does not change the UI default of 256.

The user-supplied Nabokov OCR source `/workspace/Untitled.txt` can be prepared separately:

```sh
python tests/manual/prepare_nabokov_corpus.py --source /workspace/Untitled.txt --output-dir /workspace/literary-corpus/prepared
```

This writes `/workspace/literary-corpus/prepared/vladimir_nabokov/lolita.txt` and a separate `nabokov_manifest.json` identifying the source as user-provided, with hashes and raw line boundaries. Exact inspected markers retain the narrative from raw line 180 through line 15460, excluding front matter and the author afterword. Every byte between those boundaries remains intact, including all narrative parts, page numbers, line wrapping, and OCR artifacts; the original file remains unchanged. No public-domain or redistribution permission is claimed for this source. Select `scanned_book` during ingestion to apply formatting cleanup to the prepared text. The current sample index contains **4,458 Nabokov candidates**; cleanup reports **73 furniture lines** and **503 OCR `¬` wraps**. Visible OCR word errors remain and can affect retrieval quality.

A local comparison saved 72 rewrites: eight neutral queries against each of these three author corpora, with Nemo at temperature 0.3 and Gemma at both 0.3 and 1.0. Each case uses identical references across models, seed 42, bf16, and thinking disabled. Nabokov cases received additional composition guidance, so cross-author differences do not isolate the corpus alone. In these examples Gemma preserved claims more reliably, while Nemo more often introduced meaning drift or omitted details. Both can still change implications; distinct author voice remains inconsistent. The standalone `/workspace/literary-corpus/comparison.html` contains the outputs and retrieved references. This is a small qualitative comparison, with one sample per condition, not an established general model ranking.

Final validation passed 143 checks with the Gemma environment, including the explicit CUDA scoring check. The original Transformers 5.6 environment passed 142 checks with that CUDA opt-in skipped. The `tests/test_rewrite_*.py` suites cover segmentation, token limits, cache invalidation and transactional failure, retrieval, reranking, prompt budgets, native generation adapters, and UI application guards. They use controlled fixtures for reproducible behavior; these checks do not establish semantic accuracy for arbitrary text or runtime support for every loader.

For an opt-in browser smoke workflow, install Playwright and its Chromium browser in your developer environment, start textgen with a generation model loaded and Rewrite dependencies installed, then run:

```sh
python -m pip install playwright
python -m playwright install chromium
python tests/manual/rewrite_browser.py --url http://127.0.0.1:7860 --device cuda:1 --output-dir rewrite-browser-artifacts
```

The default corpus is `tests/fixtures/rewrite_reference_sentences.txt`; override it with `--corpus /absolute/path/to/references.txt`. The script and server must share the same filesystem because resolved paths are submitted as server-local paths. Use a scratch Notebook/session: the script changes text, sampling settings, and layout, explicitly starts with one column, and sets seed 42 and temperature 0.3. It exercises failure/retry, lock states, previews, application/undo, stale-proposal protection, stopping, seed/repeat behavior, exact token matching, incomplete output and custom-stop errors, ordinary generation, and two-column output. JSON results are saved even on exceptions; a successful run also saves a screenshot. These live checks depend on the loaded model and hardware, and are not part of the default unit suite.

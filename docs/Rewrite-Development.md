# Rewrite development

This guide gives the code structure, maintenance contracts, and validation record for Notebook Rewrite.
For instructions and initial control values, refer to [the user guide](Notebook-Rewrite.md).
For screening rules and evaluation, refer to [the quality guide](Rewrite-Quality.md).
[Upstream maintenance](UPSTREAM-MAINTENANCE.md) gives the procedure for integration with a subsequent upstream release.

## Technical terms

These subject-field terms apply to the Rewrite documents.
Control labels, file names, model identifiers, and code identifiers keep their software spelling.
The documents use the writing rules and dictionary of [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).

| Technical noun | Meaning |
| --- | --- |
| Corpus | The group of source text files. |
| Span | A text region with start and end character offsets. |
| Window | One or more consecutive completed sentence spans from one file. |
| Candidate | An indexed window that can become a reference. |
| Occurrence | One candidate location in a source file. |
| Reference | A candidate that retrieval selects for the generation prompt. |
| Token | A text unit that a model tokenizer supplies. |
| Content-token count | The embedding token count minus the empty document's marker and special-token count, with a minimum of zero. |
| Embedding | A trained vector for a token. |
| Late interaction | Retrieval that compares contextual token vectors. |
| Mean MaxSim | The mean of each query token's maximum cosine similarity with a document token. |
| STS | Semantic textual similarity. |
| NLI | Natural language inference, with entailment and contradiction scores. |
| Prompt | The input that the generation model receives. |
| Answer prefix | Text that the system supplies at the assistant answer cursor before generation. |
| Cache | A completed SQLite index and its build identity. |
| Automatic mode | Notebook generation with a draft, retrieval, and rewrite cycle for each new sentence. |

Technical verbs identify software processes: `index`, `encode`, `tokenize`, `rerank`, `generate`, and `rewrite`.
`Index` makes the corpus cache. `Encode` makes trained token vectors. `Tokenize` converts text to model tokens.
`Rerank` calculates the sequence of candidates again with STS and NLI.

`Generate` gives model output. `Rewrite` gives a replacement sentence from a target and references.

Other computing terms include tokenizer, checkpoint, settings, inference, sampling, cache identity, and token budget.
The documents use computer-process verbs such as load, save, download, install, enable, disable, copy, validate, and score.
These terms identify software operations and data only.

## Module map

| Module | Function |
| --- | --- |
| `modules/ui_sentence_rewrite.py` | Session state, controls, cache locks, proposals, cancellation, guarded application, and undo. |
| `modules/sentence_rewrite/sentences.py` | Period boundaries, source spans, target selection, and replacement. |
| `modules/sentence_rewrite/cleanup.py` | Format cleanup and mapping to source character offsets. |
| `modules/sentence_rewrite/quality.py` | Quality flags and exclusion policy. |
| `modules/sentence_rewrite/embeddings.py` | Trained ColBERT encoders and exact token scoring. |
| `modules/sentence_rewrite/corpus.py` | Source hashes, transactional SQLite builds, candidate search, and audits. |
| `modules/sentence_rewrite/reranker.py` | Bidirectional STS and NLI inference. |
| `modules/sentence_rewrite/engine.py` | Rewrite prompts, native token budgets, and generation. |
| `modules/sentence_rewrite/automatic.py` | Sentence drafting, continuation, retrieval, rewrite cycles, and finite allowances. |
| `modules/ui_notebook.py`, `modules/ui_default.py` | Native generation routes in the two Notebook layouts. |
| `modules/text_generation.py` | Native generation, token counts, extension hooks, and per-request stops. |
| `modules/exllamav3.py`, `modules/llama_cpp_server.py`, `modules/tensorrt_llm.py` | Backend token budgets and request cleanup. |

## Maintenance contracts

### Text and application

- Keep Rewrite available in the two Notebook layouts.
- Do not start text generation from edits, indexing, or previews.
- Keep all characters before and after the selected manual span the same.
- Use the same period parser for targets, corpus spans, completion, and replacement.
- Keep seed substitution in the working copy until application.
- Keep proposal review, guarded application, stale-edit rejection, and guarded undo.
- Make sure that the input, output, and selected prompt did not change before application.
- Keep usual Notebook generation the same when automatic mode is off.

### Retrieval

The encoder uses Sentence Transformers' native `MultiVectorEncoder`, trained projections, and query/document roles.
Do not use pooled vectors or lexical search as alternatives.
The encoder normalizes token vectors and keeps punctuation.
Occurrences with the same text share an embedding and keep their source locations and quality decisions.

Symmetric scoring uses document encoding for the two inputs.
Its score is the mean of the two directional mean MaxSim scores.
Directional scoring uses trained query/document roles.
The scorer calculates late-interaction scores for all candidates that the length filters accept.
There is no approximate vector index.
CUDA uses float32 batches or tiles with memory limits. CPU uses tiles for exact token matching.
MPS encoding and reranking use CPU for exact token matching.

Sentence mode selects indexed windows with the specified sentence count.
Its length ratio uses embedding content-token counts.
Token mode uses the loaded backend's token count and tolerance.
Token mode does not use the embedding length ratio.
A tokenizer change during search stops the operation.
Extensions prevent token-length reuse across searches because they can change tokenization.

The rerankers score query-to-candidate and candidate-to-query pairs.
The composite score is:

```text
mean(sigmoid(STS logits))
+ 0.25 * min(entailment probabilities)
- 0.25 * max(contradiction probabilities)
```

The mean, minimum, and maximum apply across the two directions.
The checkpoints are `cross-encoder/stsb-roberta-large` and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
If a pair has more tokens than a model's token limit, the reranker rejects the pair without truncation.
The rerankers use only the selected late-interaction pool.
The selected pool size is K or more.
Acceptance filters can supply less than K references or stop retrieval with no reference.
Scores do not show meaning equivalence or literary style.

### Source and cache

- Do not change source files during cleanup or ingestion.
- Keep offsets in decoded source characters, with the initial byte-order mark if the source contains one.
- Offsets must identify source characters, not bytes or cleaned-text positions.
- Do not make windows across rejected spans.
- Make a replacement cache available only after a completed build.
- Keep the previous completed cache after an error or cancellation.
- Keep server-file retrieval disabled in multi-user mode, with callback guards.
- Do not commit corpora, caches, paths from private sources, model weights, or runtime files.

The cache identity includes source hashes, build settings, checkpoint identity, library versions, and these version identifiers:

- `SCHEMA_VERSION`: 2.
- Splitter identity: `period-spans-v1`.
- `CLEANUP_VERSION`: 2.
- `QUALITY_VERSION`: 2.

Changes to segmentation, encoding, cleanup, or screening must change the applicable identity or include a migration.
A usual build replaces a previous schema through a transactional rebuild.
Automatic-mode activation does not change this identity.
The builder uses permission masks `0700` for new cache directories and `0600` for SQLite files.
Access protection is also necessary for the containing user-data directory.

### Native generation

- Use the loaded model, native samplers, selected template, and extension hooks.
- Count the prompt tokens that the backend receives, with BOS and extension changes.
- Keep all references and the target in the prompt.
- If necessary, remove previous context from the prompt before the target or references.
- Show context reduction in the status.
- If the target and references contain more tokens than the context limit, stop with an error.
- Keep cancellation local to the Rewrite request.
- Close generation iterators and backend requests on all exits.
- Reject replacements that are not completed. Do not apply partial output.

### Automatic mode

Each cycle generates one completed sentence, finds references, rewrites the sentence, and continues from the accepted document.
The system keeps the initial completed spans the same.
An initial span that is not completed becomes part of the first target after sentence completion.
Manual seed and review controls do not apply.
The final state check prevents application if the input, output, or selected prompt changed.
One undo entry includes the applied run.

Stop or an error discards the temporary sentence and keeps accepted rewrites for guarded application.

The run shares a draft allowance across its cycles, with the loaded tokenizer's text-token estimate.
Model reasoning and discarded boundary lookahead use that allowance.
Each rewrite receives a token allowance for that rewrite only.
These allowances do not limit the total backend work or final document length.

Enable the template control for templated drafts.
Select an instruction template.
Templated drafts ask for only the next sentence.
For text that is not completed, the draft prompt supplies the same prefix at the assistant answer cursor when possible.
Active reasoning or recognized control markers select the full-sentence fallback.
The fallback must keep the copied prefix the same after output hooks.
The system uses only the suffix to complete the document before retrieval.

It does not add spaces or repair words in that suffix.

A supplied prefix counts against the prompt budget.
Only its generated suffix uses the draft allowance.
In the fallback, the generated copied prefix also uses that allowance.
Extensions must keep the prefix and native token cursor.
Supplied-prefix drafting does not accept external Transformers `inputs_embeds`.
Context reduction keeps the continuation instructions and the last sentence or fragment.

Templated end-of-response can stop a draft step while the run continues.
If the template control is disabled or no template is selected, drafts use raw Notebook continuation.
Raw terminal EOS stops the run. Custom stopping strings stop their route.
Decoder-only models are necessary for automatic mode.

## Verification procedure

The lead agent does tests and model experiments.
Reviewer agents examine source and can write tests. They do not do tests or benchmarks.
This prevents GPU experiments at the same time from changing results.
After retrieval or generation changes, reviewers must examine source and workflows independently.

1. Activate an environment with the selected installation profile.
2. Do the dependency check:

   ```sh
   python -m pip check
   ```

3. If pytest is missing, install it:

   ```sh
   python -m pip install pytest
   ```

4. Do the Rewrite regression tests:

   ```sh
   python -m pytest -q tests/test_rewrite_*.py
   ```

5. On a CUDA host, do the opt-in scoring test:

   ```sh
   REWRITE_TEST_CUDA=1 python -m pytest -q tests/test_rewrite_embeddings.py
   ```

6. Do the packaged-runtime check for the selected profile:

   ```sh
   python scripts/verify_portable_rewrite.py --torch-backend cuda --cuda-version 12.4
   ```

7. Do the real retrieval check in a temporary directory:

   ```sh
   python tests/manual/rewrite_retrieval.py --device cuda:0 --output-dir /tmp/rewrite-retrieval
   ```

The runtime-check example uses the CUDA 12.4 portable profile.
Other profiles use their compiled CUDA version, `--torch-backend rocm`, or `--torch-backend cpu`.
The real retrieval check downloads models unless `--offline` selects cached files.
A CPU retrieval check uses `--device cpu`.
The controlled suite does not download models or measure semantic accuracy for all prose.

### Browser workflows

Browser checks change Notebook text, settings, and layout.
Use a test server with isolated user data and a loaded generation model.
The script and server must share a filesystem for corpus paths.
Select new output directories for the browser results.

1. Install Playwright:

   ```sh
   python -m pip install playwright
   ```

2. Install Chromium:

   ```sh
   python -m playwright install chromium
   ```

3. Start the test server on an available port:

   ```sh
   python server.py --portable --listen-port 7861 \
     --user-data-dir /tmp/rewrite-user-data \
     --model-dir /path/to/models --model model.gguf
   ```

4. Do the manual workflow:

   ```sh
   python tests/manual/rewrite_browser.py --url http://127.0.0.1:7861 \
     --device cuda:0 --output-dir /tmp/rewrite-browser
   ```

5. Do the automatic workflow:

   ```sh
   python tests/manual/rewrite_automatic_browser.py --url http://127.0.0.1:7861 \
     --device cuda:0 --output-dir /tmp/rewrite-automatic
   ```

The initial corpus is `tests/fixtures/rewrite_reference_sentences.txt`.
`--corpus` selects a different server-local path.
The scripts record JSON results and screenshots.
Manual checks include build/retry, preview, seeds, rewrites, review/apply/undo, edits during operations, Stop, token matching, and usual generation.
Automatic checks include the six generation routes, sentence cycles, source selection, locks, Stop/retry, undo, and checkbox-off behavior.
Relaxed acceptance settings show the operation of controls. They do not show retrieval quality.

### Runtime coverage

Previous runtime checks used Linux with NVIDIA GPUs.
These records do not show compatibility after code or dependency changes.
Do the applicable checks again for the selected environment.

| Mode | Model and loader coverage |
| --- | --- |
| Automatic and manual Rewrite | Gemma 4 12B IT BF16 through Transformers. Mistral-Nemo-Instruct-2407 Q4_K_M through llama.cpp. |
| Other manual Rewrite coverage | Nemo BF16 through Transformers, ExLlamav3, and ExLlamav3_HF. Nemo GGUF through ik_llama.cpp. |
| TensorRT-LLM | Controlled adapter and worker tests only. No completed real engine build or generation validation. |

The automatic checks used Python 3.12.3, PyTorch 2.8.0+cu128, and Transformers 5.10.4.
They also used Sentence Transformers 6.1.0, Gradio 4.37.2+custom.21, and llama.cpp binaries 0.138.0 CUDA 12.4.

Linux Python 3.13 installation checks included these profiles:

| Profile | Packages and scope |
| --- | --- |
| CUDA 12.4 portable | PyTorch 2.6.0+cu124. Dependency checks, imports, and browser workflow with GPU generation and retrieval. |
| CPU portable | PyTorch 2.9.0+cpu. Dependency checks, imports, and real LateOn/SQLite/MaxSim/STS/NLI retrieval. |
| Full NVIDIA | PyTorch 2.9.0+cu128. Nemo BF16 browser workflow with ExLlama 0.0.34 and Flash Attention 2.8.3. |

Those installation checks used Transformers 5.6.2 and Sentence Transformers 6.1.0.
Windows, macOS, ROCm, CUDA 13.1, Docker builds, and Colab execution had no runtime validation in those checks.
TorchAO-quantized checkpoints also had no validation.
Do checks for other models, extensions, samplers, and automatic-mode loaders before their first operation.

## Corpus preparation tools

`tests/manual/prepare_literary_corpus.py` processes specified Project Gutenberg editions:
`pg1906.txt` (*Erewhon*), `pg2084.txt` (*The Way of All Flesh*), and `pg432.txt` (*The Ambassadors*).
It validates edition markers before output.
It removes specified front matter, headings, editorial text, footnotes, and reconstructed chapters IV–V of *The Way of All Flesh*.
It keeps narrative words and line wraps, with normalized line endings and blank spacing.
Its manifest records attribution, source URLs, hashes, and removals.

```sh
python tests/manual/prepare_literary_corpus.py \
  --source-dir /path/to/sources --output-dir /path/to/prepared
```

The source and output directories must be different. Neither directory can contain the other.
The script does not change source files.
Corpus paths select the prepared author directories, without the source notices or manifest.
Examine the source license before distribution.

`tests/manual/prepare_nabokov_corpus.py` processes one previously examined user-supplied OCR edition with fixed text markers.
It is not a general book extraction tool.
It keeps all bytes between the selected narrative boundaries and writes a source manifest.
It does not give redistribution permission.

```sh
python tests/manual/prepare_nabokov_corpus.py \
  --source /path/to/source.txt --output-dir /path/to/prepared
```

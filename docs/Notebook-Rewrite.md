# Notebook sentence rewriting

The **Rewrite** tab uses examples from a local text corpus to rewrite sentences with the loaded generation model.
It is available in the two Notebook layouts.
A corpus is a group of reference text files.
A reference is a passage that the retrieval system selects from that corpus.

Manual Rewrite replaces the last completed sentence.
Automatic mode rewrites each new completed sentence during Notebook generation.
Text edits, corpus builds, and reference previews do not start text generation.

For code changes, refer to [Rewrite development](Rewrite-Development.md).
For screening rules and audits, refer to [Rewrite quality](Rewrite-Quality.md).

## Installation

Use an installation or portable archive from this fork.
The supported installation profiles include the Rewrite dependencies.
The system downloads model files before the first operation.
Retrieval models use memory in addition to the generation model.

To add missing Rewrite dependencies:

1. Activate the textgen Python environment.
2. Make sure that it contains PyTorch for your hardware.
3. Install the Rewrite dependencies:

   ```sh
   python -m pip install -r requirements/rewrite.txt
   ```

Python 3.10 or a subsequent version is necessary.
Rewrite does not support native Intel macOS.
The full installer uses Python 3.13.

Retrieval uses PyTorch independently of the generation backend.
The initial device selection is CUDA, then Apple MPS, then CPU, if available.
Saved corpus settings can change this selection.

| Installation profile | Retrieval device |
| --- | --- |
| NVIDIA CUDA | CUDA. |
| Linux AMD | ROCm. Select the PyTorch device name `cuda`. |
| Windows AMD | CPU. |
| CPU or Vulkan | CPU. |
| Apple Silicon | MPS or CPU. MPS uses CPU for exact token matching. |

For Gemma 4 Unified, use a different environment with the usual textgen dependencies.
Install `requirements/rewrite-gemma4.txt` in that environment.
This file selects Transformers 5.10.4.
One-click updates keep that selection.
`REWRITE_GEMMA4=1` enables it, and `REWRITE_GEMMA4=0` selects the usual version.
After a manual installation of the base dependencies, install the Gemma file again.

## Build a corpus

Corpus paths refer to files on the server.
Relative paths start in the application directory.
Files must use UTF-8. A UTF-8 byte-order mark is permitted.
The system rejects symbolic links to files and explicit paths through symbolic links.
It does not use symbolic links to subdirectories.

1. Open Notebook.
2. Select **Rewrite**.
3. Enter one `.txt` file or directory path on each line.
4. Select the corpus settings.
5. Click **Build corpus / retry**.
6. Examine the **Cleanup report**.
7. Examine the **Ingestion quality report**.

**Include subdirectories** is initially enabled.
Controls lock during a build and unlock at its end.
If the build stops with an error, correct the cause.
Then, start the build again.
If a build has an error, the system keeps the previous completed cache.
To use that cache, the settings and source files must be the same as before.

**Build corpus / retry** uses a cache again if the settings and source files did not change.
**Force rebuild** makes a new cache if the settings and source files are the same.
**Clear this corpus** releases this tab's corpus and retrieval models.
It deletes the selected cache only if no other Rewrite tab uses or builds it.
The system does not change source files.

## Rewrite one sentence

1. Load a generation model.
2. Enter the text in Notebook.
3. Click **Preview references**.
4. Examine the reference text, scores, and source locations.
5. Click **Rewrite**.

An accepted rewrite applies immediately unless **Review before applying** is enabled.
With that control enabled, **Apply rewrite** applies the proposal.
If the input, output, or selected prompt changed during the operation, the system rejects application.
**Undo rewrite** keeps a maximum of 20 changes in the browser session.
It does not overwrite subsequent edits.

In the single-column layout, Rewrite uses the Notebook text.
In the two-column layout, it uses output if the output contains text. If output is empty, it uses input.
The system writes the result to output.
The single-column layout uses its usual autosave.
The two-column layout keeps its usual storage behavior.

Source locations identify characters in the initial decoded text.
They are not byte positions or positions in the cleaned text.

### Sentence selection

Only a terminating period completes a sentence.
The period can have closing quotes or brackets after it.
Whitespace or the end of the text must come after that boundary.
Question marks and exclamation marks do not complete a sentence without a period.
Ellipses can complete one.
Some abbreviations and initials can prevent sentence completion at the end of the text.

Manual Rewrite replaces only the last completed sentence.
The system keeps all other characters the same.
This includes a fragment after the selected span.
The replacement must be one completed sentence with a terminating period.

**Optional seed sentence** supplies a different retrieval query and rewrite target.
It must contain one completed sentence only.
It changes the working copy before application.
After application, the seed clears only if its text did not change during the operation.
**Optional writing guidance** supplies additional instructions to the generation model.

## Corpus settings

These are the initial settings. Saved build settings can replace them.

| Setting | Initial value and function |
| --- | --- |
| Trained multi-vector embedding model | `lightonai/LateOn`. A trained ColBERT model supplies token embeddings for retrieval. |
| Model revision | Empty. A Hugging Face commit identifies a fixed model version. |
| Offline: cached/local models only | Disabled. When enabled, all retrieval models must be local or cached. |
| Embedding token limit | 256. This includes document markers and special tokens. |
| Embedding batch size | 16. A smaller batch uses less memory. |
| Index windows up to this many sentences | 3. The control lets you select 1–8 consecutive sentences for each window. |
| Cleanup applied to every corpus file | `conservative`. The table below gives the function of each mode. |
| Join words hyphenated across lines | Disabled. This control can change compound words that you want to keep. |
| Ingestion quality screening | `balanced`. This rejects spans with specified signs of damage. |

The model menu also contains `answerdotai/answerai-colbert-small-v1`, `lightonai/mLateOn`, and `lightonai/GTE-ModernColBERT-v1`.
A custom Hugging Face identifier or local trained ColBERT checkpoint is permitted.
Rewrite does not support pooled sentence encoders.

The index contains completed sentence windows from each file.
It does not contain fragments that are not completed or token slices.
A single sentence that screening accepts stops the build with an error if its token count is more than the embedding limit.
The build does not include longer windows with more tokens than that limit.

| Cleanup mode | Function |
| --- | --- |
| `none` | Keeps the decoded source format. |
| `conservative` | Removes the initial byte-order mark and soft hyphens. Changes whitespace and line endings to the selected format. Puts single line wraps together. Keeps paragraph boundaries. |
| `scanned_book` | Adds removal of standalone number lines and numbered page or chapter labels. Puts OCR `¬` word wraps together. |

Cleanup does not correct OCR word errors.
The `scanned_book` mode can remove numbers or headings that you want to keep.
Hard-hyphen joining can remove hyphens that you want to keep.
Examine the reports before retrieval.

Balanced screening rejects spans with replacement characters, non-whitespace control characters, or punctuation alone.
With `scanned_book`, it also applies limited OCR and sentence-boundary rules.
It can reject literary constructions that you want to keep.
Short sentences, lowercase prose, and non-Latin scripts stay permitted input.
The `off` policy keeps flagged spans without quality exclusions.
Cleanup also applies.

After changes to source files, embedding settings, windows, cleanup, or screening, make the index again.
Changes to search acceptance settings do not make a corpus rebuild necessary.

## Reference selection

| Setting | Initial value and function |
| --- | --- |
| Top K references | 5. The control lets you select 1–50 references. |
| Match length by | `sentences`. Selects windows with the specified sentence count. |
| Sentences per reference | 1. |
| Minimum reference length relative to target | 0.5. In sentence mode, the reference must contain half the target's embedding content-token count or more. Zero disables this limit. |
| Generation-token length tolerance | 0.15. The maximum difference is 15% of the query token count. Zero makes the counts the same. |
| Late-interaction score | `symmetric`. Compares token matches in the two directions. |
| Exclude exact copies of the query | Enabled. Ignores outer whitespace when it compares text. Letter case must also be the same. |
| Reference diversity | 0. Higher values decrease word-set overlap among references. |
| Rerank for meaning and nuance | Enabled. Uses English STS and NLI models. |
| Late-interaction candidates to rerank | 200, or K if K is larger. The rerankers use only this pool. |
| Minimum semantic similarity | 0.3. Zero disables this filter. |
| Maximum contradiction score | 0.8. One disables this filter. |

Token mode uses the loaded generation tokenizer.
Sentence mode uses embedding content-token counts for its minimum length ratio.
Neither mode makes windows that the build did not include.

The system can supply less than K references.
If the system rejects all references, generation does not start.
There is no substitution with rejected references.
If you disable meaning-and-nuance reranking, the system disables its semantic and contradiction filters.

Scores are model outputs, not confidence percentages.
The English rerankers can reject style examples or text in other languages.
Retrieval and generation cannot make sure that the meaning or an author's style stays the same.
Examine facts, names, numbers, and qualifiers in the result.

## Generation controls

Rewrite uses the native generation path and the selected sampling settings.
**Use the selected instruction template** is initially enabled.
**Enable model thinking for this rewrite** is initially disabled.
Model thinking can use the token allowance before the model supplies a completed replacement.

The prompt keeps the target and all selected references.
It can remove previous Notebook context from the prompt to stay in the context limit.
The status shows this change.
If the target and references are too long for the context limit, the operation stops with an error.
If the prompt is too long, decrease K.
A smaller reference length or a larger context limit also gives more space.

**Stop generation** cancels this Rewrite operation.
The system can wait for model loading or an active inference batch before cancellation.
A replacement that is not completed, or a model error, prevents manual application.
A token limit or custom stopping string can leave the replacement without a completed sentence.

## Automatic mode

1. Load a generation model.
2. Build a corpus.
3. Enable **Automatically retrieve and rewrite generated sentences**.
4. Set **Maximum sentences per automatic generation**.
5. Start Notebook generation.

The checkbox is initially disabled. The browser session keeps its state.
The initial sentence limit is 5. The control lets you select 1–100.
The model generates one sentence, finds references, and rewrites that sentence before it continues.
The system keeps completed sentences from the initial document the same.
An initial sentence that is not completed can change after the model completes and rewrites it.

| Control | Starting text |
| --- | --- |
| Generate or Shift+Enter, either layout | Input. |
| Regenerate, single-column layout | Input from the previous generation request. |
| Continue, two-column layout | Output. Empty output also applies. |

Plain Enter inserts a newline.
Automatic mode uses the reference settings, writing guidance, template selection, and rewrite thinking control.
It ignores the manual seed and review controls.
A decoder-only generation model is necessary.
Automatic mode does not support encoder-decoder models.

With **Use the selected instruction template** enabled and a template selected, the draft prompt asks for the next sentence.
For input that is not completed, the template can use the same fragment as an answer prefix.
If it cannot use that prefix, it asks for the completed sentence.
It makes sure that the copied prefix is the same.
A changed prefix stops the operation before retrieval.
If the control is disabled or no template is selected, drafting uses raw Notebook continuation.

The initial **Max new tokens** is a shared estimated draft allowance for the run.
Model reasoning and discarded lookahead use that allowance.
Each rewrite receives an allowance from the same setting for that rewrite only.
These limits do not give the total backend token count or final document length.
Automatic mode uses more generation work than usual Notebook generation.

A usual template response can stop one draft while the operation continues.
Raw end-of-output stops the run.
Custom stopping strings stop either route.
Stop or an error discards the temporary sentence.
Accepted rewrites can apply after the final check of the input, output, and prompt.
An edit during the operation prevents application.

One **Undo rewrite** operation puts back the text from before the applied run.

If you disable the checkbox, the system uses usual Notebook generation again.
Automatic mode does not change corpus cache format.
Activation does not make a corpus rebuild necessary.

## Storage and limits

Caches and build settings are under `user_data/retrieval_indexes`, or the selected user-data directory.
Caches contain source text, local paths, and embeddings.
The browser session keeps proposals and undo history.
Multi-user mode does not let Rewrite retrieve server files.
There is no directory watcher or background rebuild.

| Per-index limit | Maximum |
| --- | --- |
| One source file | 64 MiB. |
| Total source bytes | 512 MiB. |
| Source files | 10,000. |
| Examined sentence spans | 200,000. |
| Examined windows and exclusions | 200,000. |
| Kept candidate occurrences | 200,000. |
| Unique embedding payload | 8 GiB. |

These are rejection limits, not performance targets.
Text, metadata, audit records, and database overhead use additional storage.
A rebuild uses space for the previous database and its replacement.
Other cache configurations also stay on disk.

# Notebook retrieval rewrite: implementation plan

Upstream baseline: `c93f8871239550de2ccfe1e95d469aa82616f07e`.

The requested workflow is an explicit Rewrite button in a sixth Notebook subtab. A local TXT corpus supplies top-K examples for regenerating the last completed period-delimited sentence. Users may edit a seed sentence and repeat indefinitely. Ordinary generation must remain unchanged.

## Design and implementation sequence

1. Preserve source spans with one period-aware sentence parser shared by indexing, target selection, streaming completion, and replacement. Preserve both preceding text and incomplete trailing text exactly.
2. Use trained ColBERT contextual token matrices, with Sentence Transformers' native MultiVectorEncoder. Compare model-native query/document mean MaxSim with symmetric document/document mean MaxSim. Provide GTE-ModernColBERT and AnswerAI ColBERT-small choices. Do not substitute pooled vectors or lexical search for the requested multi-vector retrieval.
3. Build a persistent, private SQLite corpus from explicit files and recursive directories. Index bounded contiguous sentence windows, deduplicate text, preserve provenance, validate lengths, fingerprint inputs/model settings, and activate only complete indexes. Display progress; disable all tab controls during construction; restore controls with an explicit error and retry on failure.
4. Search every eligible candidate using bounded-memory late interaction and a bounded top-K heap. Offer sentence-count and generation-token-length eligibility, exact-token matching, optional diversity, exact-match exclusion, and transparent result scores/source excerpts.
5. Render references and the target using native templates or a plain completion prompt. Budget the fully rendered prompt with the loaded model tokenizer. Keep all selected references or fail explicitly; trim older context first. Use native generation, samplers, hooks, and stop handling. Stage output before applying it to a fresh, unchanged source snapshot.
6. Expose the workflow in both Notebook layouts, with an editable seed, retrieval-only preview, optional review before applying, guarded application and undo, cache reuse/rebuild/clear, and optional offline model loading. Never trigger generation from edits, tab selection, or indexing.
7. Validate centrally: parser/index/scoring/lifecycle contracts, real embedding models, real Gradio callbacks/browser behavior, and native generation integration. Agents author/review code but do not run tests or benchmarks. Compare scoring approaches on a small diagnostic corpus; report its limited scope.
8. After implementation, recruit fresh independent review agents plus an adversarial reviewer. Address findings and obtain agreement from every reviewer before completion.

## Integration constraints

- Keep heavy retrieval dependencies optional and lazily imported.
- Preserve native loader abstraction and close generation iterators on every exit.
- Do not expose corpus files or persisted indexes through Gradio file serving.
- Keep session state and cancellation separate; validate stale edits before apply/undo.
- Disable server-file operations in multi-user mode with backend checks.
- Treat corpus excerpts as prompt data and load retrieval models without remote code execution.
- Do not silently truncate sentences, ignore unreadable inputs, reuse stale indexes, replace failed output, or claim retrieval scores measure factual equivalence.

## Evidence required for completion

Tests and runtime checks must cover complete and failed indexing with retry, persisted cache reuse/invalidation, exact replacement and repeated manual rewriting, both length modes, actual token matrices and late interaction, stop/error recovery, preview/apply/undo conflicts, UI control locking, and coexistence with normal Notebook generation. Backend-specific limits and any untested combinations must be stated accurately.

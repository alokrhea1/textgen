# Working on this fork

This is `alokrhea1/textgen`, with native Notebook sentence rewriting built on
oobabooga/textgen. The normal updater pulls this fork; integrating a new upstream
release is a separate maintenance task.

Read [the upstream maintenance guide](docs/UPSTREAM-MAINTENANCE.md) before changing
upstream integration, dependencies, installers, or backend adapters. Read
[the current feature guide](docs/Notebook-Rewrite.md) for shipped behavior.
Refer to [the developer guide](docs/Rewrite-Development.md) for code contracts and
verification procedures.

## Contracts to preserve

- Keep Rewrite available in both Notebook layouts. Editing, indexing, and preview
  must not start generation. Preserve ordinary generation behavior outside the
  explicit Rewrite request.
- Use trained contextual token embeddings and late interaction, not pooled-vector
  substitutes. Preserve sentence/token length eligibility and source provenance.
- Clean every corpus through the shared ingestion pipeline without modifying
  source files. Publish a new cache only after a successful complete build. Change
  cache identity deliberately when indexing/encoding/segmentation behavior changes.
- Preserve every character outside the selected sentence. Keep seed/repeat,
  review/apply, guarded undo, stale-edit protection, visible errors, and retry.
- Count the tokens actually sent by the loaded backend, including its BOS and
  extension behavior. Keep all references or fail explicitly; never silently
  truncate them. Preserve cancellation and backend cleanup on every exit.
- Keep retrieval dependencies included in every supported installation path and
  preserve the fork's source during installation/updates. Portable CUDA includes
  GPU retrieval. Consult the guide before changing the Gemma or TensorRT stacks.
- Keep server-file retrieval disabled in multi-user mode. Do not stage private
  corpora, indexes, credentials, model weights, environments, or runtime artifacts.

## Verification and reviews

The lead agent runs tests and model experiments. Reviewer agents inspect source
and may author tests, but do not run tests or benchmarks. This prevents competing
GPU runs from invalidating results.

Use independent review for integration changes and a domain/workflow review when
retrieval or writing behavior changes. Fix material findings before handoff.
Document what was actually tested; old pass counts and mocked adapter tests do not
establish compatibility with a new upstream version, model, or platform.

Keep published history intact. Prepare upstream updates on a separate branch or
worktree; do not reset this fork to upstream or force-push an update. Update
`docs/upstream-sync.json` only after the selected upstream commit has been merged
and the relevant checks have passed. Follow the user's existing authorization for
committing and pushing.

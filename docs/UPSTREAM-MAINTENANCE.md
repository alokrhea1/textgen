# Updating this fork from upstream

Use this guide when integrating a new oobabooga/textgen release into
`alokrhea1/textgen`. It describes maintenance of the source fork. The ordinary
update wizard only pulls an installed copy's configured fork branch; it does not
perform this integration for you.

## Starting point

The machine-readable record is [upstream-sync.json](upstream-sync.json). It records
the **last integrated upstream commit**, not the latest available release and not
the fork's HEAD. The initial baseline is
`c93f8871239550de2ccfe1e95d469aa82616f07e`, an upstream development checkout; no
release tag is claimed for it.

The first fork commits are `d06e7b6c` (Notebook Rewrite) and `b353d397` (installation
integration and backend hardening). They explain the original patch, but future
updates must compare against the current sync record rather than replaying those
two commits or assuming this list is exhaustive.

[Notebook-Rewrite.md](Notebook-Rewrite.md) describes current behavior, tested
combinations, and limitations. [Notebook-Rewrite-Plan.md](Notebook-Rewrite-Plan.md)
is historical. Preserve the native integration; this feature uses the loaded
generation model and its normal templates/samplers rather than a separate text
generation service.

## Inspect a candidate without changing application code

Start from a clean checkout containing the latest fork changes. Check `git status`
and `git remote -v`; `origin` is the user's fork and `upstream` is oobabooga/textgen.
If `upstream` is absent, add it with
`git remote add upstream https://github.com/oobabooga/textgen.git`. Do not repoint
`origin`. The older `oobabooga/text-generation-webui` URL is a valid existing
upstream remote for this checkout.

Select a concrete upstream release and read its release notes and dependency
changes. The examples below use a placeholder `vX.Y`; substitute the actual tag.
Keep upstream tags in a separate namespace so they cannot replace fork release
tags:

```sh
git fetch --no-tags upstream refs/tags/vX.Y:refs/upstream-tags/vX.Y
python scripts/upstream_status.py refs/upstream-tags/vX.Y
python scripts/upstream_status.py refs/upstream-tags/vX.Y --json
```

The helper uses only Git and the Python standard library. It checks baseline
ancestry, prints both commits, lists incoming/fork changes and their intersecting
paths, and flags a dirty tree. It issues local Git inspection commands and does
**not fetch, merge, check out files, install packages, load models, or update the
sync record**. Partial/promisor clones are rejected to prevent implicit object
fetches. Exit zero only means the
comparison succeeded. Its default ref is the locally cached `upstream/main`,
which may be stale; it does not discover releases. A clean overlap report does not
prove compatibility: upstream API changes can break callers in different files.

If ancestry fails, check the ref and sync record, deepen a shallow clone if needed,
and investigate a rewritten upstream history. An older upstream release that
predates the integrated baseline is not a forward update. Do not change the
baseline just to make the report pass.

## Merge on an isolated branch

The following POSIX-shell example keeps the working installation untouched:

```sh
git worktree add -b maintenance/upstream-vX.Y ../textgen-upstream-vX.Y HEAD
cd ../textgen-upstream-vX.Y
git merge --no-ff --no-commit refs/upstream-tags/vX.Y
```

Inspect the incoming commits and changed paths using the helper's exact baseline
and target SHAs. Resolve conflicts individually, then inspect the resulting diff
against the pre-merge fork HEAD. Do not resolve whole files with blanket
`--ours`/`--theirs`, replace the fork with an upstream archive, reset `main`, or
rebase/force-push published history. A conflict-free merge still needs semantic
review of the integration points below.

If abandoning the attempt, `git merge --abort` in this dedicated worktree restores
its pre-merge state. It does not undo later commits or changes outside the merge;
do not use it in a worktree that had unrelated edits.

## Integration map and checks

Most feature code is isolated under `modules/sentence_rewrite/` and
`modules/ui_sentence_rewrite.py`. Changes to existing upstream modules are the
main conflict risk:

| Area | Paths to inspect together | Required behavior / checks |
| --- | --- | --- |
| Notebook UI | `modules/ui_notebook.py`, `ui_default.py`, `ui_sentence_rewrite.py`, `callbacks.py` | Both layouts; native state gathering; build locks/errors/retry; seed/repeat; review/apply/undo; stale-edit protection; escaped HTML and autosave behavior. `test_rewrite_ui.py` plus browser smoke. |
| Generation boundary | `modules/text_generation.py`, `sentence_rewrite/engine.py` | Match actual backend BOS/special-token/extension tokenization. Preserve all references or fail; retain context-trim notices. Keep native sampling/hooks, per-request stopping, iterator cleanup, and visible Rewrite errors without changing ordinary generation. Engine/native tests plus real rewrites and ordinary Generate afterward. |
| Native backends | `modules/exllamav3.py`, `llama_cpp_server.py`, `tensorrt_llm.py` | Recheck against actual context after loading; no silent prompt slicing. Cancel blocked requests and release jobs/sockets. Preserve reasoning markers for sentence extraction. Backend adapter tests plus real models for each changed backend. `--ik` is a distinct binary to check. |
| Transformers/Gemma | `modules/transformers_loader.py`, `text_generation.py`, `requirements/rewrite-gemma4.txt` | Gemma Unified loading/EOS handling stays scoped to `gemma4_unified`. Recheck native template/thinking output, HF generation APIs, and model loading when changing Transformers. |
| Retrieval and cache | `modules/sentence_rewrite/{cleanup,sentences,embeddings,corpus,reranker}.py` | Trained token projection/roles, exact MaxSim, native-token eligibility, STS/NLI scores, source offsets, transactional cache and cancellation. Corresponding unit suites and real retrieval smoke. |
| Install/update | `one_click.py`, `requirements/{full,portable}/`, `requirements/rewrite*.txt`, startup/update scripts | All supported profiles include retrieval dependencies; nested includes resolve correctly. Fork updates retain origin/tracking branch; ZIP installs preserve application code. Gemma selection survives extension installs. Installer tests and clean profile installs. |
| Distribution | `.github/workflows/build-*.yml`, `scripts/verify_portable_rewrite.py`, Dockerfiles/Compose, `.dockerignore`, `Colab-TextGen-GPU.ipynb` | Build this fork, retain final `app/` contents, verify intended Torch backend and `pip check`, exclude private runtime data. Recheck CPU/CUDA/ROCm/MPS policy, source URLs and each changed package path. |
| Optional TensorRT isolation | `modules/tensorrt_{proxy,worker,protocol}.py`, `shared.py`, `ui_model_menu.py`, `scripts/setup_tensorrt.py`, TensorRT Dockerfile | Keep incompatible worker dependencies isolated, exact token IDs, bounded IPC, first-token cancellation and owned process-group cleanup. Controlled tests are not a real engine-build test. |

### Dependency and cache traps

- The updater currently recognizes the literal baseline pin `transformers==5.6.*`
  and the exact Gemma override `5.10.4`. When upstream changes that pin, revise
  `select_rewrite_requirements`, selection preservation, requirements and installer
  tests together. Do not let an extension downgrade the selected stack first.
- Native wheels couple Python, Torch, CUDA/ROCm, ABI and driver support. Check the
  **published wheels** for each platform; changing a version string does not prove
  that a matching wheel exists. Keep the compiled-backend checks aligned with the
  requirements. Portable means GGUF generation, not CPU-only retrieval.
- Sentence Transformers integration uses `MultiVectorEncoder`, `MultiVectorMask`
  and backbone query/document length and expansion settings. Check these APIs,
  role prompts, trained projections and punctuation retention against real token
  matrices after a library upgrade.
- Current platform choices and exact tested versions are in the feature guide.
  Windows AMD retrieval currently uses CPU; Apple MPS uses exact CPU matching;
  Intel macOS is unsupported by the current dependency stack. Reassess with
  evidence when upstream changes rather than silently restoring incompatible pins.
- TensorRT's worker stack is separate because its pinned Transformers dependencies
  conflict with retrieval. TensorRT runtime is experimental and was not verified
  with a real engine in the initial follow-up; do not promote it based on mocks.
- Cache identities currently include `SCHEMA_VERSION`, the splitter identity
  `period-spans-v1`, `CLEANUP_VERSION`, model/library/checkpoint identity and build
  settings. Change the appropriate identity or implement a migration when semantics
  change. Never accept an old matrix/provenance cache just because SQLite opens it.
- Browser tests write Notebook text/settings. Run them against a scratch
  `--user-data-dir` and port; use `--model-dir` to reference existing model weights.
  Do not run them against the user's active session or copy private corpora into a
  validation artifact or container context.
- `.dockerignore` permits only named shipped user-data templates. Add new upstream
  defaults deliberately; a broad directory exception would include private
  settings, corpora and models in images.

## Validation before declaring an update ready

Use a fresh environment with the candidate's actual dependency profile. Do not
assume temporary environments, `/workspace` paths, GPUs or cached models from the
original development session still exist. Reserve GPU resources for one lead
process; reviewers inspect code and may author tests, but do not run experiments.

For a full or portable candidate, select/install its documented hardware profile,
then use the following checks as appropriate (commands below are POSIX examples):

```sh
python -m pip check
python -m pip install pytest
python -m pytest -q tests/test_rewrite_*.py
```

The ordinary suite uses controlled fixtures and does not download models; its
explicit CUDA scoring check is opt-in. On a suitable GPU host run:

```sh
REWRITE_TEST_CUDA=1 python -m pytest -q tests/test_rewrite_embeddings.py
python scripts/verify_portable_rewrite.py --torch-backend cuda --cuda-version 12.4
python tests/manual/rewrite_retrieval.py --device cuda:0 --offline --output-dir /path/to/scratch/retrieval
```

The compiled CUDA version above is an example for the CUDA 12.4 portable profile;
use the value expected by the candidate, `--torch-backend rocm`, or
`--torch-backend cpu` as applicable. The import check needs no GPU or downloads.
The real retrieval check loads LateOn and both rerankers; omit `--offline` only
when downloading models is allowed. Use `--device cpu` for a CPU profile.

Run `tests/manual/rewrite_browser.py` against a scratch server with a real model
loaded, following the feature guide. Verify both Notebook layouts, build lock and
retry, seeds/repetition, Stop/retry, review conflicts, undo, exact token matching,
incomplete/custom-stop errors, and ordinary generation after Rewrite. Exercise
each changed loader and dependency profile; do not infer new binary/Transformers
compatibility from the old version's browser results. A UI/template/streaming
change merits a browser run even when the unit suite passes.

Check normal Notebook behavior, Chat/API smoke and applicable upstream tests when
the merge changes those routes. Check template selection, tokenization, loaded
adapters and sampling hooks relevant to the change. Run clean install/update
checks for affected full and portable profiles, including Gemma override handling;
build each changed distribution where a suitable runner is available. Update the
validation matrix honestly for unavailable platforms rather than claiming they
passed. Historical pass counts in the guide are evidence for their recorded
versions, not an acceptance threshold for the next update.

## Finish and leave a record

1. Obtain fresh independent source/integration review. If retrieval or writing
   behavior changed, also ask for expert-domain/workflow review for conspicuous
   omissions. Resolve findings and rerun the affected checks.
2. After the merge and relevant validation succeed, update `upstream-sync.json`:
   set `last_integrated_commit` to the **peeled upstream target SHA** printed by the
   helper, and `last_integrated_ref` to the selected release tag. Never put the fork
   merge commit here. Stage this record with the completed integration. During a
   `--no-commit` merge, HEAD still names the old fork commit; rerun the helper with
   the new baseline after completing the merge commit.
3. Update the feature guide's compatibility/validation matrix and release notes:
   upstream tag/SHA, important conflict resolutions, dependency/model changes,
   cache migration/rebuild implications, commands/results and unverified cases.
   Inspect both `git diff --check` and `git diff --cached --check`, plus
   `git diff --cached --stat` and the final file list for accidental runtime data.
4. Complete the merge commit and publish the maintenance branch or merge it into
   the fork according to the user's existing instructions. Do not force-push.
   Verify that the remote ref matches the intended local commit. GitHub tokens
   need repository Contents write and Workflows write when workflow files change;
   request only the missing permission if GitHub rejects that action.
5. Build portable artifacts from the tested **fork ref** when publishing a release.
   A source push does not create an updated portable ZIP automatically. Preserve
   release labels/provenance so users can distinguish it from upstream's artifacts.

Suggested next-agent request:

> Integrate upstream release vX.Y into this fork. Read AGENTS.md and
> docs/UPSTREAM-MAINTENANCE.md, compare the recorded upstream baseline with that
> release, preserve Notebook Rewrite and normal textgen behavior, validate the
> affected loaders/installers, obtain independent review, and update the sync
> record and compatibility notes. Report untested cases explicitly.

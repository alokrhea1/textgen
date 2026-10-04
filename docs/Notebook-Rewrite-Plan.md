# Notebook Rewrite: historical plan

This document records the initial design.
It is not the specification for the feature.
The initial upstream baseline was `c93f8871239550de2ccfe1e95d469aa82616f07e`.

For operation, refer to [Notebook sentence rewriting](Notebook-Rewrite.md).
For code changes and validation, refer to [Rewrite development](Rewrite-Development.md).
For upstream integration, refer to [Upstream maintenance](UPSTREAM-MAINTENANCE.md).

## Initial design

The initial feature was a manual Rewrite button in the sixth Notebook subtab.
A local TXT corpus supplied top-K references for the last completed sentence.
A period parser kept the target span and all other characters.
Users could supply a seed sentence and do the operation again.
The design kept usual generation the same.

The design selected trained contextual token embeddings and late interaction.
The initial encoders were GTE-ModernColBERT and AnswerAI ColBERT-small.
Corpus storage used a private SQLite cache with source locations.
The design made each cache available after a completed build only.
The search used sentence-count or generation-token eligibility.
The prompt used native templates, samplers, and token budgets.
Application and undo used session state guards.

## Changes after the plan

The feature includes these additions:

- Retrieval dependencies in supported installation profiles, with model loading before the first operation.
- LateOn as the initial encoder selection.
- Bidirectional STS and NLI reranking.
- Shared format cleanup and ingestion quality screening.
- Reference acceptance limits and source audits.
- Automatic sentence rewriting during Notebook generation.

The user and developer guides give the functions of those additions.
This plan does not show loader compatibility or test results.

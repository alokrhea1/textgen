# Corpus quality and reference acceptance

This developer guide gives information about corpus screening and reference acceptance.
For user instructions, refer to [Notebook Rewrite](Notebook-Rewrite.md).

## Corpus screening

Each corpus uses this sequence:

```text
UTF-8 source → formatting cleanup → period spans → quality screening
            → sentence windows → token embeddings → SQLite cache
```

The system does not change source files.

| Screening policy | Result |
| --- | --- |
| `balanced` (default) | Records flags. Does not use passages with damage flags. |
| `off` | Records flags. Keeps passages with damage flags. Index limits do not change. |

After a policy change, build the corpus again.
Use `off` if screening rejects correct text.

Quality version 2 uses these flags:

| Flag | Condition | Cleanup mode |
| --- | --- | --- |
| `short_reference` | Maximum four character runs and 24 letters. This flag does not reject text. | All |
| `replacement_character` | Unicode replacement character U+FFFD. | All |
| `control_character` | A control character other than whitespace, or an incorrect Unicode surrogate. | All |
| `punctuation_fragment` | All characters that you can see are punctuation. | All |
| `symbol_noise` | A high separator-symbol ratio. | `scanned_book` |
| `suspicious_boundary` | A short lowercase passage after a passage that starts with an uppercase letter. There are more conditions. | `scanned_book` |
| `dangling_function_boundary` | English `if`, `in`, `of`, or `to` before a lowercase continuation. There are more conditions. | `scanned_book` |
| `ocr_quote_digit` | One ASCII quotation digit or contraction digit `5` in a specified English pattern. | `scanned_book` |
| `leading_digit_paragraph` | A single initial digit before a paragraph break and prose. | `scanned_book` |

Refer to [quality.py](../modules/sentence_rewrite/quality.py) for the full conditions.

Boundary checks flag the two adjacent passages.
They do not use quotation, paragraph, or ellipsis boundaries.
No indexed window contains a rejected passage or puts text together across it.

Counts measure character runs, not words.
Short text, lowercase text, and different scripts can stay indexed.
English scan rules can miss damage or reject correct prose, such as numbered headings.
They do not repair text.

## Source records and cache

Each occurrence records its source path, initial character offsets, flags, and maximum 200 cleaned characters on each side.
`excluded_spans` contains rejected text and reasons.
The same accepted passages share one embedding and keep different source records.

The UI shows cleaned text.
Offsets identify decoded source characters and include a byte-order mark if the source contains one.
They do not identify bytes or cleaned-text positions.

Schema 2 adds audit fields and content-token counts.
Source hashes, model identity, settings, and cleanup/quality versions determine cache identity.
**Build corpus / retry** rebuilds caches with a previous schema.
If a build has an error, the previous completed cache stays available.
If screening rejects all completed spans, the build deletes its temporary database.
The UI keeps counts and samples for retry.

### Examine exclusions

A completed cache contains all rejected occurrences.

1. Replace the example path below with the completed cache path for the selected Notebook tab.
2. Run this command:

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

Keep the output, with corpus text and server paths, in a directory that is not in the repository.
For accepted text, join `occurrences.candidate` to `candidates.id`.
Examine `occurrences.quality_flags`, `context_before`, and `context_after` for each location.

## Reference acceptance

```text
Target → length check → exact late-interaction scan → STS/NLI reranking
       → score thresholds → maximum K references → rewrite prompt
```

| UI control | Default | Function |
| --- | --- | --- |
| **Top K references** | 5 | Maximum reference count. |
| **Minimum reference length relative to target** | 0.5 | The reference must contain half the target's embedding content-token count or more. Sentence mode only. Zero disables this check. |
| **Late-interaction candidates to rerank** | 200 | Candidate pool for STS/NLI reranking. |
| **Minimum semantic similarity** | 0.3 | Rejects lower STS scores. Zero disables this filter. |
| **Maximum contradiction score** | 0.8 | Rejects higher NLI contradiction scores. One disables this filter. |

Content-token count subtracts the empty document's marker and special-token count from the embedding token count.
The ratio sets no maximum.
Token mode uses the generation backend's token count and tolerance and does not use the ratio.

The exact scan scores all candidates that the selected length and exact-copy filters accept.
STS/NLI uses only the selected pool: UI range 5–500, module/script maximum 2,000.
A larger pool increases computation and can find satisfactory references below the previous cutoff.

The score filters operate only with **Rerank for meaning and nuance**.
Disabled reranking disables the two filters.
The maximum reference count is K.
If the system rejects all references, generation does not start.
It does not add rejected references to fill K.

STS is semantic textual similarity.
NLI is natural language inference.
Each model scores the two pair directions.

The semantic score is the mean STS value after sigmoid.
Entailment uses the smaller NLI value.
Contradiction uses the larger value.
NLI values use softmax.
The ranking score is:

```text
semantic + 0.25 × entailment − 0.25 × contradiction
```

Scores are not confidence percentages or a check for equivalent meaning.
The default STS/NLI models use English.
They can reject indirect references or give unsatisfactory results for different languages.
No automatic threshold calibration or literary-style ranking is available.

## Retrieval evaluation

[rewrite_quality_cases.json](../tests/fixtures/rewrite_quality_cases.json) contains diagnostic text and labels.
[rewrite_quality.py](../tests/manual/rewrite_quality.py) uses real retrieval models.
It records model/source identities, settings, timing, errors, and retention probes in `results.json`.
It does not measure generation or literary quality.
Each run must use an empty output directory.

1. Activate the textgen Python environment.
2. Select two new output directories that are not in the repository.
3. Run with quality controls disabled:

```sh
python tests/manual/rewrite_quality.py --device cuda:0 \
  --cleanup scanned_book --quality-policy off --min-length-ratio 0 \
  --min-semantic-score 0 --max-contradiction-score 1 \
  --output-dir /tmp/rewrite-quality-controls-off
```

4. Run with the default quality controls:

```sh
python tests/manual/rewrite_quality.py --device cuda:0 \
  --cleanup scanned_book \
  --compare /tmp/rewrite-quality-controls-off/results.json \
  --output-dir /tmp/rewrite-quality-balanced
```

Use `--device cpu` if CUDA is not available.
Add `--offline` only when all three models are cached.

The first configuration gives an approximation of previous acceptance behavior.
For a previous checkout, record its revision and manifest settings.
Code without `quality_policy` cannot use the requested policy.
Retention probes with settings that do not agree give `applicable: false` and `expectation_met: null`, neither pass nor failure.

Add `--corpus /absolute/path/to/corpus.txt` for external text.
Use this argument again for more paths.
Fixture labels do not measure external-corpus relevance.
Use `--rerank-pool 500` for a pool comparison with the same settings.
Keep passages and results in a directory that is not in the repository.

Examine exclusions, offsets, duplicate source records, and windows near rejected spans.
Make sure that correct text stays available and retrieval refusal prevents generation without text changes.
Refer to [Rewrite development](Rewrite-Development.md) for regression checks.
The lead agent does tests, as specified in [AGENTS.md](../AGENTS.md).

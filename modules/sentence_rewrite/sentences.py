"""Deterministic period-based spans, not linguistic sentence segmentation.

Question/exclamation marks alone do not end sentences. Ellipses do when
followed by whitespace or EOF. Known abbreviations (including at EOF) and
single-letter initials are conservatively incomplete; this deliberately
misses some genuine endings such as ``and so on etc.``. Dotted words are
kept intact. Streaming requires a character beyond the closing punctuation.
Offsets refer to the original Python string; replacement preserves all
characters outside the selected span.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    text: str


_CLOSERS = frozenset('\"\'”’)]}')
_OPENERS = '\"\'“‘([{'
_ABBREVIATIONS = frozenset({
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs',
    'etc', 'e.g', 'i.e', 'a.m', 'p.m', 'inc', 'ltd', 'no',
})
_MAX_ABBREVIATION = max(map(len, _ABBREVIATIONS))


def _spans(text: str, final: bool):
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    start = token_start = i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char.isspace():
            token_start = i + 1
            if start == i:
                start = i + 1
            i += 1
            continue
        if char != '.':
            i += 1
            continue
        dot_start = i
        while i < length and text[i] == '.':
            i += 1
        dot_end = i
        while i < length and text[i] in _CLOSERS:
            i += 1
        end = i
        if end < length and not text[end].isspace():
            continue
        if end == length and not final:
            continue
        # Slice only bounded tokens: long words cannot be abbreviations.
        if dot_end - dot_start == 1 and dot_start - token_start <= _MAX_ABBREVIATION + 4:
            word = text[token_start:dot_start].lstrip(_OPENERS).casefold()
            if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue
        if start < dot_start:
            yield SentenceSpan(start, end, text[start:end])
            start = end


def sentence_spans(text: str) -> list[SentenceSpan]:
    """Return completed spans; ignore an unfinished trailing fragment."""
    return list(_spans(text, final=True))


def iter_sentence_spans(text: str):
    """Yield completed spans without accumulating a whole document's spans."""
    return _spans(text, final=True)


def first_sentence(text: str, final: bool = False) -> SentenceSpan | None:
    """Return the first confirmed span, waiting at stream EOF by default."""
    return next(_spans(text, final=final), None)


def last_sentence(text: str) -> SentenceSpan:
    last = None
    for span in iter_sentence_spans(text):
        last = span
    if last is None:
        raise ValueError('No completed sentence found: a terminating period is required.')
    return last


def replace_sentence(text: str, span: SentenceSpan, replacement: str) -> str:
    """Replace a current parser span with exactly one completed sentence."""
    if not isinstance(span, SentenceSpan) or span not in sentence_spans(text):
        raise ValueError('Sentence span is invalid or does not match the current text.')
    if not isinstance(replacement, str):
        raise TypeError('replacement must be a string')
    replacement = replacement.strip()
    completed = sentence_spans(replacement)
    if len(completed) != 1 or completed[0].start != 0 or completed[0].end != len(replacement):
        raise ValueError('Replacement must contain exactly one completed period-terminated sentence and no trailing fragment.')
    return text[:span.start] + replacement + text[span.end:]

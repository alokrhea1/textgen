"""Conservative, inspectable quality signals for corpus ingestion.

These signals do not prove grammaticality or OCR accuracy.  They never change
text, and do not alter the period-based Notebook selection contract.  In
particular, a short reference is legitimate: its usefulness depends on the
query.  The optional scanned-book boundary check quarantines suspicious spans
rather than guessing which punctuation or words the scan should contain.
"""

import re
import unicodedata
from dataclasses import dataclass


QUALITY_VERSION = 2
QUALITY_POLICIES = ('balanced', 'off')
QUALITY_REASONS = {
    'short_reference': 'A short reference; retained for suitably short queries.',
    'replacement_character': 'Contains a Unicode replacement character, indicating possible decoding or OCR damage.',
    'control_character': 'Contains a non-whitespace control character or invalid Unicode surrogate.',
    'punctuation_fragment': 'Contains only punctuation, without letters, numbers, mathematical symbols, currency, or emoji.',
    'symbol_noise': 'Scanned text contains a high concentration of repeated OCR-like separator symbols.',
    'suspicious_boundary': 'Scanned text has a short lowercase fragment after an uppercase-led passage in the same paragraph; the period boundary needs inspection.',
    'dangling_function_boundary': 'Scanned text ends a passage with English if/in/of/to before a lowercase continuation; inspect this potentially dangling boundary, which can also occur in deliberate prose.',
    'ocr_quote_digit': 'Scanned text contains a digit in a narrowly recognized English contraction or dialogue-quotation pattern; inspect the source.',
    'leading_digit_paragraph': 'Scanned text begins with a lone digit separated from prose by a paragraph break; inspect possible quotation debris or an intentional numbered heading.',
}
_EXCLUDED = frozenset(QUALITY_REASONS) - {'short_reference'}
_WORD_JOINERS = frozenset("'’ʼ-‐‑")
_QUOTE_MARKS = frozenset('"\'“”‘’«»‹›„‟‚‛')
_LEADING_DECORATION = frozenset('([{—–-')
_TRAILING_DECORATION = frozenset(')]}')
_DANGLING_END = re.compile(r'(?<![\w’\'-])(if|in|of|to)\.$', re.I)
_LEADING_DIGIT_PARAGRAPH = re.compile(
    r'^\s*[0-9][ \t]*(?:(?:\r?\n[ \t]*){2,}|[\u2029\f][ \t]*)(?=[^\W\d_])'
)
# These are deliberately narrow English OCR patterns, not a grammar or
# spelling classifier.  They only apply when scanned-book cleanup is selected.
# Requiring a sentence-leading pronoun and a following word avoids interpreting
# valid quantities/durations such as "Give it 5 s to settle." as contractions.
_CONTRACTION_OCR = re.compile(
    r'''^\s*["'“‘]?(?:it|he|she|that|there|here|what|who)\s+5\s+s(?=\s+[^\W\d_])''',
    re.I,
)
_QUOTE_OCR = re.compile(
    r'''(?<!\w)["'“‘][^.!?]{0,160}[!?.,]\s+[0-9]\s+(?:she|he|i|we|they|you)\s+'''
    r'(?:said|asked|replied|answered|added|whispered|cried|called|murmured|'
    r'demanded|insisted|shouted|muttered)\b',
    re.I,
)


@dataclass(frozen=True)
class CandidateQuality:
    # These are Unicode character runs, not linguistic words.  A CJK sentence
    # without spaces may be one run; use model tokens for length eligibility.
    word_count: int
    letter_count: int
    flags: tuple


def _counts(text):
    """Count Unicode letter/mark/number runs, including internal word joiners."""
    categories = [unicodedata.category(char) for char in text]
    letter_count = sum(category.startswith('L') for category in categories)
    units = 0
    inside = False
    for index, (char, category) in enumerate(zip(text, categories)):
        if category[0] in 'LN':
            if not inside:
                units += 1
            inside = True
        elif category.startswith('M') and inside:
            continue
        elif (char in _WORD_JOINERS and inside and index + 1 < len(text)
              and categories[index + 1][0] in 'LN'):
            continue
        else:
            inside = False
    return units, letter_count


def _first_letter(text):
    text = text.lstrip()
    while text and text[0] in _LEADING_DECORATION:
        text = text[1:].lstrip()
    # A quotation can intentionally begin in the middle of a sentence.
    if not text or text[0] in _QUOTE_MARKS:
        return None
    return text[0] if text[0].isalpha() else None


def _boundary_context(text, left, right):
    """Return the left text only for adjacent, unquoted, non-ellipsis spans."""
    if left is None or right is None or left.end > right.start:
        return None
    gap = text[left.end:right.start]
    if (not gap.isspace() or '\n\n' in gap or '\r\n\r\n' in gap
            or '\u2029' in gap or '\f' in gap):
        return None
    left_text = left.text.rstrip()
    # Closing quotation marks and ellipses are legitimate discontinuities.
    if not left_text or left_text[-1] in _QUOTE_MARKS:
        return None
    left_text = left_text.rstrip(''.join(_TRAILING_DECORATION))
    if not left_text.endswith('.') or left_text.endswith('..'):
        return None
    return left_text


def _dubious_boundary(text, left, right):
    """Look for composite OCR evidence, never lowercase or brevity alone."""
    if _boundary_context(text, left, right) is None:
        return False
    left_first = _first_letter(left.text)
    right_first = _first_letter(right.text)
    if not (left_first and left_first.isupper() and right_first and right_first.islower()):
        return False
    left_words, left_letters = _counts(left.text)
    right_words, right_letters = _counts(right.text)
    # Avoid treating lowercase writing, uncased scripts, or ordinary short
    # dialogue as OCR errors.  This remains a reviewable heuristic, restricted
    # to an explicitly selected scanned-book cleanup mode.
    return (left_words >= 4 and left_letters >= 12
            and 1 <= right_words <= 4 and 1 <= right_letters <= 24)


def _dangling_function_boundary(text, left, right):
    """Recognize a narrowly scoped English OCR split, not arbitrary grammar."""
    left_text = _boundary_context(text, left, right)
    if left_text is None or not _DANGLING_END.search(left_text):
        return False
    # Keep deliberate all-lowercase writing outside the long-passage heuristic.
    # A standalone "in." is still inspected, as is an anomalous capitalized
    # "If." embedded at the end of a longer OCR passage.
    if _counts(left_text)[0] > 1 and not any(char.isupper() for char in left_text):
        return False
    right_first = _first_letter(right.text)
    # Two-word minimum preserves short response sequences such as "In. yes."
    # Quotation/paragraph boundaries were excluded by _boundary_context.
    return bool(right_first and right_first.islower() and _counts(right.text)[0] >= 2)


def assess_span(text, span, previous=None, following=None, cleanup='conservative'):
    """Assess an original cleaned-text span using its immediate neighbors.

    The caller must pass neighbors before excluding anything.  Contextual
    boundary flags belong to this occurrence, not to every occurrence of its
    text.  Both sides of a suspicious boundary are flagged so no larger window
    can silently retain the damaged pair.
    """
    value = span.text
    words, letters = _counts(value)
    flags = []
    if words <= 4 and letters <= 24:
        flags.append('short_reference')
    if '\ufffd' in value:
        flags.append('replacement_character')
    if any((unicodedata.category(char) == 'Cc' and not char.isspace())
           or unicodedata.category(char) == 'Cs' for char in value):
        flags.append('control_character')
    visible_characters = [char for char in value if not char.isspace()]
    if visible_characters and all(unicodedata.category(char).startswith('P') for char in visible_characters):
        flags.append('punctuation_fragment')
    if cleanup == 'scanned_book':
        # Do not penalize punctuation, mathematical notation, currency, emoji,
        # or unfamiliar scripts.  This narrowly catches scan separator debris.
        separators = sum(char in '|¦_' for char in value)
        visible = sum(not char.isspace() for char in value)
        if (separators >= 6 and separators / max(visible, 1) >= 0.4
                and re.search(r'[|¦_]{3,}', value)):
            flags.append('symbol_noise')
        if (_dubious_boundary(text, previous, span)
                or _dubious_boundary(text, span, following)):
            flags.append('suspicious_boundary')
        if (_dangling_function_boundary(text, previous, span)
                or _dangling_function_boundary(text, span, following)):
            flags.append('dangling_function_boundary')
        if _CONTRACTION_OCR.search(value) or _QUOTE_OCR.search(value):
            flags.append('ocr_quote_digit')
        digit_paragraph = _LEADING_DIGIT_PARAGRAPH.match(value)
        if digit_paragraph:
            prose_words, prose_letters = _counts(value[digit_paragraph.end():])
            # Require prose after the detached digit, not another number,
            # a one-word label, or a mathematical/currency expression.
            if prose_words >= 2 or prose_letters >= 12:
                flags.append('leading_digit_paragraph')
    return CandidateQuality(words, letters, tuple(flags))


def quality_exclusions(quality, policy='balanced'):
    """Return reason codes to quarantine, retaining legitimate short spans."""
    if policy not in QUALITY_POLICIES:
        raise ValueError('Corpus quality policy must be balanced or off.')
    if policy == 'off':
        return ()
    return tuple(reason for reason in quality.flags if reason in _EXCLUDED)

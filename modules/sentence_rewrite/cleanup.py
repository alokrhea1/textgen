"""Explicit corpus cleanup with compact original Unicode offset tracking."""
import re
from io import StringIO
from array import array
from dataclasses import dataclass

CLEANUP_VERSION = 2
MODES = ('none', 'conservative', 'scanned_book')


@dataclass
class CleanedText:
    text: str
    offsets: array
    report: dict

    def source_span(self, start, end):
        if not 0 <= start < end <= len(self.text):
            raise ValueError('Invalid cleaned-text span.')
        return int(self.offsets[start]), int(self.offsets[end - 1]) + 1


def clean_text(original, mode='conservative', join_hyphenated_lines=False):
    """Return cleaned text and a uint32 map into the untouched decoded source.

    Offsets map retained/replacement characters to their original positions;
    spans include removed interior whitespace, furniture, and hyphens.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown corpus cleanup mode: {mode}')
    if len(original) >= 2 ** 32:
        raise ValueError('Source exceeds compact Unicode offset mapping capacity.')
    text = original
    offsets = array('I', range(len(original)))
    counts = {}
    snippets = []

    def replace(pattern, replacement, category, flags=0):
        nonlocal text, offsets
        pieces = StringIO()
        mapped = array('I')
        position = 0
        count = 0
        for match in re.finditer(pattern, text, flags):
            value = replacement(match) if callable(replacement) else replacement
            if value == match.group():
                continue
            pieces.write(text[position:match.start()])
            mapped.extend(offsets[position:match.start()])
            pieces.write(value)
            # Replacements only synthesize whitespace. Semantic characters
            # remain outside deletion matches with their original mapping.
            if value:
                mapped.extend([offsets[match.start()]] * len(value))
            if len(snippets) < 10 and not any(item['kind'] == category for item in snippets):
                before_start = max(0, match.start() - 40)
                after_end = min(len(text), match.end() + 40)
                left = text[before_start:match.start()]
                right = text[match.end():after_end]
                snippets.append({'kind': category,
                                 'before': (left + match.group()[:80] + right)[:160],
                                 'after': (left + value + right)[:160]})
            position = match.end()
            count += 1
        if count:
            pieces.write(text[position:])
            mapped.extend(offsets[position:])
            text = pieces.getvalue()
            offsets = mapped
            counts[category] = counts.get(category, 0) + count

    if mode != 'none':
        replace(r'\A\ufeff', '', 'byte_order_mark')
        replace(r'\r\n?', '\n', 'line_endings')
        replace(r'[\u2029\f]', '\n\n', 'paragraph_separators')
        replace('\u2028', '\n', 'line_separators')
        if mode == 'scanned_book':
            replace(r'^[^\S\n]*(?:\d+|(?:page|chapter)[^\S\n]+(?:\d+|[ivxlcdm]+))[^\S\n]*(?:\n|$)', '', 'furniture_lines', re.I | re.M)
        # Delete only the join punctuation and separating whitespace. The
        # letters on either side stay outside the match with exact mappings.
        replace(r'(?<=[^\W\d_])\u00ad[^\S\n]*\n[^\S\n]*(?=[^\W\d_])', '', 'soft_hyphen_wraps')
        replace('\u00ad', '', 'soft_hyphens')
        if mode == 'scanned_book':
            replace(r'(?<=[^\W\d_])¬[^\S\n]*\n[^\S\n]*(?=[^\W\d_])', '', 'ocr_wraps')
        if join_hyphenated_lines:
            replace(r'(?<=[^\W\d_])-[^\S\n]*\n[^\S\n]*(?=[^\W\d_])', '', 'hyphenated_wraps')
        replace(r'[^\S\n]+', ' ', 'horizontal_whitespace')
        # Preserve blank-line paragraph boundaries, even when blank lines
        # contain horizontal spaces. Single line wraps become a space.
        replace(r' *\n(?: *\n)+ *', '\n\n', 'paragraph_whitespace')
        replace(r'(?<!\n) *\n *(?!\n)', ' ', 'line_wraps')
    return CleanedText(text, offsets, {'mode': mode, 'counts': counts,
                                     'changes': sum(counts.values()), 'snippets': snippets,
                                     'warning': 'Scanned-book cleanup can remove meaningful standalone numbers or chapter labels.' if mode == 'scanned_book' else ''})

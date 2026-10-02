#!/usr/bin/env python3
"""Curate three inspected Gutenberg editions; never modify the source files."""
import argparse
import hashlib
import json
from pathlib import Path
import re


BOOKS = (
    (1906, "Samuel Butler", "1835–1902", "samuel_butler", "Erewhon; Or, Over the Range", "EREWHON; OR, OVER THE RANGE", "CHAPTER I. WASTE LANDS", 29),
    (2084, "Samuel Butler", "1835–1902", "samuel_butler", "The Way of All Flesh", "THE WAY OF ALL FLESH", "CHAPTER I", 86),
    (432, "Henry James", "1843–1916", "henry_james", "The Ambassadors", "THE AMBASSADORS", "Book First", 12),
)
JAMES_NOTE = """ [Note: In the 1909 New York Edition the following two chapters were
 placed in the reverse of the order appearing below. Since 1950, most
 scholars have agreed, because of the internal evidence of the two
 chapters, that an editorial error caused them to be printed in reverse
 order. This Etext, like other editions of the past four decades,
 corrects the apparent error.—Richard D. Hathaway, preparer of this
 electronic text]"""


def unique_index(lines, marker):
    matches = [i for i, line in enumerate(lines) if line == marker]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact marker {marker!r}; found {len(matches)}")
    return matches[0]


def curate(raw, number, marker_title, first_heading, expected_headings):
    lines = raw.decode("utf-8-sig").splitlines()
    start = unique_index(lines, f"*** START OF THE PROJECT GUTENBERG EBOOK {marker_title} ***")
    end = unique_index(lines, f"*** END OF THE PROJECT GUTENBERG EBOOK {marker_title} ***")
    if end <= start:
        raise ValueError("Gutenberg markers are out of order")
    body = lines[start + 1:end]
    first = unique_index(body, first_heading)
    body = body[first:]
    removals = ["Gutenberg header/footer", "front matter before first narrative chapter (titles, contents, epigraphs, prefaces)"]
    if number == 1906:
        body = body[:unique_index(body, "Footnotes")]
        removals += ["entire final Footnotes section, including editorial DP note", "inline footnote reference numbers [2]–[6]"]
    if number == 2084:
        fourth = unique_index(body, "CHAPTER IV")
        sixth = unique_index(body, "CHAPTER VI")
        if sixth <= fourth:
            raise ValueError("Excluded chapters are out of order")
        body = body[:fourth] + body[sixth:]
        removals += ["chapters IV and V reconstructed by R. A. Streatfeild", "Streatfeild preface"]
        expected_headings -= 2
    if number == 432:
        joined = "\n".join(body)
        if joined.count(JAMES_NOTE) != 1:
            raise ValueError("Expected the exact inspected Hathaway editorial note")
        body = joined.replace(JAMES_NOTE, "").splitlines()
        removals += ["James authorial preface (narrative-only selection)", "Richard D. Hathaway note at Book Eleventh", "Volume II and Roman-numeral subsection headings"]
    output = []
    heading_count = 0
    i = 0
    while i < len(body):
        line = body[i]
        if ((number != 432 and re.fullmatch(r"CHAPTER [IVXLCDM]+(?:\. .*)?", line))
                or (number == 432 and re.fullmatch(r"Book (First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth)", line))):
            heading_count += 1
            i += 1
            # Erewhon has two chapter titles wrapped onto a second line.
            if number == 1906:
                while i < len(body) and body[i].strip():
                    if body[i] not in ("OF ANIMALS", "RIGHTS OF VEGETABLES"):
                        raise ValueError(f"Unexpected chapter-title continuation: {body[i]!r}")
                    i += 1
            continue
        if number == 432 and (re.fullmatch(r" [IVX]+", line) or line == " Volume II"):
            i += 1
            continue
        # Remove conversion labels only; retain any surrounding original prose.
        line = line.replace("[Illustration]", "")
        if number == 1906:
            line = re.sub(r"\[[2-6]\]", "", line)
        output.append(line)
        i += 1
    if heading_count != expected_headings:
        raise ValueError(f"Expected {expected_headings} chapter/book headings, found {heading_count}")
    text = "\n".join(output).strip() + "\n"
    # Normalize blank-line spacing without reflowing or rewriting prose.
    text = re.sub(r"\n(?:[ \t]*\n){2,}", "\n\n", text)
    removals += ["standalone chapter/book headings", "[Illustration] conversion labels"]
    return text, removals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir or source_dir.is_relative_to(output_dir) or output_dir.is_relative_to(source_dir):
        parser.error("Source and output directories must be separate, non-nested directories")
    prepared = []
    manifest = {"schema_version": 1, "license": "Public domain in the USA; Project Gutenberg license/trademark terms apply to Gutenberg distributions: https://www.gutenberg.org/license", "normalization": "UTF-8, LF line endings, at most one blank line; original prose wording and line wrapping preserved; no token/length filtering", "books": []}
    # Validate every edition before writing any output.
    for number, author, dates, directory, title, marker_title, first, count in BOOKS:
        source = source_dir / f"pg{number}.txt"
        raw = source.read_bytes()
        text, removed = curate(raw, number, marker_title, first, count)
        relative = Path(directory) / f"pg{number}.txt"
        prepared.append((output_dir / relative, text))
        manifest["books"].append({"author": author, "author_dates": dates, "title": title, "gutenberg_id": number, "source_file": str(source), "source_url": f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt", "catalog_url": f"https://www.gutenberg.org/ebooks/{number}", "source_sha256": hashlib.sha256(raw).hexdigest(), "output_file": relative.as_posix(), "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "removed": removed})
    for destination, text in prepared:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Slice narrative from the inspected user-supplied Lolita OCR, preserving bytes."""
import argparse
import hashlib
import json
from pathlib import Path


START = "lolita, light of my liie, nre or my loms. My sm, my soul*"
END = "VLADIMIR NABOKO V"
AFTERWORD = "ON A BOOK ENTITLED LOLITA"


def marker_index(lines, marker):
    matches = [i for i, line in enumerate(lines) if line.strip() == marker]
    if len(matches) != 1:
        raise ValueError(f"Expected one inspected marker {marker!r}; found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    destination = output_dir / "vladimir_nabokov" / "lolita.txt"
    manifest_path = output_dir / "nabokov_manifest.json"
    if source in (destination.resolve(), manifest_path.resolve()):
        parser.error("Output must not overwrite the source")
    raw = source.read_bytes()
    # Retain each original line ending and all text bytes within the boundaries.
    raw_lines = raw.splitlines(keepends=True)
    lines = [line.decode("utf-8-sig") for line in raw_lines]
    start = marker_index(lines, START)
    end = marker_index(lines, END)
    afterword = marker_index(lines, AFTERWORD)
    if not start < end < afterword or any(line.strip() for line in lines[end + 1:afterword]):
        raise ValueError("Inspected narrative/afterword boundaries are out of order or unexpected")
    prepared = b"".join(raw_lines[start:end])
    manifest = {
        "schema_version": 1,
        "author": "Vladimir Nabokov",
        "title": "Lolita",
        "provenance": "User-provided local OCR text; not downloaded by this script",
        "rights": "No public-domain or redistribution permission claim is made",
        "source_file": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output_file": "vladimir_nabokov/lolita.txt",
        "output_sha256": hashlib.sha256(prepared).hexdigest(),
        "raw_line_boundaries": {
            "numbering": "1-based; narrative start inclusive, afterword boundary exclusive",
            "narrative_start": start + 1,
            "afterword_boundary": end + 1,
            "afterword_title": afterword + 1,
            "source_line_count": len(raw_lines),
        },
        "validated_markers": {"narrative_start": START, "afterword_boundary": END, "afterword_title": AFTERWORD},
        "removed": ["All material before the first narrative line", "Author afterword and all subsequent material"],
        "preserved": "Every byte between boundaries, including page numbers, wrapping, original line endings, OCR artifacts, and all narrative parts; ingestion performs optional scanned-book cleanup",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(prepared)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

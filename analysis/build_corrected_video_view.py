#!/usr/bin/env python
"""Build a timestamp-corrected hard-link view of MP4 files.

The original videos are never renamed or modified. The script finds a 14-digit
YYYYMMDDhhmmss timestamp in each filename, transforms it using one fixed anchor
offset (or an optional two-anchor linear drift correction), and creates a new
filename as a hard link on the same Windows volume.

Run without --apply first. A CSV manifest is always written for review.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable


STAMP_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")
STAMP_FORMAT = "%Y%m%d%H%M%S"
ANCHOR_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class MappingRow:
    source_path: str
    original_name: str
    wrong_datetime: str
    corrected_datetime: str
    corrected_date: str
    corrected_name: str
    corrected_path: str
    source_size_bytes: int
    correction_mode: str
    fixed_offset_seconds: str
    clock_scale: str
    status: str
    error: str


def parse_anchor(value: str) -> datetime:
    try:
        return datetime.strptime(value, ANCHOR_FORMAT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected timestamp like 2026-03-02 14:52:49, got: {value}"
        ) from exc


def build_corrector(args: argparse.Namespace) -> tuple[Callable[[datetime], datetime], str, str, str]:
    wrong1 = args.wrong_anchor
    correct1 = args.correct_anchor

    if (args.wrong_anchor_2 is None) != (args.correct_anchor_2 is None):
        raise ValueError("Provide both --wrong-anchor-2 and --correct-anchor-2, or neither.")

    if args.wrong_anchor_2 is None:
        offset = correct1 - wrong1

        def correct(value: datetime) -> datetime:
            return value + offset

        return correct, "fixed_offset", str(offset.total_seconds()), "1.0"

    wrong2 = args.wrong_anchor_2
    correct2 = args.correct_anchor_2
    wrong_span = (wrong2 - wrong1).total_seconds()
    correct_span = (correct2 - correct1).total_seconds()
    if wrong_span == 0:
        raise ValueError("The two wrong-clock anchors cannot have the same time.")
    scale = correct_span / wrong_span
    if scale <= 0:
        raise ValueError("Two-anchor mapping is not chronological; check the anchors.")

    def correct(value: datetime) -> datetime:
        elapsed = (value - wrong1).total_seconds()
        return correct1 + timedelta(seconds=elapsed * scale)

    offset = correct1 - wrong1
    return correct, "two_anchor_linear", str(offset.total_seconds()), f"{scale:.12f}"


def corrected_destination(
    source: Path,
    source_root: Path,
    view_root: Path,
    corrected_time: datetime,
    corrected_name: str,
    layout: str,
) -> Path:
    if layout == "by-date":
        return view_root / corrected_time.strftime("%Y-%m-%d") / corrected_name
    relative_parent = source.relative_to(source_root).parent
    return view_root / relative_parent / corrected_name


def write_manifest(path: Path, rows: list[MappingRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MappingRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--view-root", required=True, type=Path)
    parser.add_argument("--wrong-anchor", required=True, type=parse_anchor)
    parser.add_argument("--correct-anchor", required=True, type=parse_anchor)
    parser.add_argument("--wrong-anchor-2", type=parse_anchor)
    parser.add_argument("--correct-anchor-2", type=parse_anchor)
    parser.add_argument("--layout", choices=("preserve", "by-date"), default="preserve")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true", help="Create hard links after validation.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    view_root = args.view_root.resolve()
    manifest = (args.manifest or (view_root / "corrected_timestamp_manifest.csv")).resolve()

    if not source_root.is_dir():
        print(f"ERROR: source root does not exist: {source_root}", file=sys.stderr)
        return 2
    if source_root == view_root or source_root in view_root.parents:
        print("ERROR: view root must not be inside source root.", file=sys.stderr)
        return 2
    if os.name == "nt" and source_root.drive.lower() != view_root.drive.lower():
        print("ERROR: hard-link view must be on the same drive as the source videos.", file=sys.stderr)
        return 2

    try:
        correct_time, mode, offset_seconds, clock_scale = build_corrector(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    videos = sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        print(f"ERROR: no MP4 files found below {source_root}", file=sys.stderr)
        return 2

    rows: list[MappingRow] = []
    planned_destinations: dict[str, Path] = {}
    corrected_times: list[datetime] = []
    errors = 0

    for source in videos:
        match = STAMP_RE.search(source.name)
        if not match:
            rows.append(MappingRow(
                str(source), source.name, "", "", "", "", "",
                source.stat().st_size, mode, offset_seconds, clock_scale,
                "error", "No 14-digit YYYYMMDDhhmmss timestamp in filename",
            ))
            errors += 1
            continue

        stamp = match.group(1)
        try:
            wrong_time = datetime.strptime(stamp, STAMP_FORMAT)
        except ValueError as exc:
            rows.append(MappingRow(
                str(source), source.name, stamp, "", "", "", "",
                source.stat().st_size, mode, offset_seconds, clock_scale,
                "error", f"Invalid timestamp: {exc}",
            ))
            errors += 1
            continue

        corrected = correct_time(wrong_time).replace(microsecond=0)
        corrected_stamp = corrected.strftime(STAMP_FORMAT)
        corrected_name = source.name[:match.start(1)] + corrected_stamp + source.name[match.end(1):]
        destination = corrected_destination(
            source, source_root, view_root, corrected, corrected_name, args.layout
        )
        destination_key = os.path.normcase(str(destination))

        if destination_key in planned_destinations and planned_destinations[destination_key] != source:
            rows.append(MappingRow(
                str(source), source.name, wrong_time.strftime(ANCHOR_FORMAT),
                corrected.strftime(ANCHOR_FORMAT), corrected.strftime("%Y-%m-%d"),
                corrected_name, str(destination), source.stat().st_size, mode,
                offset_seconds, clock_scale, "error",
                f"Destination collision with {planned_destinations[destination_key]}",
            ))
            errors += 1
            continue
        planned_destinations[destination_key] = source
        corrected_times.append(corrected)

        status = "preview"
        error = ""
        if args.apply:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if os.path.samefile(source, destination):
                        status = "existing_same_hardlink"
                    else:
                        raise FileExistsError(f"Destination already exists: {destination}")
                else:
                    os.link(source, destination)
                    status = "hardlink_created"
            except Exception as exc:  # keep manifest even if one item fails
                status = "error"
                error = str(exc)
                errors += 1

        rows.append(MappingRow(
            str(source), source.name, wrong_time.strftime(ANCHOR_FORMAT),
            corrected.strftime(ANCHOR_FORMAT), corrected.strftime("%Y-%m-%d"),
            corrected_name, str(destination), source.stat().st_size, mode,
            offset_seconds, clock_scale, status, error,
        ))

    write_manifest(manifest, rows)

    print("=" * 72)
    print("TIMESTAMP CORRECTION PREVIEW" if not args.apply else "TIMESTAMP CORRECTION APPLIED")
    print(f"Source root: {source_root}")
    print(f"View root:   {view_root}")
    print(f"Videos:      {len(videos)}")
    print(f"Mode:        {mode}")
    print(f"Manifest:    {manifest}")
    if corrected_times:
        print(f"Corrected first time: {min(corrected_times).strftime(ANCHOR_FORMAT)}")
        print(f"Corrected last time:  {max(corrected_times).strftime(ANCHOR_FORMAT)}")
    print(f"Errors:      {errors}")
    print("=" * 72)
    if not args.apply:
        print("PREVIEW ONLY: no video links were created.")
        print("Verify the first and last corrected times, then rerun with --apply.")
    else:
        print("Original videos were not renamed, copied, or modified.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

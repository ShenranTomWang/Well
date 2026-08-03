#!/usr/bin/env python3
"""Apply CancerMythNFP annotations to JSONL files recursively, matched by ID."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if "id" not in row:
                raise ValueError(f"{path}:{line_number}: missing 'id' field")
            rows.append(row)
    return rows


def index_annotations(path: Path) -> dict[str, dict]:
    annotations: dict[str, dict] = {}
    for row in load_jsonl(path):
        row_id = str(row["id"])
        if row_id in annotations:
            raise ValueError(f"Duplicate id {row_id!r} in annotated file {path}")
        values = {}
        if "presuppositions" in row and not isinstance(row["presuppositions"], list):
            raise ValueError(f"Annotated datapoint {row_id!r}: 'presuppositions' must be a list")
        if "presuppositions" in row:
            values["presuppositions"] = row["presuppositions"]
        if "doctor_suggestion" in row:
            suggestions = row["doctor_suggestion"]
            if not isinstance(suggestions, (bool, list)) or (
                isinstance(suggestions, list)
                and not all(isinstance(value, bool) for value in suggestions)
            ):
                raise ValueError(
                    f"Annotated datapoint {row_id!r}: 'doctor_suggestion' must be a boolean "
                    "or a list of booleans"
                )
            if (
                isinstance(suggestions, list)
                and "presuppositions" in row
                and len(suggestions) != len(row["presuppositions"])
            ):
                raise ValueError(
                    f"Annotated datapoint {row_id!r}: 'doctor_suggestion' and "
                    "'presuppositions' must have the same length"
                )
        if "doctor_suggestion" in row:
            values["doctor_suggestion"] = row["doctor_suggestion"]
        if not values:
            raise ValueError(
                f"Annotated datapoint {row_id!r} has neither 'presuppositions' nor "
                "'doctor_suggestion'"
            )
        annotations[row_id] = values
    if not annotations:
        raise ValueError(f"Annotated file is empty: {path}")
    return annotations


def find_jsonl_files(directory: Path, annotated_file: Path) -> list[Path]:
    files: list[Path] = []
    for root, directory_names, file_names in os.walk(directory):
        directory_names.sort()
        for file_name in sorted(file_names):
            if file_name.lower().endswith(".jsonl"):
                path = (Path(root) / file_name).resolve()
                if path != annotated_file:
                    files.append(path)
    return files


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively update JSONL annotation fields using annotations matched by id."
    )
    parser.add_argument("--directory", required=True, type=Path, help="Directory containing target JSONL files")
    parser.add_argument("--annotated_file", required=True, type=Path, help="Annotated JSONL source")
    parser.add_argument("--dry_run", action="store_true", help="Validate and report without modifying files")
    args = parser.parse_args()

    directory = args.directory.expanduser().resolve()
    annotated_file = args.annotated_file.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"directory does not exist: {directory}")
    if not annotated_file.is_file():
        parser.error(f"annotated file does not exist: {annotated_file}")

    annotations = index_annotations(annotated_file)
    files = find_jsonl_files(directory, annotated_file)
    if not files:
        parser.error(f"no target .jsonl files found under {directory}")

    loaded: dict[Path, list[dict]] = {}
    failed: dict[Path, str] = {}
    annotation_ids = set(annotations)
    for path in files:
        try:
            rows = load_jsonl(path)
        except (OSError, ValueError) as exc:
            failed[path] = str(exc)
            continue
        file_ids: set[str] = set()
        duplicate_id = None
        for row in rows:
            row_id = str(row["id"])
            if row_id in file_ids:
                duplicate_id = row_id
                break
            file_ids.add(row_id)
        if duplicate_id is not None:
            failed[path] = f"duplicate id {duplicate_id!r} within file"
            continue

        missing_annotations = sorted(file_ids - annotation_ids)
        extra_annotations = sorted(annotation_ids - file_ids)
        if missing_annotations or extra_annotations:
            details = []
            if missing_annotations:
                details.append(
                    f"missing annotations for {len(missing_annotations)} ids: {missing_annotations[:10]}"
                )
            if extra_annotations:
                details.append(
                    f"missing {len(extra_annotations)} annotated datapoints: {extra_annotations[:10]}"
                )
            failed[path] = "ID sets do not match; " + "; ".join(details)
            continue
        loaded[path] = rows

    for rows in loaded.values():
        for row in rows:
            row.update(annotations[str(row["id"])])

    action = "Would update" if args.dry_run else "Updated"
    if not args.dry_run:
        for path, rows in loaded.items():
            write_jsonl_atomic(path, rows)
    total_rows = sum(len(rows) for rows in loaded.values())
    print(f"{action} {total_rows} datapoints across {len(loaded)} JSONL file(s).")
    if failed:
        print(f"Skipped {len(failed)} file(s):")
        for path, reason in failed.items():
            print(f"  - {path}: {reason}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path


FILE_SUFFIX = "response_level_score_evaluated.jsonl"


def sample_directory(directory: Path, sample_size: int, rng: random.Random) -> list[str]:
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist or is not a directory: {directory}")

    files = sorted(
        path
        for path in directory.rglob(f"*{FILE_SUFFIX}")
        if path.is_file()
    )
    if not files:
        raise ValueError(
            f"No files ending with {FILE_SUFFIX!r} found under {directory}"
        )

    sample: list[str] = []
    record_count = 0
    for path in files:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                try:
                    json.loads(stripped_line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error

                record_count += 1
                if len(sample) < sample_size:
                    sample.append(stripped_line)
                else:
                    replacement_index = rng.randrange(record_count)
                    if replacement_index < sample_size:
                        sample[replacement_index] = stripped_line

    if record_count < sample_size:
        raise ValueError(
            f"Only {record_count} datapoints found under {directory}; "
            f"at least {sample_size} are required"
        )
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample datapoints from evaluated response JSONL files in exactly three "
            "dataset directories."
        )
    )
    parser.add_argument(
        "--dataset_dirs",
        type=Path,
        nargs=3,
        required=True,
        metavar=("DIR1", "DIR2", "DIR3"),
        help="Three dataset directories to search recursively (25 datapoints each).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Destination .jsonl file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    if args.output_file.suffix.lower() != ".jsonl":
        parser.error("--output_file must end with .jsonl")

    rng = random.Random(args.seed)
    sampled_records: list[str] = []
    try:
        for directory in args.dataset_dirs:
            directory_sample = sample_directory(directory, 25, rng)
            sampled_records.extend(directory_sample)
            print(f"Sampled 25 datapoints from {directory}")
    except ValueError as error:
        parser.error(str(error))

    rng.shuffle(sampled_records)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output_file:
        for record in sampled_records:
            output_file.write(record + "\n")

    print(f"Saved {len(sampled_records)} datapoints to {args.output_file}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Any

from constant.response_level_score import RESPONSE_LEVEL_SCORE_KEY


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
                records.append(record)
        return records

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of datapoints.")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError(f"{path} must contain only JSON objects.")
    return data


def compare_records(
    records1: list[dict[str, Any]],
    records2: list[dict[str, Any]],
    file1_label: str,
    file2_label: str,
) -> list[dict[str, dict[str, Any]]]:
    if len(records1) != len(records2):
        raise ValueError(f"Files have different lengths: {len(records1)} vs {len(records2)}.")

    comparisons = []
    for idx, (dp1, dp2) in enumerate(zip(records1, records2)):
        if RESPONSE_LEVEL_SCORE_KEY not in dp1:
            raise ValueError(f"file1 datapoint {idx} is missing {RESPONSE_LEVEL_SCORE_KEY!r}.")
        if RESPONSE_LEVEL_SCORE_KEY not in dp2:
            raise ValueError(f"file2 datapoint {idx} is missing {RESPONSE_LEVEL_SCORE_KEY!r}.")

        if dp1[RESPONSE_LEVEL_SCORE_KEY] != dp2[RESPONSE_LEVEL_SCORE_KEY]:
            comparisons.append(split_common_fields(dp1, dp2, file1_label, file2_label))

    return comparisons


def split_common_fields(
    dp1: dict[str, Any],
    dp2: dict[str, Any],
    file1_label: str,
    file2_label: str,
) -> dict[str, dict[str, Any]]:
    common = {}
    only_dp1 = {}
    only_dp2 = {}

    for key in sorted(dp1.keys() | dp2.keys()):
        in_dp1 = key in dp1
        in_dp2 = key in dp2

        if in_dp1 and in_dp2 and dp1[key] == dp2[key]:
            common[key] = dp1[key]
        else:
            if in_dp1:
                only_dp1[key] = dp1[key]
            if in_dp2:
                only_dp2[key] = dp2[key]

    return {"common": common, file1_label: only_dp1, file2_label: only_dp2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two files and save datapoints with different response_level_score values."
    )
    parser.add_argument("--file1", type=Path, required=True, help="First input .json or .jsonl file.")
    parser.add_argument("--file2", type=Path, required=True, help="Second input .json or .jsonl file.")
    parser.add_argument(
        "--out_file",
        type=Path,
        default=Path("out/comparison.json"),
        help="Output .json file for differing datapoint pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file1 = args.file1.expanduser()
    file2 = args.file2.expanduser()
    out_file = args.out_file.expanduser()

    records1 = load_json_or_jsonl(file1)
    records2 = load_json_or_jsonl(file2)
    file1_label = str(file1)
    file2_label = str(file2)
    if file1_label == file2_label:
        raise ValueError(
            f"--file1 and --file2 have the same filename {file1_label!r}; "
        )

    comparisons = compare_records(records1, records2, file1_label, file2_label)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as file:
        json.dump(comparisons, file, ensure_ascii=False, indent=2)

    print(f"num_different: {len(comparisons)}")
    print(f"out_file: {out_file}")


if __name__ == "__main__":
    main()

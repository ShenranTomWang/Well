import argparse
import json, os
from pathlib import Path
from typing import Any
import pandas as pd
from constant.constant import FACTCHECK_RESULTS_KEY


def run_print_examples(args: argparse.Namespace):
    dataset = pd.read_json(args.file, lines=True)
    dataset = dataset.query(args.query) if args.query else dataset
    out_path = os.path.join(args.out_dir, 'printed_examples.json')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        counter = 0
        data = []
        for _, dp in dataset.iterrows():
            if args.k > 0 and counter > args.k:
                break
            skip = False
            for result in dp[FACTCHECK_RESULTS_KEY]:
                if result not in args.filter_results:
                    skip = True
                    break
            if skip:
                continue
            dp = dp.to_dict()
            dp.pop("few_shot_data", None)
            dp.pop("eval_few_shot_data", None)
            data.append(dp)
        json.dump(data, f, ensure_ascii=False, indent=4)
        counter += 1


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def validate_factcheck_results(value: Any, line_number: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(
            f"Line {line_number}: {FACTCHECK_RESULTS_KEY!r} must be a list, "
            f"got {type(value).__name__}."
        )

    invalid_values = [result for result in value if result not in (0, 1)]
    if invalid_values:
        raise ValueError(
            f"Line {line_number}: {FACTCHECK_RESULTS_KEY!r} contains values "
            f"other than 0 or 1: {invalid_values}"
        )

    return value


def compute_incorrect_rate(path: Path, expected_result: int) -> tuple[int, int, float]:
    incorrect_count = 0
    total_count = 0

    for line_number, dp in iter_jsonl(path):
        if FACTCHECK_RESULTS_KEY not in dp:
            raise ValueError(f"Line {line_number}: missing {FACTCHECK_RESULTS_KEY!r}.")

        factcheck_results = validate_factcheck_results(
            dp[FACTCHECK_RESULTS_KEY],
            line_number,
        )
        incorrect_count += sum(
            result != expected_result for result in factcheck_results
        )
        total_count += len(factcheck_results)

    if total_count == 0:
        raise ValueError(f"No factcheck results found in {path}.")

    return incorrect_count, total_count, incorrect_count / total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the fraction of factcheck results that are not expected."
    )
    subparsers = parser.add_subparsers(dest='command')
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the incorrect rate of factcheck results')
    eval_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the input .jsonl file.",
    )
    eval_parser.add_argument(
        "--expected_result",
        type=int,
        choices=(0, 1),
        required=True,
        help="Factcheck result value considered correct.",
    )

    print_parser = subparsers.add_parser('print_examples', help='Print examples from dataset')
    print_parser.add_argument('--file', type=Path, default=None, help='Path to the dataset file (JSONL format)')
    print_parser.add_argument('--k', type=int, default=-1, help='Number of examples to print, default to -1 to print all examples')
    print_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the printed examples')
    print_parser.add_argument('--query', type=str, default=None, help='Only print examples that satisfy the query condition, e.g., "response_level_score == 0" to only print examples with response level score of 0')
    print_parser.add_argument('--filter_results', type=str, default="0,1", help='Filter examples based on factcheck results, e.g., "0" to only print examples with factcheck result of 0, default to "0,1" to print all examples')

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = args.file.expanduser()

    if args.command == 'evaluate':
        incorrect_count, total_count, incorrect_rate = compute_incorrect_rate(
            file_path,
            args.expected_result,
        )
        print(f"expected_result: {args.expected_result}")
        print(f"incorrect_rate: {incorrect_rate}")
        print(f"incorrect_count: {incorrect_count}")
        print(f"total_count: {total_count}")
    elif args.command == 'print_examples':
        args.filter_results = {int(x) for x in args.filter_results.split(',')}
        run_print_examples(args)

if __name__ == "__main__":
    main()

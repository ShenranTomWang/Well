#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import ttest_1samp


def load_annotations(input_file: Path) -> list[dict]:
    records = []
    with input_file.open(encoding="utf-8") as annotated_file:
        for line_number, line in enumerate(annotated_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {input_file}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {input_file}:{line_number}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {input_file}")
    return records


def numeric_score(record: dict, field: str, item_number: int, path: Path):
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"{field} for item {item_number} must be numeric in {path}"
        )
    if not math.isfinite(value):
        raise ValueError(
            f"{field} for item {item_number} must be finite in {path}"
        )
    return float(value)


def one_sided_p_value(differences: list[int], epsilon: float) -> float:
    if len(differences) < 2:
        raise ValueError("At least two eligible datapoints are required per human")
    if all(value == differences[0] for value in differences):
        return 0.0 if statistics.mean(differences) < epsilon else 1.0
    return float(
        ttest_1samp(
            differences,
            popmean=epsilon,
            alternative="less",
        ).pvalue
    )


def benjamini_yekutieli(p_values: list[float], q_fdr: float) -> tuple[set[int], list[float]]:
    test_count = len(p_values)
    harmonic_sum = sum(1 / rank for rank in range(1, test_count + 1))
    sorted_indices = sorted(range(test_count), key=p_values.__getitem__)
    largest_rejected_rank = 0
    for rank, index in enumerate(sorted_indices, start=1):
        threshold = rank * q_fdr / (test_count * harmonic_sum)
        if p_values[index] <= threshold:
            largest_rejected_rank = rank
    rejected = set(sorted_indices[:largest_rejected_rank])

    sorted_adjusted = [
        min(1.0, p_values[index] * test_count * harmonic_sum / rank)
        for rank, index in enumerate(sorted_indices, start=1)
    ]
    for index in range(test_count - 2, -1, -1):
        sorted_adjusted[index] = min(
            sorted_adjusted[index], sorted_adjusted[index + 1]
        )
    adjusted = [0.0] * test_count
    for index, value in zip(sorted_indices, sorted_adjusted):
        adjusted[index] = value
    return rejected, adjusted


def run_alternative_annotator_test(
    input_files: list[Path],
    epsilon: float,
    q_fdr: float,
    min_instances: int,
    win_threshold: float,
) -> dict:
    annotations = [load_annotations(path) for path in input_files]
    shared_count = min(len(records) for records in annotations)
    unmatched_count = sum(len(records) - shared_count for records in annotations)
    indicators = [
        {"llm": [], "human": [], "discarded": 0, "missing": 0}
        for _ in input_files
    ]

    for item_index in range(shared_count):
        item_number = item_index + 1
        records = [annotation[item_index] for annotation in annotations]
        if any(record.get("human_discarded") is True for record in records):
            for values in indicators:
                values["discarded"] += 1
            continue

        human_scores = [
            numeric_score(record, "human_score", item_number, path)
            for record, path in zip(records, input_files)
        ]
        llm_scores = [
            numeric_score(record, "response_level_score", item_number, path)
            for record, path in zip(records, input_files)
        ]
        available_llm_scores = [score for score in llm_scores if score is not None]
        if any(score is None for score in human_scores) or not available_llm_scores:
            for values in indicators:
                values["missing"] += 1
            continue
        if any(score != available_llm_scores[0] for score in available_llm_scores[1:]):
            raise ValueError(
                f"response_level_score differs across files for item {item_number}"
            )

        llm_score = available_llm_scores[0]
        for target_index, values in enumerate(indicators):
            held_out_scores = [
                score for index, score in enumerate(human_scores)
                if index != target_index
            ]
            held_out_mean = statistics.mean(held_out_scores)
            llm_distance = abs(llm_score - held_out_mean)
            human_distance = abs(human_scores[target_index] - held_out_mean)
            values["llm"].append(int(llm_distance <= human_distance))
            values["human"].append(int(human_distance <= llm_distance))

    human_results = []
    p_values = []
    for human_index, values in enumerate(indicators, start=1):
        eligible_count = len(values["llm"])
        if eligible_count < min_instances:
            raise ValueError(
                f"Human {human_index} has only {eligible_count} eligible datapoints; "
                f"at least {min_instances} are required"
            )
        differences = [
            human_indicator - llm_indicator
            for human_indicator, llm_indicator in zip(
                values["human"], values["llm"]
            )
        ]
        p_value = one_sided_p_value(differences, epsilon)
        p_values.append(p_value)
        human_results.append(
            {
                "eligible": eligible_count,
                "discarded": values["discarded"],
                "missing": values["missing"],
                "llm_advantage_probability": statistics.mean(values["llm"]),
                "human_advantage_probability": statistics.mean(values["human"]),
                "mean_indicator_difference": statistics.mean(differences),
                "p_value": p_value,
            }
        )

    rejected, adjusted_p_values = benjamini_yekutieli(p_values, q_fdr)
    for index, result in enumerate(human_results):
        result["adjusted_p_value"] = adjusted_p_values[index]
        result["llm_statistical_win"] = index in rejected

    winning_rate = len(rejected) / len(human_results)
    return {
        "human_results": human_results,
        "winning_rate": winning_rate,
        "average_advantage_probability": statistics.mean(
            result["llm_advantage_probability"] for result in human_results
        ),
        "passes": winning_rate >= win_threshold,
        "unmatched": unmatched_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Alternative Annotator Test from arXiv:2501.10970 on three "
            "ordered human-annotation JSONL files using continuous score ratings."
        )
    )
    parser.add_argument(
        "--input_files",
        type=Path,
        nargs=3,
        required=True,
        metavar=("HUMAN_1", "HUMAN_2", "HUMAN_3"),
        help="Exactly three ordered human-annotated .jsonl files.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="Cost-benefit margin (default: 0.15, recommended for skilled annotators).",
    )
    parser.add_argument(
        "--q_fdr",
        type=float,
        default=0.05,
        help="Benjamini-Yekutieli target FDR (default: 0.05).",
    )
    parser.add_argument(
        "--min_instances",
        type=int,
        default=30,
        help="Minimum eligible datapoints per human (default: 30).",
    )
    parser.add_argument(
        "--win_threshold",
        type=float,
        default=0.5,
        help="Winning-rate threshold required to pass (default: 0.5).",
    )
    args = parser.parse_args()

    for input_file in args.input_files:
        if input_file.suffix.lower() != ".jsonl":
            parser.error(f"Input file must end with .jsonl: {input_file}")
    if not 0 <= args.epsilon <= 1:
        parser.error("--epsilon must be between 0 and 1")
    if not 0 < args.q_fdr < 1:
        parser.error("--q_fdr must be between 0 and 1")
    if args.min_instances < 2:
        parser.error("--min_instances must be at least 2")
    if not 0 <= args.win_threshold <= 1:
        parser.error("--win_threshold must be between 0 and 1")

    try:
        result = run_alternative_annotator_test(
            args.input_files,
            args.epsilon,
            args.q_fdr,
            args.min_instances,
            args.win_threshold,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for index, (input_file, human_result) in enumerate(
        zip(args.input_files, result["human_results"]), start=1
    ):
        outcome = "WIN" if human_result["llm_statistical_win"] else "NO WIN"
        print(f"LLM vs Human {index} ({input_file}) [{outcome}]:")
        print(
            f"  LLM advantage probability: "
            f"{human_result['llm_advantage_probability']:.6f}"
        )
        print(
            f"  Human advantage probability: "
            f"{human_result['human_advantage_probability']:.6f}"
        )
        print(f"  Mean W_human - W_LLM: {human_result['mean_indicator_difference']:.6f}")
        print(f"  One-sided p-value: {human_result['p_value']:.6g}")
        print(f"  BY-adjusted p-value: {human_result['adjusted_p_value']:.6g}")
        print(f"  Eligible datapoints: {human_result['eligible']}")
        print(f"  Ignored discarded datapoints: {human_result['discarded']}")
        print(f"  Skipped datapoints with missing scores: {human_result['missing']}")

    outcome = "PASSED" if result["passes"] else "FAILED"
    print(f"Alt-test result: {outcome}")
    print(f"Winning rate: {result['winning_rate']:.6f}")
    print(
        f"Average advantage probability: "
        f"{result['average_advantage_probability']:.6f}"
    )
    print(f"Unmatched datapoints across files: {result['unmatched']}")


if __name__ == "__main__":
    main()

"""Calculate agreement statistics for a WildChat reannotation JSONL file.

Example:
    python calculate_reannotation_agreement.py \
        --file wildchat_reannotator_annotated.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def calculate_metrics(pairs: list[tuple[object, object]]) -> tuple[float, float]:
    agreement = sum(first == second for first, second in pairs) / len(pairs)
    first_counts = Counter(first for first, _ in pairs)
    second_counts = Counter(second for _, second in pairs)
    labels = first_counts.keys() | second_counts.keys()
    expected = sum(
        first_counts[label] * second_counts[label] for label in labels
    ) / len(pairs) ** 2
    kappa = (agreement - expected) / (1 - expected) if expected != 1 else float("nan")
    return agreement, kappa


def print_confusion_matrix(pairs: list[tuple[object, object]]) -> None:
    """Print original annotator rows against reannotation columns."""
    comparable = [pair for pair in pairs if pair[1] is not None]
    if not comparable:
        print("\nConfusion matrix: N/A (all reannotations were discarded)")
        return

    counts = Counter(comparable)
    total = len(comparable)

    def cell(original: bool, reannotation: bool) -> str:
        count = counts[original, reannotation]
        return f"{count} ({count / total:.2%})"

    print("\nConfusion matrix (count and % of non-discarded records)")
    print("Rows: annotator; columns: reannotation")
    print(f"{'':14}{'TPQ':>16}{'FPQ':>16}")
    print(f"{'TPQ':14}{cell(False, False):>16}{cell(False, True):>16}")
    print(f"{'FPQ':14}{cell(True, False):>16}{cell(True, True):>16}")


def print_label_counts(pairs: list[tuple[object, object]]) -> None:
    original_counts = Counter(original for original, _ in pairs)
    reannotation_counts = Counter(reannotation for _, reannotation in pairs)
    original_total = len(pairs)
    reannotation_total = original_total - reannotation_counts[None]

    def value(count: int, total: int) -> str:
        return f"{count} ({count / total:.2%})" if total else "0 (N/A)"

    print("\nLabel counts")
    print(f"{'':16}{'TPQ':>16}{'FPQ':>16}{'Discard':>16}")
    print(
        f"{'Annotator':16}"
        f"{value(original_counts[False], original_total):>16}"
        f"{value(original_counts[True], original_total):>16}"
        f"{'0 (0.00%)':>16}"
    )
    print(
        f"{'Reannotation':16}"
        f"{value(reannotation_counts[False], reannotation_total):>16}"
        f"{value(reannotation_counts[True], reannotation_total):>16}"
        f"{value(reannotation_counts[None], original_total):>16}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate raw annotator agreement and Cohen's kappa from JSONL "
            "downloaded by build_question_reannotator.py."
        )
    )
    parser.add_argument("--file", required=True, type=Path, help="Reannotated JSONL file.")
    args = parser.parse_args()

    pairs: list[tuple[object, object]] = []
    with args.file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")

            original = record.get("has_false_presupposition")
            reannotation = record.get("has_false_presupposition_reannotation", "missing")
            if not isinstance(original, bool):
                raise ValueError(
                    f"Line {line_number} has no boolean 'has_false_presupposition'."
                )
            if reannotation is not None and not isinstance(reannotation, bool):
                raise ValueError(
                    f"Line {line_number} has no boolean or null "
                    "'has_false_presupposition_reannotation'."
                )
            pairs.append((original, reannotation))

    if not pairs:
        raise ValueError(f"{args.file} contains no reannotated records.")

    agreement, kappa = calculate_metrics(pairs)
    comparable = [pair for pair in pairs if pair[1] is not None]
    discarded = len(pairs) - len(comparable)

    print(f"Records: {len(pairs)}")
    print(f"Discarded reannotations: {discarded}")
    print(f"Agreement (discard is a third category): {agreement:.4f} ({agreement:.2%})")
    print(f"Cohen's kappa (discard is a third category): {kappa:.4f}")

    if comparable:
        comparable_agreement, comparable_kappa = calculate_metrics(comparable)
        print(f"Comparable non-discarded records: {len(comparable)}")
        print(
            f"Agreement (excluding discarded): {comparable_agreement:.4f} "
            f"({comparable_agreement:.2%})"
        )
        print(f"Cohen's kappa (excluding discarded): {comparable_kappa:.4f}")
    else:
        print("Agreement and kappa excluding discarded: N/A")

    print_label_counts(pairs)
    print_confusion_matrix(pairs)


if __name__ == "__main__":
    main()

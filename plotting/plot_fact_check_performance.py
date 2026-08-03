import json
import argparse
from pathlib import Path

import pandas as pd


def validate_factcheck_results(value, line_number: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(
            f"Line {line_number}: {'factcheck_results'!r} must be a list, "
            f"got {type(value).__name__}."
        )

    invalid_values = [result for result in value if result not in (0, 1)]
    if invalid_values:
        raise ValueError(
            f"Line {line_number}: {'factcheck_results'!r} contains values "
            f"other than 0 or 1: {invalid_values}"
        )

    return value

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


def compute_incorrect_rate(path: Path, expected_result: int) -> tuple[int, int, float]:
    incorrect_count = 0
    total_count = 0

    for line_number, dp in iter_jsonl(path):
        if "factcheck_results" not in dp:
            continue
            # raise ValueError(f"Line {line_number}: missing {"factcheck_results"!r}.")

        factcheck_results = validate_factcheck_results(
            dp["factcheck_results"],
            line_number,
        )

        incorrect_count += sum(
            result != expected_result for result in factcheck_results
        )
        total_count += len(factcheck_results)

    if total_count == 0:
        raise ValueError(f"No factcheck results found in {path}.")

    return incorrect_count, total_count, incorrect_count / total_count


def base_file_stem(file_stem: str) -> str:
    for suffix in ("_thinking_ablation", "_thinking", "_ablation"):
        if file_stem.endswith(suffix):
            return file_stem.removesuffix(suffix)
    return file_stem


def variant_hatch(file_stem: str) -> str | None:
    if file_stem.endswith("_thinking_ablation"):
        return "xxx"
    if file_stem.endswith("_thinking"):
        return "\\\\\\"
    if file_stem.endswith("_ablation"):
        return "///"
    return None


def parse_pairs(specs: list[str], kind: str) -> list[tuple[str, str]]:
    pairs = []
    for spec in specs:
        if "/" not in spec:
            raise ValueError(f"Invalid {kind} {spec!r}; expected display_name/value.")
        display_name, value = spec.split("/", maxsplit=1)
        if not display_name or not value:
            raise ValueError(f"Invalid {kind} {spec!r}; both parts are required.")
        pairs.append((display_name, value))
    return pairs


def load_results(
    out_root: Path,
    datasets: list[tuple[str, str]],
    settings: list[tuple[str, str]]):
    rows = []

    dataset_names, expected_values = zip(*datasets)
    subsets = ["TPQ" if "TPQ" in dataset else "FPQ" for dataset in dataset_names]
    dataset_dir = out_root / dataset_names[0] / "Check_Gold"
    model_names = sorted([path.name for path in dataset_dir.iterdir() if path.is_dir()])

    for model in model_names:
        for setting_name, file_stem in settings:

            # Load common results
            curr_row = { "model": model, "setting": setting_name }

            # Load results for each of TPQ and FPQ
            for subset, dataset, expected_value in zip(subsets, dataset_names, expected_values):

                path = out_root / dataset / "Check_Gold" / model / f"{file_stem}.jsonl"
                incorrect, total, incorrect_rate = None, None, None

                if not path.is_file():
                    continue

                incorrect, total, incorrect_rate = compute_incorrect_rate(path, int(expected_value))
                curr_row[f"{subset} Accuracy"] = f"{(1.0 - incorrect_rate) * 100.0:.1f} ({total - incorrect}/{total})"

            rows.append(curr_row)

    df = pd.DataFrame(rows)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vertical grouped bars comparing fact-check accuracy across "
            "models, settings, and datasets."
        )
    )
    parser.add_argument(
        "--out_root",
        required=True,
        type=Path,
        help="Root containing <dataset>/Check_Gold/<model>/<setting>.jsonl.",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        nargs="+",
        help="Datasets as dataset_name/expected_result pairs.",
    )
    parser.add_argument(
        "--settings",
        required=True,
        nargs="+",
        help="Settings as display_name/file_stem pairs.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Fact-checking performance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_pairs(args.datasets, "dataset")
    settings = parse_pairs(args.settings, "setting")
    for dataset, expected_result in datasets:
        if expected_result not in {"0", "1"}:
            raise ValueError(
                f"Expected result for {dataset!r} must be 0 or 1, got {expected_result!r}."
            )

    df = load_results(args.out_root, datasets, settings)
    latex_code = df.dropna().to_latex(index=False)
    print(latex_code)

if __name__ == "__main__":
    main()

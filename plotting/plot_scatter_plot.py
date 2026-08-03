import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from plotting.plot_all_response_level_scores import parse_settings
from plotting.plot_response_level_scores import SCORE_KEY, load_distributions

IGNORED_MODELS = {"gemini-2.5-flash", "Olmo-3-7B-Instruct-SFT", "Olmo-3-7B-Instruct-DPO"}
IGNORED_CONDITIONS = {"RAG=0_Llama3-Med42-8B"}
METHOD_COLORS = {
    "Direct QA": "black",
    "GEPA (FPQ)": "#ff7f0e",
    "GEPA (FPQ + TPQ)": "#3db7c5",
    "Presupposition Extraction + Fact Checking": "#2ca02c",
    "PreWoMe": "#d62728",
    "FAITH": "#9467bd",
    "FP Identification": "#8c564b",
    "Question to Statement": "#fe52ca",
    "Self-Dual-Critique": "#7f7f7f",
    "Fine-tuning": "#bcbd22",
}


def average_score(rows: list[dict[str, object]]) -> float:
    nonzero_total = 0
    weighted_sum = 0.0
    for row in rows:
        for score, count in row["counts"].items():
            try:
                numeric_score = float(score)
            except ValueError as error:
                raise ValueError(
                    f"Score {score!r} in {row['path']} is not numeric."
                ) from error
            if numeric_score == 0:
                continue
            weighted_sum += numeric_score * count
            nonzero_total += count

    if nonzero_total == 0:
        raise ValueError("Cannot average a collection containing only zero scores.")
    return weighted_sum / nonzero_total


def discover_models(fpq_root: Path, tpq_root: Path, storage_dir: str) -> list[str]:
    fpq_dir = fpq_root / storage_dir
    tpq_dir = tpq_root / storage_dir
    if not fpq_dir.is_dir() or not tpq_dir.is_dir():
        return []
    fpq_models = {path.name for path in fpq_dir.iterdir() if path.is_dir()}
    tpq_models = {path.name for path in tpq_dir.iterdir() if path.is_dir()}
    return sorted((fpq_models & tpq_models) - IGNORED_MODELS)


def load_points(
    fpq_root: Path,
    tpq_root: Path,
    settings: list[tuple[str, str]],
    score_key: str,
) -> list[dict[str, object]]:
    points = []
    for method, storage_dir in settings:
        for model in discover_models(fpq_root, tpq_root, storage_dir):
            fpq_rows = load_distributions(fpq_root / storage_dir / model, score_key)
            tpq_rows = load_distributions(tpq_root / storage_dir / model, score_key)
            if not fpq_rows or not tpq_rows:
                print(f"Skipping incomplete scores: {method} / {model}")
                continue
            fpq_by_condition = {row["label"]: row for row in fpq_rows}
            tpq_by_condition = {row["label"]: row for row in tpq_rows}
            conditions = sorted(
                (fpq_by_condition.keys() & tpq_by_condition.keys())
                - IGNORED_CONDITIONS
            )
            unmatched = fpq_by_condition.keys() ^ tpq_by_condition.keys()
            for condition in sorted(unmatched):
                print(
                    f"Skipping unmatched condition: {method} / {model} / {condition}"
                )
            for condition in conditions:
                try:
                    points.append(
                        {
                            "method": method,
                            "model": model,
                            "condition": condition,
                            "fpq": average_score([fpq_by_condition[condition]]),
                            "tpq": average_score([tpq_by_condition[condition]]),
                        }
                    )
                except ValueError as error:
                    print(
                        f"Skipping {method} / {model} / {condition}: {error}"
                    )
    return points


def non_dominated_points(
    points: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected = []
    for point in points:
        dominated = any(
            other["tpq"] >= point["tpq"]
            and other["fpq"] >= point["fpq"]
            and (other["tpq"] > point["tpq"] or other["fpq"] > point["fpq"])
            for other in points
        )
        if not dominated:
            selected.append(point)
    return sorted(selected, key=lambda point: (point["tpq"], point["fpq"]))


def plot_points(
    datasets: list[list[dict[str, object]]],
    out_path: Path,
    titles: list[str],
    fpq_ratio: float | None = None,
) -> None:
    if any(not points for points in datasets):
        raise ValueError("No paired FPQ/TPQ scores were loaded for a dataset pair.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(9 * len(datasets), 7))
    axes = np.atleast_1d(axes)
    methods = list(
        dict.fromkeys(
            point["method"] for points in datasets for point in points
        )
    )
    missing_colors = set(methods) - METHOD_COLORS.keys()
    if missing_colors:
        raise ValueError(
            "Missing colors for methods: " + ", ".join(sorted(missing_colors))
        )

    performance_colors = None
    for ax, points, title in zip(axes, datasets, titles):
        if fpq_ratio is not None:
            tpq_values = np.linspace(0.9, 5.1, 301)
            fpq_values = np.linspace(0.9, 5.1, 301)
            tpq_grid, fpq_grid = np.meshgrid(tpq_values, fpq_values)
            performance = (
                (1 - fpq_ratio) * tpq_grid + fpq_ratio * fpq_grid
            )
            performance_colors = ax.contourf(
                tpq_grid,
                fpq_grid,
                performance,
                levels=np.linspace(1, 5, 17),
                cmap="viridis",
                alpha=0.3,
                zorder=0,
            )

        for method in methods:
            group_points = [point for point in points if point["method"] == method]
            ax.scatter(
                [point["tpq"] for point in group_points],
                [point["fpq"] for point in group_points],
                s=140 if method == "Direct QA" else 70,
                color=METHOD_COLORS[method],
                marker="o",
                alpha=0.85,
            )

        ax.set_xlabel("Average TPQ response-level score")
        ax.set_ylabel("Average FPQ response-level score")
        ax.set_xlim(0.9, 5.1)
        ax.set_ylim(0.9, 5.1)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    if performance_colors is not None:
        colorbar_axis = fig.add_axes([0.91, 0.23, 0.015, 0.64])
        colorbar = fig.colorbar(performance_colors, cax=colorbar_axis)
        colorbar.set_ticks(np.arange(1, 6))
        colorbar.set_label("Weighted performance score (1–5)")

    legend_handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            marker="o",
            linestyle="None",
            markersize=10 if method == "Direct QA" else 7,
            label=method,
        )
        for method in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(legend_handles)),
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.2, wspace=0.25, right=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot condition-level average FPQ versus TPQ response-level scores for "
            "all models and methods in side-by-side dataset pairs."
        )
    )
    parser.add_argument("--fpq_roots", required=True, nargs="+", type=Path)
    parser.add_argument("--tpq_roots", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--settings",
        required=True,
        nargs="+",
        help="Methods as method_name/storage_dir pairs, in display order.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--score_key", default=SCORE_KEY)
    parser.add_argument("--titles", required=True, nargs="+")
    parser.add_argument(
        "--FPQ_ratio",
        default=0.13,
        type=float,
        help=(
            "FPQ fraction of the dataset (between 0 and 1). If provided, draw "
            "equal-performance contours using normalized FPQ and TPQ scores."
        ),
    )
    args = parser.parse_args()
    if args.FPQ_ratio is not None and not 0 <= args.FPQ_ratio <= 1:
        parser.error("--FPQ_ratio must be between 0 and 1.")
    if not (len(args.fpq_roots) == len(args.tpq_roots) == len(args.titles)):
        parser.error("--fpq_roots, --tpq_roots, and --titles must have equal lengths.")
    return args


def main() -> None:
    args = parse_args()
    settings = parse_settings(args.settings)
    datasets = [
        load_points(fpq_root, tpq_root, settings, args.score_key)
        for fpq_root, tpq_root in zip(args.fpq_roots, args.tpq_roots)
    ]
    plot_points(datasets, args.out, args.titles, args.FPQ_ratio)
    for title, points in zip(args.titles, datasets):
        for point in points:
            print(
                f"{title} / {point['method']} / {point['model']} / "
                f"{point['condition']}: TPQ={point['tpq']:.4f}, "
                f"FPQ={point['fpq']:.4f}"
            )
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

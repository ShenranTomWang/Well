import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SCORE_KEY = "response_level_score"
SCORE_COLORS = {
    "0": "#1f77b4",
    "1": "#cd2b02",
    "2": "#e3d61e",
    "3": "#30a71d",
    "4": "#ff8c00",
    "5": "#07b8c8",
    "-1": "#7f7f7f",
}
FALLBACK_SCORE_COLORS = (
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
)


def score_color(score: str, score_idx: int) -> str:
    if score in SCORE_COLORS:
        return SCORE_COLORS[score]
    return FALLBACK_SCORE_COLORS[score_idx % len(FALLBACK_SCORE_COLORS)]


def discover_jsonl_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {base_dir}")
    return sorted(path for path in base_dir.glob("*response_level_score_evaluated.jsonl") if path.is_file())


def read_scores(path: Path, score_key: str) -> list[str]:
    scores = []
    with path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if score_key not in row:
                print(f"Skipping line without {score_key!r}: {path}:{line_number}")
                continue
            scores.append(format_score(row[score_key]))
    return scores


def format_score(score: object) -> str:
    if isinstance(score, float) and score.is_integer():
        return str(int(score))
    return str(score)


def label_for_path(path: Path, base_dir: Path) -> str:
    relative_path = path.relative_to(base_dir)
    return str(relative_path.with_suffix("")).removesuffix("_response_level_score_evaluated")


def load_distributions(base_dir: Path, score_key: str) -> list[dict[str, object]]:
    rows = []
    for path in discover_jsonl_files(base_dir):
        scores = read_scores(path, score_key)
        if not scores:
            print(f"Skipping file with no usable scores: {path}")
            continue

        counts = Counter(scores)
        rows.append(
            {
                "label": label_for_path(path, base_dir),
                "path": path,
                "counts": counts,
                "total": len(scores),
            }
        )
    return rows


def draw_distribution_axis(
    ax,
    rows: list[dict[str, object]],
    score_order: list[str],
    title: str,
    labelled_scores: set[str],
) -> None:
    y_positions = list(range(len(rows)))
    for y, row in zip(y_positions, rows):
        left = 0.0
        counts = row["counts"]
        total = row["total"]
        for score_idx, score in enumerate(score_order):
            pct = counts.get(score, 0) / total * 100
            if pct == 0:
                continue
            label = None
            if score not in labelled_scores:
                label = f"Score {score}"
                labelled_scores.add(score)
            ax.barh(
                y,
                pct,
                left=left,
                color=score_color(score, score_idx),
                edgecolor="white",
                linewidth=0.8,
                label=label,
            )
            if pct >= 6:
                ax.text(
                    left + pct / 2,
                    y,
                    f"{pct:.0f}%",
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white",
                )
            left += pct

    labels = [f"{row['label']}" for row in rows]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(title, loc="left", fontweight="bold")


def score_legend_handles(score_order: list[str]) -> list[Patch]:
    return [
        Patch(color=score_color(score, idx), label=f"Score {score}")
        for idx, score in enumerate(score_order)
    ]


def collect_legend(fig, axes, score_order: list[str]) -> None:
    fig.legend(
        handles=score_legend_handles(score_order),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=max(1, len(score_order)),
    )


def plot_grouped_distributions(
    grouped_rows: list[tuple[str, list[dict[str, object]]]],
    score_order: list[str],
    out_path: Path,
    title: str,
) -> None:
    grouped_rows = [(label, rows) for label, rows in grouped_rows if rows]
    if not grouped_rows:
        raise ValueError("No evaluation results were loaded.")

    height = sum(max(2.8, 0.55 * len(rows) + 1.2) for _, rows in grouped_rows) + 1.2
    fig, axes_grid = plt.subplots(
        nrows=len(grouped_rows),
        ncols=1,
        figsize=(12, height),
        sharex=True,
        squeeze=False,
        gridspec_kw={
            "height_ratios": [max(1, len(rows)) for _, rows in grouped_rows],
        },
    )
    axes = list(axes_grid[:, 0])
    labelled_scores = set()
    for ax, (group_label, rows) in zip(axes, grouped_rows):
        draw_distribution_axis(ax, rows, score_order, group_label, labelled_scores)

    axes[-1].set_xlabel("Response-level score distribution (%)")
    fig.suptitle(title, y=0.995)
    collect_legend(fig, axes, score_order)

    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot response-level score distributions for two directories of JSONL files "
            "as horizontal 100% stacked bars."
        )
    )
    parser.add_argument(
        "--fpq_dir",
        required=True,
        type=Path,
        help="Directory containing the already-filtered FPQ .jsonl files to plot.",
    )
    parser.add_argument(
        "--tpq_dir",
        required=True,
        type=Path,
        help="Directory containing the already-filtered TPQ .jsonl files to plot.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path where the plot should be saved.",
    )
    parser.add_argument(
        "--score_key",
        default=SCORE_KEY,
        help=f"JSONL field containing the response-level score. Defaults to {SCORE_KEY!r}.",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fpq_rows = load_distributions(args.fpq_dir, args.score_key)
    tpq_rows = load_distributions(args.tpq_dir, args.score_key)
    if not fpq_rows and not tpq_rows:
        raise ValueError("No JSONL files with usable scores were loaded.")

    plot_grouped_distributions(
        grouped_rows=[
            ("FPQ", fpq_rows),
            ("TPQ", tpq_rows),
        ],
        score_order=["0", "1", "2", "3", "4", "5"],
        out_path=args.out,
        title=args.title,
    )
    loaded_rows = fpq_rows + tpq_rows

    print(f"Loaded {len(loaded_rows)} evaluated file(s):")
    for row in loaded_rows:
        print(f"- {row['label']} n={row['total']} {row['path']}")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

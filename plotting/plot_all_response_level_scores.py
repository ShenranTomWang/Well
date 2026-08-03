import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.plot_response_level_scores import (
    SCORE_KEY,
    collect_legend,
    draw_distribution_axis,
    load_distributions,
)


SCORE_ORDER = ["0", "1", "2", "3", "4", "5"]


def plot_all_settings(
    fpq_root: Path,
    tpq_root: Path,
    model: str,
    settings: list[tuple[str, str]],
    score_key: str,
    out_path: Path,
    title: str,
) -> list[tuple[str, str, list[dict[str, object]]]]:
    loaded = []
    for setting_name, storage_dir in settings:
        fpq_rows = load_distributions(fpq_root / storage_dir / model, score_key)
        tpq_rows = load_distributions(tpq_root / storage_dir / model, score_key)
        if not fpq_rows and not tpq_rows:
            print(f"Skipping setting with no usable scores: {setting_name} ({storage_dir})")
            continue
        loaded.extend([(setting_name, "FPQ", fpq_rows), (setting_name, "TPQ", tpq_rows)])

    plotted_settings = [
        setting_name
        for setting_name, _ in settings
        if any(item_setting == setting_name and rows for item_setting, _, rows in loaded)
    ]
    if not plotted_settings:
        raise ValueError("No JSONL files with usable scores were loaded.")

    row_heights = [
        max(
            2,
            len(next((rows for setting, kind, rows in loaded if setting == name and kind == "FPQ"), [])),
            len(next((rows for setting, kind, rows in loaded if setting == name and kind == "TPQ"), [])),
        )
        for name in plotted_settings
    ]
    fig, axes = plt.subplots(
        nrows=len(plotted_settings),
        ncols=2,
        figsize=(
            20,
            0.8 * (sum(0.35 * height + 0.75 for height in row_heights) + 1.0),
        ),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": row_heights, "hspace": 0.3, "wspace": 0.4},
    )

    labelled_scores = set()
    for row_idx, setting in enumerate(plotted_settings):
        for col_idx, kind in enumerate(("FPQ", "TPQ")):
            ax = axes[row_idx, col_idx]
            rows = next(
                (rows for item_setting, item_kind, rows in loaded
                 if item_setting == setting and item_kind == kind),
                [],
            )
            if rows:
                draw_distribution_axis(
                    ax,
                    rows,
                    SCORE_ORDER,
                    f"{setting} — {kind}",
                    labelled_scores,
                )
            else:
                ax.set_axis_off()
                ax.set_title(f"{setting} — {kind} (no data)", loc="left", fontweight="bold")

    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Response-level score distribution (%)")
    fig.suptitle(title, y=0.95)
    collect_legend(fig, axes.ravel(), SCORE_ORDER)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=0.5, w_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot response-level score distributions for multiple settings and one "
            "model in a single figure. Roots must contain <storage_dir>/<model>/ directories."
        )
    )
    parser.add_argument("--fpq_root", required=True, type=Path)
    parser.add_argument("--tpq_root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--settings",
        required=True,
        nargs="+",
        help="Settings as setting_name/storage_dir pairs, in display order.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--score_key", default=SCORE_KEY)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def parse_settings(setting_specs: list[str]) -> list[tuple[str, str]]:
    settings = []
    for spec in setting_specs:
        if "/" not in spec:
            raise ValueError(
                f"Invalid setting {spec!r}; expected setting_name/storage_dir."
            )
        setting_name, storage_dir = spec.split("/", maxsplit=1)
        if not setting_name or not storage_dir:
            raise ValueError(
                f"Invalid setting {spec!r}; setting name and storage directory are required."
            )
        settings.append((setting_name, storage_dir))
    return settings


def main() -> None:
    args = parse_args()
    loaded = plot_all_settings(
        fpq_root=args.fpq_root,
        tpq_root=args.tpq_root,
        model=args.model,
        settings=parse_settings(args.settings),
        score_key=args.score_key,
        out_path=args.out,
        title=args.title or f"Response-level score distributions: {args.model}",
    )
    for setting, kind, rows in loaded:
        print(f"{setting} / {kind}: loaded {len(rows)} evaluated file(s)")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

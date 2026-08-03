import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from tqdm import tqdm


DATASET_PAIRS = (
    ("CancerMyth", "CancerMythNFP", "CancerMyth", "CancerMyth"),
    ("QA2FPQ", "QA2TPQ", "QA$^2$", "QA2"),
    ("SynQA2FPQ", "SynQA2TPQ", "Syn-QA$^2$", "SynQA2"),
    ("CREPEFPQ", "CREPETPQ", "CREPE", "CREPE"),
)
METHODS = (
    ("Direct QA", "Direct_QA"),
    ("GEPA (FPQ)", "Direct_QA_GEPA"),
    ("GEPA (FPQ + TPQ)", "Direct_QA_GEPA_balanced"),
    ("Presupposition Extraction + Fact Checking", "Final_Response"),
    ("PreWoMe", "Feedback_Action_Final_Response"),
    ("FAITH", "FAITH"),
    ("FP Identification", "FP_Identification_Final_Response"),
    ("Question to Statement", "Statement_Final_Response"),
    ("Self-Dual-Critique", "SDualCritique"),
    ("Fine-tuning", "FalseQA"),
)
SCORE_KEY = "response_level_score"
SCORES = range(1, 6)
IGNORED_MODELS = {"Olmo-3-7B-Instruct-SFT", "Olmo-3-7B-Instruct-DPO", "gemini-2.5-flash"}


def condition_name(path):
    return path.name.removesuffix("_response_level_score_evaluated.jsonl")


def rag_name(condition: str):
    reasoning = condition.endswith("_thinking")
    condition = condition.removesuffix("_thinking")
    value, _, checker = condition.removeprefix("RAG=").partition("_")
    rag = {"0": "None", "4": "Top-4", "all": "All", "web": "Web"}.get(value, value)
    checker_name = {
        "gemini_checked": "LLM Check",
        "minichecked": "MiniCheck",
        "transformers_checked": "LLM Check",
    }.get(checker, checker.replace("_", " ").title())
    name = f"{rag} ({checker_name})" if checker else rag
    return f"{name} + reasoning" if reasoning else name


def condition_sort_key(condition):
    reasoning = condition.endswith("_thinking")
    condition = condition.removesuffix("_thinking")
    value, _, checker = condition.removeprefix("RAG=").partition("_")
    return ({"0": 0, "4": 1, "all": 2, "web": 3}.get(value, 4), checker, reasoning)


def score_percentages(path):
    counts = Counter()
    with path.open() as f_in:
        for line_number, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if SCORE_KEY not in row:
                raise KeyError(f"Missing {SCORE_KEY!r} in {path}:{line_number}")
            score = float(row[SCORE_KEY])
            if score == 0:
                continue
            if not score.is_integer() or int(score) not in SCORES:
                raise ValueError(f"Unexpected {SCORE_KEY}={score!r} in {path}:{line_number}")
            counts[int(score)] += 1

    total = sum(counts.values())
    if not total:
        return [0] * len(SCORES)
    return [round(100 * counts[score] / total) for score in SCORES]


def evaluated_files(directory):
    return {
        condition_name(path): path
        for path in directory.glob("*_response_level_score_evaluated.jsonl")
    }


def construct_table(data_dir, fpq_dataset, tpq_dataset, model):
    rows = []
    for method, storage_dir in METHODS:
        fpq_files = evaluated_files(data_dir / fpq_dataset / storage_dir / model)
        tpq_files = evaluated_files(data_dir / tpq_dataset / storage_dir / model)
        for condition in sorted(fpq_files.keys() & tpq_files.keys(), key=condition_sort_key):
            rows.append(
                [method, rag_name(condition)]
                + score_percentages(fpq_files[condition])
                + score_percentages(tpq_files[condition])
            )

        for condition in sorted(fpq_files.keys() ^ tpq_files.keys()):
            print(f"Skipping unmatched result: {method} / {model} / {condition}")

    columns = ["Method", "RAG"] + [f"FPQ_S{i}" for i in SCORES] + [f"TPQ_S{i}" for i in SCORES]
    return pd.DataFrame(rows, columns=columns)


def discover_models(data_dir, fpq_dataset, tpq_dataset):
    models = set()
    for _, storage_dir in METHODS:
        for dataset in (fpq_dataset, tpq_dataset):
            directory = data_dir / dataset / storage_dir
            if directory.is_dir():
                models.update(path.name for path in directory.iterdir() if path.is_dir())
    return sorted(models)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path, help="Root folder containing evaluated result JSONL files")
    parser.add_argument("--out_dir", required=True, type=Path, help="Folder to output the table .tex files")
    args = parser.parse_args()

    jobs = [
        (fpq_dataset, tpq_dataset, display_name, filename, model)
        for fpq_dataset, tpq_dataset, display_name, filename in DATASET_PAIRS
        for model in discover_models(args.data_dir, fpq_dataset, tpq_dataset)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(jobs), desc="Creating Tables") as p_bar:
        for fpq_dataset, tpq_dataset, display_name, filename, model in jobs:
            if model in IGNORED_MODELS:
                print(f"Skipping ignored model: {model}")
                p_bar.update(1)
                continue
            df = construct_table(args.data_dir, fpq_dataset, tpq_dataset, model)
            if df.empty:
                print(f"Skipping table with no paired results: {fpq_dataset} / {model}")
                p_bar.update(1)
                continue
            table_tex = df_to_latex_table(df, model, display_name, filename)
            out_path = args.out_dir / f"{filename}-{model.removesuffix("-preview")}.tex"
            with out_path.open("w") as f_out:
                f_out.write(table_tex + "\n")
            p_bar.update(1)

    # Example, just to show what the df should contain. Don't use this data.
    # rows = [
    #     ("Direct QA", "None", 69, 1, 1, 0, 29, 0, 0, 0, 0, 100),
    #     ("Direct QA", "Top-4", 62, 4, 0, 0, 34, 1, 0, 0, 1, 98),
    #     ("Direct QA", "All", 52, 5, 0, 3, 40, 0, 0, 0, 1, 99),
    #     ("Direct QA", "Web", 68, 2, 0, 1, 29, 0, 0, 0, 0, 100),
    #     ("GEPA (FPQ)", "None", 1, 2, 0, 0, 97, 14, 61, 0, 4, 21),
    #     ("GEPA (FPQ)", "Top-4", 3, 2, 0, 0, 95, 22, 51, 0, 2, 25),
    #     ("GEPA (FPQ)", "All", 4, 0, 0, 0, 96, 19, 55, 0, 4, 22),
    #     ("GEPA (FPQ)", "Web", 5, 3, 0, 0, 92, 9, 23, 0, 5, 63),
    #     ("GEPA (FPQ+TPQ)", "None", 49, 1, 0, 0, 50, 1, 2, 0, 1, 96),
    #     ("GEPA (FPQ+TPQ)", "Top-4", 45, 4, 0, 1, 50, 1, 1, 0, 2, 96),
    #     ("GEPA (FPQ+TPQ)", "All", 35, 1, 0, 0, 64, 3, 2, 0, 2, 93),
    #     ("GEPA (FPQ+TPQ)", "Web", 51, 3, 0, 1, 45, 0, 3, 0, 1, 96),
    #     ("Decompose + Fact Check", "None (Gemini)", 40, 2, 0, 0, 58, 11, 20, 0, 5, 64),
    #     ("Decompose + Fact Check", "Top-4 (Gemini)", 42, 3, 0, 0, 55, 17, 21, 0, 2, 60),
    #     ("Decompose + Fact Check", "Top-4 (MiniCheck)", 5, 0, 0, 1, 94, 18, 30, 0, 8, 44),
    #     ("Decompose + Fact Check", "All (Gemini)", 31, 3, 0, 1, 65, 16, 19, 0, 5, 60),
    #     ("Decompose + Fact Check", "All (MiniCheck)", 9, 1, 0, 1, 89, 27, 20, 0, 10, 43),
    #     ("Decompose + Fact Check", "Web (Gemini)", 30, 2, 0, 0, 68, 15, 21, 0, 2, 62),
    #     ("PreWoMe", "None (Gemini)", 10, 1, 0, 1, 88, 14, 23, 0, 6, 57),
    #     ("PreWoMe", "Top-4 (Gemini)", 7, 2, 0, 1, 90, 26, 14, 0, 5, 55),
    #     ("PreWoMe", "All (Gemini)", 7, 2, 0, 0, 91, 8, 31, 0, 6, 55),
    #     ("PreWoMe", "Web (Gemini)", 10, 1, 0, 1, 88, 19, 21, 0, 6, 54),
    #     ("FP Identification", "None", 5, 2, 0, 0, 93, 93, 1, 0, 0, 5),
    #     ("FP Identification", "Top-4", 48, 3, 0, 1, 48, 28, 1, 0, 0, 71),
    #     ("FP Identification", "All", 48, 1, 3, 0, 48, 34, 0, 0, 1, 65),
    #     ("FP Identification", "Web", 71, 2, 0, 0, 27, 1, 0, 0, 0, 99),
    #     ("Question to Statement", "None (Gemini)", 14, 1, 0, 0, 85, 21, 20, 0, 5, 54),
    #     ("Question to Statement", "Top-4 (Gemini)", 10, 1, 0, 2, 87, 28, 37, 0, 3, 32),
    #     ("Question to Statement", "Top-4 (MiniCheck)", 3, 5, 0, 1, 91, 35, 35, 0, 3, 27),
    #     ("Question to Statement", "All (Gemini)", 8, 1, 0, 1, 90, 29, 29, 0, 5, 37),
    #     ("Question to Statement", "All (MiniCheck)", 91, 2, 0, 0, 7, 38, 37, 0, 2, 23),
    #     ("Self-Dual-Critique", "None", 46, 0, 0, 0, 54, 10, 16, 0, 17, 57),
    #     ("Self-Dual-Critique", "Top-4", 34, 3, 1, 2, 60, 26, 13, 0, 10, 51),
    #     ("Self-Dual-Critique", "All", 31, 0, 0, 3, 66, 23, 12, 0, 4, 61),
    #     ("Self-Dual-Critique", "Web", 48, 3, 1, 0, 48, 18, 7, 0, 6, 69),
    # ]
    # cols = [
    #     "Method", "RAG",
    #     "FPQ_S1", "FPQ_S2", "FPQ_S3", "FPQ_S4", "FPQ_S5",
    #     "TPQ_S1", "TPQ_S2", "TPQ_S3", "TPQ_S4", "TPQ_S5",
    # ]
    # df = pd.DataFrame(rows, columns=cols)
    # table_tex = df_to_latex_table(df, "Gemini", "Cancer-Myth")
    # print(table_tex)


def df_to_latex_table(df, model, dataset, dataset_label=None):
    if model.endswith("-preview"):
        model = model.removesuffix("-preview")
    subset_names = ["FPQ", "TPQ"]

    # Print table header
    col_spec = "l l " + " ".join(["c"] * 10)

    lines = ["\\begin{table}[ht]", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
             f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}", "\\toprule"]

    mcol_cells = ["", ""] + [f"\\multicolumn{{{5}}}{{c}}{{\\textbf{{{name}}}}}" for name in subset_names]
    lines.append(" & ".join(mcol_cells) + " \\\\")
    cmidrules = [f"\\cmidrule(lr){{3-7}}", f"\\cmidrule(lr){{8-12}}"]
    lines.append(" ".join(cmidrules))

    col_names = [f"\\textbf{{Method}}", f"\\textbf{{RAG}}"] + \
                   [f"\\textbf{{S}}$_\\mathbf{{{i}}}$" for i in range(1, 6)] * 2
    lines.append(" & ".join(col_names) + " \\\\")
    lines.append("\\midrule")

    # Change every first occurrence of a method to a multirow and every other occurrence to an empty string
    new_df = df.copy()

    for method in df["Method"].unique():
        n_rows = (df["Method"] == method).sum()
        new_df.loc[(df["Method"] == method).idxmax(), "Method"] = f"\\multirow{{{n_rows}}}{{1.5cm}}{{{method}}}"
        new_df.loc[new_df["Method"] == method, "Method"] = ""

    # Add the cell colours
    score_cols = [f"FPQ_S{score}" for score in range(1, 6)] + [f"TPQ_S{score}" for score in range(1, 6)]

    for col in score_cols:
        new_df[col] = new_df[col].apply(lambda x: f"\\scv{{{x}}}")

    # Print table body (keep only the rows corresponding to data and remove headers and footers)
    latex_table = new_df.dropna().to_latex(index=False).split("\n")[4:]

    # Add midrule between different methods
    latex_table_lines = []
    for i, line in enumerate(latex_table):
        if "multirow" in line and i > 0 and "\\midrule" not in latex_table[i-1]:
            latex_table_lines.append("\\midrule")
        latex_table_lines.append(line)
    lines += latex_table_lines

    caption = f"Full results for \\textsc{{{model}}} on {dataset}."
    lines.append(f"\\caption{{{caption}}}")

    label = f"tab:{dataset_label or dataset}-{model}"
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()

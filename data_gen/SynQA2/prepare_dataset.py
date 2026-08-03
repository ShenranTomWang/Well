import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.scraping_utils import get_wikipedia_passages


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = SCRIPT_DIR / "syn_qa2_singlehop.csv"
DEFAULT_FEW_SHOT_FILE = SCRIPT_DIR / "few_shot_data.json"
DEFAULT_PRESUPPOSITION_TEMPLATE_FILE = SCRIPT_DIR / "relation2presupposition_template.json"

SPLIT_CONFIG = {
    "TPQ": {
        "question_key": "original_question",
        "yes_no_question_key": "original_yes_no_question",
        "subject_key": "x",
    },
    "FPQ": {
        "question_key": "perturbed_question",
        "yes_no_question_key": "perturbed_yes_no_question",
        "subject_key": "perturbed_x",
    },
}


def get_entity_passages(entity: str, cache: dict[str, list[str]]) -> list[str]:
    if entity not in cache:
        try:
            cache[entity] = get_wikipedia_passages(entity, cache=cache)
        except Exception as err:
            print(f"{entity}: scrape error: {err}", flush=True)
            cache[entity] = []
    return cache[entity]


def build_datapoint(
    row: dict[str, str],
    idx: int,
    split: str,
    passages: list[str],
    few_shot_data: list[dict[str, Any]],
    presupposition_templates: dict[str, str],
) -> dict[str, Any]:
    config = SPLIT_CONFIG[split]
    relation = row["wikidata_property"]
    if relation not in presupposition_templates:
        raise KeyError(
            f"No presupposition template found for relation: {relation}")

    dp = {
        "id": f"{split}-{idx}",
        "question": row[config["question_key"]],
        "yes_no_question": row[config["yes_no_question_key"]],
        "answer": "",
        "passages": passages,
        "few_shot_data": few_shot_data,
        "source_split": split,
        **row,
    }
    dp["presuppositions"] = [
        presupposition_templates[relation].format(
            x=row[config["subject_key"]],
            y=row["y"],
        )
    ]
    return dp


def write_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for dp in data:
            f.write(json.dumps(dp, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_splits(
    data: list[dict[str, Any]],
    out_dir: Path,
    dev_size: int,
    test_size: int,
    seed: int,
) -> None:
    data = list(data)
    random.Random(seed).shuffle(data)
    dev_data = data[:dev_size]
    test_data = data[dev_size: dev_size + test_size]
    train_data = data[dev_size + test_size:]

    write_jsonl(train_data, out_dir / "train.jsonl")
    write_jsonl(dev_data, out_dir / "dev.jsonl")
    write_jsonl(test_data, out_dir / "test.jsonl")
    print(
        f"Wrote {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test rows to {out_dir}",
        flush=True,
    )


def run_amend(args: argparse.Namespace):
    rng = random.Random(args.seed)
    target_dir = Path(args.target_dir)

    with Path(args.input_file).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with Path(args.few_shot_path).open("r", encoding="utf-8") as f:
        few_shot_data = json.load(f)
    with DEFAULT_PRESUPPOSITION_TEMPLATE_FILE.open("r", encoding="utf-8") as f:
        presupposition_templates = json.load(f)

    few_shot_ids = {str(dp["id"]) for dp in few_shot_data}

    passage_cache: dict[str, list[str]] = {}
    source_data = {"FPQ": {}, "TPQ": {}}
    for i, row in enumerate(rows):
        for split in SPLIT_CONFIG:
            source_data[split][f"{split}-{i}"] = (row, i)

    def repair_split_dir(split: str, split_dir: Path) -> None:
        subject_key = SPLIT_CONFIG[split]["subject_key"]
        loaded_splits = {
            "dev": load_jsonl(split_dir / "dev.jsonl"),
            "test": load_jsonl(split_dir / "test.jsonl"),
            "train": load_jsonl(split_dir / "train.jsonl"),
        }
        split_data = {name: [] for name in loaded_splits}
        seen_ids = set()
        removed_few_shot = 0
        removed_duplicates = 0
        for name, data in loaded_splits.items():
            for dp in data:
                dp_id = str(dp["id"])
                if dp_id in few_shot_ids:
                    removed_few_shot += 1
                    continue
                if dp_id in seen_ids:
                    removed_duplicates += 1
                    continue
                seen_ids.add(dp_id)
                dp["few_shot_data"] = few_shot_data
                split_data[name].append(dp)

        train_data = split_data["train"]
        dev_data = split_data["dev"]
        test_data = split_data["test"]

        if len(dev_data) > args.dev_size:
            train_data.extend(dev_data[args.dev_size:])
            dev_data = dev_data[: args.dev_size]
        if len(test_data) > args.test_size:
            train_data.extend(test_data[args.test_size:])
            test_data = test_data[: args.test_size]

        for dp in train_data + dev_data + test_data:
            if "passages" in dp:
                if subject_key in dp:
                    passage_cache.setdefault(dp[subject_key], dp["passages"])
                if "y" in dp:
                    passage_cache.setdefault(dp["y"], dp["passages"])

        assigned_ids = {str(dp["id"])
                        for dp in train_data + dev_data + test_data}
        missing_ids = [
            dp_id
            for dp_id in sorted(source_data[split], key=lambda value: int(value.split("-", 1)[1]))
            if dp_id not in assigned_ids and dp_id not in few_shot_ids
        ]
        rng.shuffle(missing_ids)

        missing_data = []
        for dp_id in missing_ids:
            row, idx = source_data[split][dp_id]
            passages = get_entity_passages(row[subject_key], passage_cache)
            if "y" in row:
                passages += get_entity_passages(row["y"], passage_cache)
            missing_data.append(
                build_datapoint(
                    row=row,
                    idx=idx,
                    split=split,
                    passages=passages,
                    few_shot_data=few_shot_data,
                    presupposition_templates=presupposition_templates,
                )
            )
            time.sleep(0.2)

        while len(dev_data) < args.dev_size and missing_data:
            dev_data.append(missing_data.pop())
        while len(test_data) < args.test_size and missing_data:
            test_data.append(missing_data.pop())
        train_data.extend(missing_data)

        rng.shuffle(train_data)
        while len(dev_data) < args.dev_size and train_data:
            dev_data.append(train_data.pop())
        while len(test_data) < args.test_size and train_data:
            test_data.append(train_data.pop())

        if len(dev_data) != args.dev_size or len(test_data) != args.test_size:
            raise ValueError(
                f"Could not satisfy {split} target sizes: dev={len(dev_data)}/{args.dev_size}, "
                f"test={len(test_data)}/{args.test_size}, train={len(train_data)}"
            )

        write_jsonl(train_data, split_dir / "train.jsonl")
        write_jsonl(dev_data, split_dir / "dev.jsonl")
        write_jsonl(test_data, split_dir / "test.jsonl")
        print(
            f"Wrote {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test rows to {split_dir}; "
            f"removed {removed_few_shot} few-shot overlaps and {removed_duplicates} duplicate split rows.",
            flush=True,
        )

    repair_split_dir("FPQ", target_dir / "SynQA2FPQ")
    repair_split_dir("TPQ", target_dir / "SynQA2TPQ")


def run_preprocess(args: argparse.Namespace) -> None:
    with Path(args.input_file).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with Path(args.few_shot_path).open("r", encoding="utf-8") as f:
        few_shot_data = json.load(f)
    with DEFAULT_PRESUPPOSITION_TEMPLATE_FILE.open("r", encoding="utf-8") as f:
        presupposition_templates = json.load(f)

    few_shot_ids = {str(dp["id"]) for dp in few_shot_data}

    passage_cache: dict[str, list[str]] = {}
    fpq_data = []
    tpq_data = []

    for i, row in enumerate(rows):
        y_passages = get_entity_passages(row["y"], passage_cache)
        tpq_passages = get_entity_passages(
            row["x"], passage_cache) + y_passages
        fpq_passages = get_entity_passages(
            row["perturbed_x"], passage_cache) + y_passages

        tpq_dp = build_datapoint(
            row=row,
            idx=i,
            split="TPQ",
            passages=tpq_passages,
            few_shot_data=few_shot_data,
            presupposition_templates=presupposition_templates,
        )
        fpq_dp = build_datapoint(
            row=row,
            idx=i,
            split="FPQ",
            passages=fpq_passages,
            few_shot_data=few_shot_data,
            presupposition_templates=presupposition_templates,
        )
        if tpq_dp["id"] not in few_shot_ids:
            tpq_data.append(tpq_dp)
        if fpq_dp["id"] not in few_shot_ids:
            fpq_data.append(fpq_dp)

        print(
            f"[{i + 1}/{len(rows)}] TPQ {row['x']}: {len(tpq_passages)} passages; "
            f"FPQ {row['perturbed_x']}: {len(fpq_passages)} passages",
            flush=True,
        )
        time.sleep(args.sleep)

    out_dir = Path(args.out_dir)
    write_splits(fpq_data, out_dir / "SynQA2FPQ",
                 args.dev_size, args.test_size, args.seed)
    write_splits(tpq_data, out_dir / "SynQA2TPQ",
                 args.dev_size, args.test_size, args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare or amend SynQA2 FPQ/TPQ datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate SynQA2 FPQ/TPQ split files.")
    generate_parser.add_argument("--out_dir", type=str, required=True, help="Output directory containing SynQA2FPQ/ and SynQA2TPQ/.")
    generate_parser.add_argument("--input_file", type=str, default=str(DEFAULT_INPUT_FILE), help="SynQA2 single-hop CSV file.")
    generate_parser.add_argument("--sleep", type=float, default=0.2, help="Sleep time between rows.")
    generate_parser.add_argument("--few_shot_path", type=str, default=str(DEFAULT_FEW_SHOT_FILE), help="Path to SynQA2 few-shot examples. These IDs are excluded from output splits.")
    generate_parser.add_argument("--dev_size", type=int, default=300, help="Number of examples to write to dev.jsonl.")
    generate_parser.add_argument("--test_size", type=int, default=300, help="Number of examples to write to test.jsonl.")
    generate_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")

    amend_parser = subparsers.add_parser("amend", help="Amend train/dev/test JSONL splits in a target directory.")
    amend_parser.add_argument("--target_dir", type=Path, required=True, help="Directory containing dirs SynQA2FPQ/ and SynQA2TPQ/.")
    amend_parser.add_argument("--dev_size", type=int, default=300, help="Target dev.jsonl size.")
    amend_parser.add_argument("--test_size", type=int, default=300, help="Target test.jsonl size.")
    amend_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic amendment.")
    amend_parser.add_argument("--input_file", type=str, default=str(DEFAULT_INPUT_FILE), help="SynQA2 single-hop CSV file.")
    amend_parser.add_argument("--few_shot_path", type=str, default=str(DEFAULT_FEW_SHOT_FILE), help="Path to SynQA2 few-shot examples. These IDs are excluded from output splits.")
    args = parser.parse_args()
    if args.command == "amend":
        run_amend(args)
    else:
        run_preprocess(args)

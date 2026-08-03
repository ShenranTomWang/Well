import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.scraping_utils import get_url_passages


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILES = (
    SCRIPT_DIR / "QAQA_adaptation_set_Dec2022.csv",
    SCRIPT_DIR / "QAQA_evaluation_set_Dec2022.csv",
)
DEFAULT_FEW_SHOT_FILE = SCRIPT_DIR / "few_shot_data.json"


def split_urls(url_field: str) -> List[str]:
    urls = []
    for url in re.split(r"\s*\|\|\s*|\s+/\s+", url_field):
        url = url.strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        urls.append(url)
    return urls


def load_csv(path: Path, split_name: str) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_split"] = split_name
    return rows


def load_few_shot_data(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return json.load(f)


def scrape_passages(urls: Iterable[str], cache: Dict[str, List[str]]) -> List[str]:
    passages = []
    for url in urls:
        if url not in cache:
            try:
                cache[url] = get_url_passages(url)
            except Exception as err:
                print(f"{url}: scrape error: {err}", flush=True)
                cache[url] = []
        passages.extend(cache[url])
    return passages


def build_datapoint(row: Dict[str, str], merged_idx: int, passages: List[str], few_shot_data: List[Dict]) -> Dict:
    question_assumption = row.get("questionable_assumption", "").strip()
    dp = {
        "id": str(merged_idx),
        "answer": row["abstractive_answer"],
        "passages": passages,
        "few_shot_data": few_shot_data,
        "source_urls": split_urls(row["url_extractive_evidence_or_answer"]),
        **row,
    }
    if question_assumption:
        dp["presuppositions"] = [question_assumption]
    else:
        dp["presuppositions"] = []
    return dp


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(data: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for dp in data:
            f.write(json.dumps(dp, ensure_ascii=False) + "\n")


def write_splits(data: List[Dict], out_dir: Path, dev_size: int, seed: int) -> None:
    adaptation_data = [dp for dp in data if dp["source_split"] == "adaptation"]
    test_data = [dp for dp in data if dp["source_split"] == "evaluation"]
    random.Random(seed).shuffle(adaptation_data)
    dev_data = adaptation_data[:dev_size]
    train_data = adaptation_data[dev_size:]

    write_jsonl(train_data, out_dir / "train.jsonl")
    write_jsonl(dev_data, out_dir / "dev.jsonl")
    write_jsonl(test_data, out_dir / "test.jsonl")
    print(
        f"Wrote {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test rows to {out_dir}",
        flush=True,
    )


def run_preprocess(args: argparse.Namespace) -> None:
    input_files = [SCRIPT_DIR / "QAQA_adaptation_set_Dec2022.csv", SCRIPT_DIR / "QAQA_evaluation_set_Dec2022.csv"]
    split_names = ["adaptation", "evaluation"]
    data_by_split = {}
    for path, split_name in zip(input_files, split_names):
        data_by_split[split_name] = load_csv(path, split_name)
    data = data_by_split["adaptation"] + data_by_split["evaluation"]

    few_shot_data = load_few_shot_data(SCRIPT_DIR / "few_shot_data.json")
    few_shot_ids = {str(dp["id"]) for dp in few_shot_data}

    url_cache: Dict[str, List[str]] = {}
    fpq_data = []
    tpq_data = []
    for i, row in enumerate(data):
        urls = split_urls(row["url_extractive_evidence_or_answer"])
        passages = scrape_passages(urls, url_cache)

        if row["all_assumptions_valid"] == "has_invalid":
            dp = build_datapoint(row, i, passages, few_shot_data)
            if dp["id"] not in few_shot_ids:
                fpq_data.append(dp)
            split_name = "QA2FPQ"
        else:
            dp = build_datapoint(row, i, passages, few_shot_data)
            if dp["id"] not in few_shot_ids:
                tpq_data.append(dp)
            split_name = "QA2TPQ"

        print(f"[{i + 1}/{len(data)}] {split_name} {dp['id']}: {len(passages)} passages", flush=True)
        time.sleep(args.sleep)

    out_dir = Path(args.out_dir)
    write_splits(fpq_data, out_dir / "QA2FPQ", args.dev_size, args.seed)
    write_splits(tpq_data, out_dir / "QA2TPQ", args.dev_size, args.seed)


def run_amend(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    target_dir = Path(args.target_dir)
    input_files = [SCRIPT_DIR / "QAQA_adaptation_set_Dec2022.csv", SCRIPT_DIR / "QAQA_evaluation_set_Dec2022.csv"]
    split_names = ["adaptation", "evaluation"]
    data_by_split = {}
    for path, split_name in zip(input_files, split_names):
        data_by_split[split_name] = load_csv(path, split_name)
    rows = data_by_split["adaptation"] + data_by_split["evaluation"]

    few_shot_data = load_few_shot_data(SCRIPT_DIR / "few_shot_data.json")
    few_shot_ids = {str(dp["id"]) for dp in few_shot_data}

    source_data = {"FPQ": {}, "TPQ": {}}
    for i, row in enumerate(rows):
        split = "FPQ" if row["all_assumptions_valid"] == "has_invalid" else "TPQ"
        source_data[split][str(i)] = (row, i)

    url_cache: Dict[str, List[str]] = {}

    def repair_split_dir(split: str, split_dir: Path) -> None:
        loaded_splits = {
            "dev": load_jsonl(split_dir / "dev.jsonl"),
            "test": load_jsonl(split_dir / "test.jsonl"),
            "train": load_jsonl(split_dir / "train.jsonl"),
        }
        split_data = {name: [] for name in loaded_splits}
        seen_ids = set()
        removed_few_shot = 0
        removed_duplicates = 0
        removed_unknown = 0
        for name, data in loaded_splits.items():
            for dp in data:
                dp_id = str(dp["id"])
                if dp_id in few_shot_ids:
                    removed_few_shot += 1
                    continue
                if dp_id not in source_data[split]:
                    removed_unknown += 1
                    continue
                if dp_id in seen_ids:
                    removed_duplicates += 1
                    continue
                seen_ids.add(dp_id)
                dp["few_shot_data"] = few_shot_data
                split_data[name].append(dp)

        existing_data = split_data["dev"] + split_data["test"] + split_data["train"]
        for dp in existing_data:
            source_urls = dp.get("source_urls", [])
            if len(source_urls) == 1 and "passages" in dp:
                url_cache.setdefault(source_urls[0], dp["passages"])

        assigned_ids = {str(dp["id"]) for dp in existing_data}
        missing_ids = [
            dp_id
            for dp_id in sorted(source_data[split], key=int)
            if dp_id not in assigned_ids and dp_id not in few_shot_ids
        ]
        rng.shuffle(missing_ids)

        missing_data = []
        for dp_id in missing_ids:
            row, idx = source_data[split][dp_id]
            urls = split_urls(row["url_extractive_evidence_or_answer"])
            passages = scrape_passages(urls, url_cache)
            missing_data.append(build_datapoint(row, idx, passages, few_shot_data))
            time.sleep(0.2)

        dev_candidates = [dp for dp in split_data["dev"] if dp["source_split"] == "adaptation"]
        train_candidates = [
            dp
            for dp in split_data["train"] + split_data["test"] + split_data["dev"]
            if dp["source_split"] == "adaptation" and str(dp["id"]) not in {str(item["id"]) for item in dev_candidates}
        ]
        test_data = [dp for dp in existing_data if dp["source_split"] == "evaluation"]

        missing_adaptation = [dp for dp in missing_data if dp["source_split"] == "adaptation"]
        missing_evaluation = [dp for dp in missing_data if dp["source_split"] == "evaluation"]
        test_data.extend(missing_evaluation)

        if len(dev_candidates) > args.dev_size:
            train_candidates.extend(dev_candidates[args.dev_size :])
            dev_candidates = dev_candidates[: args.dev_size]

        while len(dev_candidates) < args.dev_size and missing_adaptation:
            dev_candidates.append(missing_adaptation.pop())
        train_candidates.extend(missing_adaptation)

        rng.shuffle(train_candidates)
        while len(dev_candidates) < args.dev_size and train_candidates:
            dev_candidates.append(train_candidates.pop())

        if len(dev_candidates) != args.dev_size:
            raise ValueError(
                f"Could not satisfy {split} target dev size: dev={len(dev_candidates)}/{args.dev_size}, "
                f"train={len(train_candidates)}, test={len(test_data)}"
            )

        write_jsonl(train_candidates, split_dir / "train.jsonl")
        write_jsonl(dev_candidates, split_dir / "dev.jsonl")
        write_jsonl(test_data, split_dir / "test.jsonl")
        print(
            f"Wrote {len(train_candidates)} train, {len(dev_candidates)} dev, {len(test_data)} test rows to {split_dir}; "
            f"removed {removed_few_shot} few-shot overlaps, {removed_duplicates} duplicate split rows, "
            f"and {removed_unknown} rows outside {split}.",
            flush=True,
        )

    repair_split_dir("FPQ", target_dir / "QA2FPQ")
    repair_split_dir("TPQ", target_dir / "QA2TPQ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare or amend QA2 FPQ/TPQ datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="Generate QA2 FPQ/TPQ split files.")
    generate_parser.add_argument("--out_dir", type=str, required=True, help="Output directory containing QA2FPQ/ and QA2TPQ/.")
    generate_parser.add_argument("--sleep", type=float, default=0.2, help="Sleep time between datapoints.")
    generate_parser.add_argument("--dev_size", type=int, default=6, help="Number of adaptation examples to write to dev.jsonl.")
    generate_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    
    amend_parser = subparsers.add_parser("amend", help="Amend QA2 train/dev/test JSONL splits by source_split.")
    amend_parser.add_argument("--target_dir", type=Path, required=True, help="Directory containing QA2FPQ/ and QA2TPQ/.")
    amend_parser.add_argument("--dev_size", type=int, default=6, help="Number of adaptation examples to write to dev.jsonl.")
    amend_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic amendment.")
    amend_parser.set_defaults(command="amend")
    args = parser.parse_args()
    if args.command == "amend":
        run_amend(args)
    else:
        run_preprocess(args)

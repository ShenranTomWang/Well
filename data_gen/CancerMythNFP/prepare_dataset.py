import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.scraping_utils import get_wikipedia_passages

SCRIPT_DIR = Path(__file__).resolve().parent

def get_cancer_passages(cancer: str, cache: dict[str, list[str]]) -> list[str]:
    if cancer not in cache:
        try:
            cache[cancer] = get_wikipedia_passages(cancer, cache=cache)
        except Exception as err:
            print(f"{cancer}: scrape error: {err}", flush=True)
            cache[cancer] = []
    return cache[cancer]


def write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_amend(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    target_dir = Path(args.target_dir)
    with (SCRIPT_DIR / "nfp.json").open("r") as f:
        source_data = json.load(f)
    with (SCRIPT_DIR / "few_shot_data.json").open("r") as f:
        few_shot_data = json.load(f)
    few_shot_ids = {str(dp["id"]) for dp in few_shot_data}

    source_by_id = {str(dp["QID"]): dp for dp in source_data}
    dev_data = []
    test_data = []
    seen_ids = set()
    removed_few_shot = 0
    removed_duplicates = 0
    for split_name, split_data in (
        ("dev", load_jsonl(target_dir / "dev.jsonl")),
        ("test", load_jsonl(target_dir / "test.jsonl")),
    ):
        out_data = dev_data if split_name == "dev" else test_data
        for dp in split_data:
            dp_id = str(dp["id"])
            if dp_id in few_shot_ids:
                removed_few_shot += 1
                continue
            if dp_id in seen_ids:
                removed_duplicates += 1
                continue
            seen_ids.add(dp_id)
            dp["few_shot_data"] = few_shot_data
            out_data.append(dp)

    if len(dev_data) > args.dev_size:
        test_data.extend(dev_data[args.dev_size :])
        dev_data = dev_data[: args.dev_size]

    assigned_ids = {str(dp["id"]) for dp in dev_data + test_data}
    missing_ids = [
        dp_id
        for dp_id in sorted(source_by_id, key=int)
        if dp_id not in assigned_ids and dp_id not in few_shot_ids
    ]
    rng.shuffle(missing_ids)

    cancer_cache = {}
    for dp in dev_data + test_data:
        if "cancer" in dp and "passages" in dp:
            cancer_cache.setdefault(dp["cancer"], dp["passages"])

    missing_data = []
    for dp_id in missing_ids:
        dp = dict(source_by_id[dp_id])
        dp["question"] = dp.pop("example_question")
        dp["cancer"] = dp.pop("source_cancer")
        dp["id"] = dp.pop("QID")
        dp["few_shot_data"] = few_shot_data
        dp["example_presuppositions"] = [dp.pop("example_assumption")]
        dp["passages"] = get_cancer_passages(dp["cancer"], cancer_cache)
        missing_data.append(dp)
        time.sleep(0.2)

    while len(dev_data) < args.dev_size and missing_data:
        dev_data.append(missing_data.pop())
    test_data.extend(missing_data)

    rng.shuffle(test_data)
    while len(dev_data) < args.dev_size and test_data:
        dev_data.append(test_data.pop())

    if len(dev_data) != args.dev_size:
        raise ValueError(f"Could not satisfy target dev size: dev={len(dev_data)}/{args.dev_size}")

    write_jsonl(target_dir / "dev.jsonl", dev_data)
    write_jsonl(target_dir / "test.jsonl", test_data)
    print(
        f"Wrote {len(dev_data)} dev and {len(test_data)} test rows to {target_dir}; "
        f"removed {removed_few_shot} few-shot overlaps and {removed_duplicates} duplicate split rows.",
        flush=True,
    )


def run_preprocess(args: argparse.Namespace):
    with (SCRIPT_DIR / "nfp.json").open("r") as f:
        data = json.load(f)
    with (SCRIPT_DIR / "few_shot_data.json").open("r") as f:
        few_shot_data = json.load(f)

    cancer_cache: dict[str, list[str]] = {}
    for i, dp in enumerate(data):
        dp["question"] = dp.pop("example_question")
        dp["cancer"] = dp.pop("source_cancer")
        dp["id"] = dp.pop("QID")
        dp["few_shot_data"] = few_shot_data
        dp["example_presuppositions"] = [dp.pop("example_assumption")]
        dp["passages"] = get_cancer_passages(dp["cancer"], cancer_cache)
        print(f"[{i + 1}/{len(data)}] {dp['cancer']}: {len(dp['passages'])} passages", flush=True)
        time.sleep(args.sleep)

    random.shuffle(data)
    dev_data = data[: args.dev_size]
    test_data = data[args.dev_size :]
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "dev.jsonl", dev_data)
    write_jsonl(out_dir / "test.jsonl", test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess or amend data for CancerMythNFP dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    generate_parser = subparsers.add_parser("generate", help="Generate CancerMythNFP split files.")
    generate_parser.add_argument("--out_dir", type=str, required=True, help="Output directory for dataset (.jsonl)")
    generate_parser.add_argument("--dev_size", type=int, default=30, help="Number of samples to use for dev set")
    generate_parser.add_argument("--sleep", type=float, default=0.2, help="Sleep time after scraping a new cancer page")
    
    amend_parser = subparsers.add_parser("amend", help="Amend dev/test JSONL splits in a target directory.")
    amend_parser.add_argument("--target_dir", type=Path, required=True, help="Directory containing dev.jsonl and test.jsonl.")
    amend_parser.add_argument("--dev_size", dest="dev_size", type=int, default=30, help="Target dev.jsonl size.")
    amend_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic amendment.")
    args = parser.parse_args()
    
    if args.command == "amend":
        run_amend(args)
    else:
        run_preprocess(args)

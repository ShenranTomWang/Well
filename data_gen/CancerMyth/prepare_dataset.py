import argparse, os, random, json, sys, time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.scraping_utils import get_wikipedia_passages, get_url_passages
import datasets

SCRIPT_DIR = Path(__file__).resolve().parent

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for dp in data:
            f.write(json.dumps(dp, ensure_ascii=False) + "\n")


def run_amend(args: argparse.Namespace):
    rng = random.Random(args.seed)
    target_dir = Path(args.target_dir)
    with open(SCRIPT_DIR / "few_shot_data.json", "r") as f:
        few_shot_data = json.load(f)
    few_shot_ids = {str(dp["id"]) for dp in few_shot_data}

    loaded_splits = {
        "dev": load_jsonl(target_dir / "dev.jsonl"),
        "test": load_jsonl(target_dir / "test.jsonl"),
        "train": load_jsonl(target_dir / "train.jsonl"),
    }
    split_data = {split: [] for split in loaded_splits}
    seen_ids = set()
    removed_few_shot = 0
    removed_duplicates = 0
    for split, data in loaded_splits.items():
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
            split_data[split].append(dp)

    train_data = split_data["train"]
    dev_data = split_data["dev"]
    test_data = split_data["test"]

    if len(dev_data) > args.dev_size:
        train_data.extend(dev_data[args.dev_size :])
        dev_data = dev_data[: args.dev_size]
    if len(test_data) > args.test_size:
        train_data.extend(test_data[args.test_size :])
        test_data = test_data[: args.test_size]

    dataset = datasets.load_dataset("Cancer-Myth/Cancer-Myth", split="validation")
    all_ids = {str(i) for i, _ in enumerate(dataset)}
    assigned_ids = {str(dp["id"]) for dp in train_data + dev_data + test_data}
    missing_ids = [dp_id for dp_id in sorted(all_ids, key=int) if dp_id not in assigned_ids and dp_id not in few_shot_ids]
    rng.shuffle(missing_ids)

    passage_cache = {}
    for dp in train_data + dev_data + test_data:
        if "cancer" in dp and "passages" in dp:
            passage_cache.setdefault(dp["cancer"], dp["passages"])

    missing_data = []
    for dp_id in missing_ids:
        item = dataset[int(dp_id)]
        passages = passage_cache.get(item["cancer"])
        if passages is None:
            passages = get_wikipedia_passages(item["cancer"], cache=passage_cache)
            passages += get_url_passages(item["source"])
            passage_cache[item["cancer"]] = passages
            time.sleep(0.2)
        dp = {
            "id": dp_id,
            "answer": item["presupposition_correction"],
            "presuppositions": [item["source_myth"]],
            "passages": passages,
            "few_shot_data": few_shot_data,
            **item,
        }
        dp.pop("presupposition_correction")
        dp.pop("source_myth")
        missing_data.append(dp)

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
            f"Could not satisfy target sizes: dev={len(dev_data)}/{args.dev_size}, "
            f"test={len(test_data)}/{args.test_size}, train={len(train_data)}"
        )

    write_jsonl(target_dir / "train.jsonl", train_data)
    write_jsonl(target_dir / "dev.jsonl", dev_data)
    write_jsonl(target_dir / "test.jsonl", test_data)
    print(
        f"Wrote {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test rows to {target_dir}; "
        f"removed {removed_few_shot} few-shot overlaps and {removed_duplicates} duplicate split rows.",
        flush=True,
    )

def run_preprocess(args: argparse.Namespace):
    dataset = datasets.load_dataset("Cancer-Myth/Cancer-Myth", split="validation")
    data = []
    with open(SCRIPT_DIR / "few_shot_data.json", "r") as f:
        few_shot_data = json.load(f)
    few_shot_ids = {str(dp["id"]) for dp in few_shot_data}
    cache = {}
    for i, item in enumerate(dataset):
        dp_id = str(i)
        if dp_id in few_shot_ids:
            continue
        passages = cache.get(item["cancer"], None)
        if not passages:
            passages = get_wikipedia_passages(item["cancer"], cache=cache)
            passages += get_url_passages(item["source"])
            cache[item["cancer"]] = passages
        dp = {
            "id": dp_id,
            "answer": item["presupposition_correction"],
            "presuppositions": [item["source_myth"]],
            "passages": passages,
            **item
        }
        dp.pop("presupposition_correction")
        dp.pop("source_myth")
        data.append(dp)
        time.sleep(0.2)
    random.shuffle(data)
    data = [{**dp, "few_shot_data": few_shot_data} for dp in data]
    val_data = data[: args.val_size]
    test_data = data[args.val_size : args.val_size + args.test_size]
    train_data = data[args.val_size + args.test_size :]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(Path(args.out_dir) / "dev.jsonl", "w") as f:
        for dp in val_data:
            f.write(json.dumps(dp) + "\n")
    with open(Path(args.out_dir) / "test.jsonl", "w") as f:
        for dp in test_data:
            f.write(json.dumps(dp) + "\n")
    with open(Path(args.out_dir) / "train.jsonl", "w") as f:
        for dp in train_data:
            f.write(json.dumps(dp) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or amend Cancer-Myth dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="Generate Cancer-Myth split files.")
    generate_parser.add_argument("--out_dir", type=str, required=True, help="The directory to save the downloaded dataset.")
    generate_parser.add_argument("--val_size", type=int, default=100, help="The number of validation samples to save.")
    generate_parser.add_argument("--test_size", type=int, default=100, help="The number of test samples to save.")
    
    amend_parser = subparsers.add_parser("amend", help="Amend train/dev/test JSONL splits in a target directory.")
    amend_parser.add_argument("--target_dir", type=Path, required=True, help="Directory containing train.jsonl, dev.jsonl, and test.jsonl.")
    amend_parser.add_argument("--dev_size", type=int, default=100, help="Target dev.jsonl size.",)
    amend_parser.add_argument("--test_size", type=int, default=100, help="Target test.jsonl size.",)
    amend_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic amendment.")
    args = parser.parse_args()
    if args.command == "amend":
        run_amend(args)
    else:
        run_preprocess(args)

"""Split CREPE into false- and true-presupposition question datasets."""

import argparse
import json
from pathlib import Path


LABEL_TO_DATASET = {
    "false presupposition": "CREPEFPQ",
    "normal": "CREPETPQ",
}
SPLIT_ALIASES = {
    "train": ("train.json", "train.jsonl"),
    "val": ("val.json", "val.jsonl", "dev.json", "dev.jsonl"),
    "test": ("test.json", "test.jsonl"),
}


def load_data(path):
    with path.open() as f:
        content = f.read().strip()
    if not content:
        return []
    if content.lstrip().startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def find_split(input_dir, split):
    for filename in SPLIT_ALIASES[split]:
        path = input_dir / filename
        if path.is_file():
            return path
    expected = ", ".join(SPLIT_ALIASES[split])
    raise FileNotFoundError(f"Could not find {split} split in {input_dir}; expected one of: {expected}")


def prepare_datapoint(datapoint):
    prepared = dict(datapoint)
    if "comment" not in prepared:
        raise KeyError(f"Datapoint {prepared.get('id', '<unknown>')} has no 'comment' field")
    prepared["answer"] = prepared.pop("comment")
    return prepared


def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for datapoint in data:
            f.write(json.dumps(datapoint, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split CREPE into CREPEFPQ and CREPETPQ datasets."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing train, val, and test JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory in which CREPEFPQ and CREPETPQ will be created.",
    )
    args = parser.parse_args()

    datasets = {name: {} for name in LABEL_TO_DATASET.values()}
    for split in SPLIT_ALIASES:
        source_path = find_split(args.input_dir, split)
        split_data = {name: [] for name in LABEL_TO_DATASET.values()}
        for datapoint in load_data(source_path):
            labels = datapoint.get("labels")
            if not isinstance(labels, list) or not labels:
                continue
            unique_labels = set(labels)
            if len(unique_labels) != 1:
                continue
            dataset_name = LABEL_TO_DATASET.get(next(iter(unique_labels)))
            if dataset_name is not None:
                split_data[dataset_name].append(prepare_datapoint(datapoint))
        for dataset_name, data in split_data.items():
            datasets[dataset_name][split] = data

    with open(Path(__file__).resolve().parent / "few_shot.json") as f:
        few_shot = json.load(f)
    for dataset_name in ("CREPEFPQ", "CREPETPQ"):
        train_data = datasets[dataset_name]["train"]
        if len(train_data) < 2:
            raise ValueError(
                f"The train split has only {len(train_data)} eligible {dataset_name} "
                "datapoints; at least 2 are required for few-shot examples."
            )
        datasets[dataset_name]["train"] = [dp for dp in train_data if dp["id"] not in {fs_dp["id"] for fs_dp in few_shot}]

    for dataset_name, splits in datasets.items():
        for split, data in splits.items():
            for datapoint in data:
                datapoint["few_shot_data"] = few_shot
            write_jsonl(args.output_dir / dataset_name / f"{split}.jsonl", data)

    print(f"Saved datasets under {args.output_dir}")
    for dataset_name, splits in datasets.items():
        counts = ", ".join(f"{split}={len(data)}" for split, data in splits.items())
        print(f"{dataset_name}: {counts}")


if __name__ == "__main__":
    main()

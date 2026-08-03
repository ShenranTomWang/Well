import argparse
import json
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    reference_file = Path(args.reference_file)
    target_dir = Path(args.target_dir)

    with reference_file.open(encoding="utf-8") as file:
        allowed_ids = {
            json.loads(line)[args.id_key]
            for line in file
            if line.strip()
        }

    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target directory does not exist: {target_dir}")

    total_kept = 0
    total_removed = 0
    for jsonl_file in sorted(target_dir.rglob("*.jsonl")):
        kept = []
        removed = 0
        with jsonl_file.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                datapoint = json.loads(line)
                if args.id_key not in datapoint:
                    raise KeyError(
                        f"Missing key {args.id_key!r} in {jsonl_file}:{line_number}"
                    )
                if datapoint[args.id_key] in allowed_ids:
                    kept.append(datapoint)
                else:
                    removed += 1

        print(f"{jsonl_file}: kept {len(kept)}, removed {removed}")
        total_kept += len(kept)
        total_removed += removed

        if not args.dry_run:
            temporary_file = jsonl_file.with_suffix(jsonl_file.suffix + ".tmp")
            with temporary_file.open("w", encoding="utf-8") as file:
                for datapoint in kept:
                    file.write(json.dumps(datapoint, ensure_ascii=False) + "\n")
            temporary_file.replace(jsonl_file)

    action = "Would remove" if args.dry_run else "Removed"
    print(f"Done. Kept {total_kept} datapoints. {action} {total_removed} datapoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively filter JSONL files to datapoints in a reference JSONL."
    )
    parser.add_argument(
        "--reference_file",
        required=True,
        help="JSONL file whose IDs define the datapoints to keep.",
    )
    parser.add_argument(
        "--target_dir",
        required=True,
        help="Directory containing JSONL files to filter recursively.",
    )
    parser.add_argument(
        "--id_key",
        default="id",
        help="Datapoint ID field (default: id).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report changes without modifying files.",
    )
    main(parser.parse_args())

import argparse, json, os, tempfile


def iter_jsonl_files(data_dir: str):
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".jsonl"):
                yield os.path.join(root, filename)


def replace_few_shot_data(jsonl_path: str, few_shot_data, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.abspath(jsonl_path) == os.path.abspath(out_path):
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(jsonl_path), suffix=".tmp")
        os.close(fd)
        write_path = tmp_path
    else:
        tmp_path = None
        write_path = out_path

    with open(jsonl_path, "r") as in_file, open(write_path, "w") as out_file:
        for line in in_file:
            if not line.strip():
                continue
            dp = json.loads(line)
            dp["few_shot_data"] = few_shot_data
            out_file.write(json.dumps(dp) + "\n")
    if tmp_path:
        os.replace(tmp_path, jsonl_path)


def main(args: argparse.Namespace):
    with open(args.few_shot_path, "r") as f:
        few_shot_data = json.load(f)

    for jsonl_path in list(iter_jsonl_files(args.data_dir)):
        if args.out_dir:
            rel_path = os.path.relpath(jsonl_path, args.data_dir)
            out_path = os.path.join(args.out_dir, rel_path)
        else:
            out_path = jsonl_path
        replace_few_shot_data(jsonl_path, few_shot_data, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace few_shot_data in every .jsonl file under a directory.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .jsonl files to update.")
    parser.add_argument("--few_shot_path", type=str, required=True, help="Path to the .json file containing few-shot data.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional output directory. If omitted, files are updated in place.")
    args = parser.parse_args()
    main(args)

import argparse, os, json, time
from utils.scraping_utils import get_wikipedia_passages, get_url_passages

def prepare_cancer_cache(data):
    cache = {}
    for dp in data:
        cancer = dp['cancer']
        if 'passages' in dp:
            cache[cancer] = dp['passages']
    return cache

def run_scrape_wikipedia_passages(args: argparse.Namespace):
    with open(args.file, "r") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    if args.out_file is None:
        fnames = args.file.split(".")
        args.out_file = f"{'.'.join(fnames[:-1])}_with_passages.{fnames[-1]}"
    cancer_cache = prepare_cancer_cache(data)
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        for i, dp in enumerate(data):
            if (not args.overwrite) and ("passages" in dp):
                f.write(json.dumps(dp) + "\n")
                continue
            cancer = dp['cancer']
            if cancer in cancer_cache:
                dp["passages"] = cancer_cache[cancer]
                f.write(json.dumps(dp) + "\n")
                print(f"[{i}] {cancer}: cached, {len(dp['passages'])} passages")
                continue

            try:
                passages = get_wikipedia_passages(cancer, cache=cancer_cache)
                passages += get_url_passages(dp['source'])
            except Exception as err:
                print(f"[{i}] {cancer} error: {err}")
                passages = []

            dp["passages"] = passages
            cancer_cache[cancer] = passages
            if len(passages) == 0:
                print(f"[{i}] {cancer}: no passages found")
            f.write(json.dumps(dp) + "\n")
            time.sleep(args.sleep)

def main(args: argparse.Namespace):
    if args.command == "scrape":
        run_scrape_wikipedia_passages(args)
    else:
        raise ValueError(f"Unknown command {args.command}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Wikipedia passages for each datapoint's cancer field.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="Scrape Wikipedia text for each cancer and save it to passages")
    scrape_parser.add_argument("--file", type=str, required=True, help="Input dataset file (.jsonl)")
    scrape_parser.add_argument("--out_file", type=str, default=None, help="Output file path")
    scrape_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing passages field")
    scrape_parser.add_argument("--sleep", type=float, default=0.2, help="Sleep time between requests")

    args = parser.parse_args()
    main(args)
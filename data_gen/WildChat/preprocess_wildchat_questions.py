"""
Preprocess WildChat into JSONL records containing user questions.

Each output line has the form:
    {"question": "...", "conversation_id": "...", "source_index": 0, "turn_index": 0}

Example:
    python preprocess_wildchat_questions.py \
        --output-path /path/to/wildchat_user_questions.jsonl \
        --streaming \
        --max-rows 10000
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def iter_user_questions(example, source_index):
    """Yield user utterances from a WildChat example that contain a question mark."""
    conversation = example.get("conversation") or []
    conversation_id = (
        example.get("conversation_id")
        or example.get("conversation_hash")
        or example.get("id")
    )

    for turn_index, turn in enumerate(conversation):
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "user":
            continue

        content = turn.get("content")
        if not isinstance(content, str):
            continue

        question = content.strip()
        if not question or "?" not in question:
            continue

        record = {
            "question": question,
            "conversation_id": conversation_id,
            "source_index": source_index,
            "turn_index": turn_index,
        }
        yield record


def write_jsonl(records, output_path, max_rows=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            if max_rows is not None and count >= max_rows:
                break
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save WildChat user turns containing '?' as a JSONL file."
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=Path,
        help="Destination .jsonl path.",
    )
    parser.add_argument(
        "--dataset_name",
        default="allenai/WildChat",
        help="Hugging Face dataset name. Defaults to allenai/WildChat.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load. Defaults to train.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset instead of downloading it first.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of WildChat datapoints to process.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional maximum number of filtered JSONL rows to write.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_path.suffix != ".jsonl":
        raise ValueError(f"--output_path must end with .jsonl: {args.output_path}")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError(f"--max_samples must be non-negative: {args.max_samples}")
    if args.max_rows is not None and args.max_rows < 0:
        raise ValueError(f"--max_rows must be non-negative: {args.max_rows}")

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        streaming=args.streaming,
    )

    def records():
        for source_index, example in enumerate(
            tqdm(dataset, desc="Filtering WildChat user questions")
        ):
            if args.max_samples is not None and source_index >= args.max_samples:
                break
            yield from iter_user_questions(example, source_index)

    count = write_jsonl(records(), args.output_path, max_rows=args.max_rows)
    print(f"Wrote {count} user questions to {args.output_path}")


if __name__ == "__main__":
    main()

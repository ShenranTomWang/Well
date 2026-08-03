"""
Identify high-influence movie heads, then knock them out during DirectQA.

This script uses the influence files produced by ``identify_heads.py``.  For a
given model it first ranks the top heads with the same frequency-based logic,
then runs DirectQA while zeroing those heads at the question position during
the forward pass.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
from FAITH.identify_heads import identify_heads
from utils.argparse_utils import add_source_subparsers

from pipeline_operator.direct_qa_operator import KnockOutDirectQAOperator
from pipeline_operator.direct_qa_operator.knock_out_direct_qa_operator import DTYPE_MAP


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def load_or_identify_heads(args: argparse.Namespace) -> Dict[str, Any]:
    heads_dir = Path(args.heads_dir).expanduser().resolve()
    default_summary_file = heads_dir / "identified_false_premise_heads.json"
    summary_file = Path(args.heads_file).expanduser().resolve() if args.heads_file else default_summary_file

    if summary_file.is_file() and not args.refresh_heads:
        with summary_file.open("r") as f:
            return json.load(f)

    influence_dir = heads_dir / "influences" if (heads_dir / "influences").is_dir() else heads_dir
    summary = identify_heads(
        influence_dir=influence_dir,
        top_k_per_sample=args.top_k_per_sample,
        score_threshold=args.score_threshold,
        selected_top_k=args.selected_top_k,
    )
    summary["influence_dir"] = str(influence_dir)
    write_json(summary, summary_file)
    return summary


def selected_head_tuples(summary: Dict[str, Any]) -> List[tuple[int, int]]:
    if "selected_heads_tuples" in summary:
        return [tuple(map(int, head)) for head in summary["selected_heads_tuples"]]
    return [(int(head["layer"]), int(head["head"])) for head in summary["selected_heads"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True, help="Transformers model name or local path.")
    parser.add_argument("--dataset-path", required=True, help="DirectQA dataset file in JSONL format.")
    parser.add_argument("--out-file", required=True, help="Output JSONL path.")
    parser.add_argument("--heads-dir", required=True, help="Directory from identify_heads.py for this model.")
    parser.add_argument("--heads-file", default=None, help="Optional selected-head summary JSON.")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index for cached runs.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts to generate in each batch.")
    parser.add_argument("--disable-few-shot", action="store_true")
    parser.add_argument("--system-role", type=str, default="system")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=tuple(DTYPE_MAP))
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit-input-length", type=int, default=3700)
    parser.add_argument("--refresh-heads", action="store_true", help="Re-rank heads even if a summary JSON exists.")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--top-k-per-sample", type=int, default=20)
    parser.add_argument("--selected-top-k", type=int, default=20)
    parser.add_argument(
        "--position-mode",
        choices=("question_end", "last_prompt_token", "all_prompt_tokens"),
        default="question_end",
        help="Prompt position where the selected heads are zeroed.",
    )
    parser.add_argument(
        "--knockout-generated-tokens",
        action="store_true",
        help="Also zero selected heads on one-token generation steps after the prompt prefill.",
    )
    add_source_subparsers(parser)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    return args


def main(args: argparse.Namespace) -> None:
    dataset = read_jsonl(Path(args.dataset_path).expanduser())
    dataset = dataset[args.start_idx :]
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]
    if args.disable_few_shot:
        for dp in dataset:
            dp["few_shot_data"] = []

    summary = load_or_identify_heads(args)
    heads = selected_head_tuples(summary)
    if not heads:
        raise ValueError(f"No selected heads found in {args.heads_dir}")

    out_file = Path(args.out_file).expanduser()
    if args.start_idx == 0:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("")
    write_json(
        {
            "model_name_or_path": args.model_name_or_path,
            "dataset_path": args.dataset_path,
            "heads_dir": args.heads_dir,
            "selected_heads": [{"layer": layer, "head": head} for layer, head in heads],
            "position_mode": args.position_mode,
            "knockout_generated_tokens": args.knockout_generated_tokens,
            "batch_size": args.batch_size,
        },
        out_file.with_suffix(out_file.suffix + ".metadata.json"),
    )

    operator = KnockOutDirectQAOperator(
        model_name=args.model_name_or_path,
        heads=heads,
        device=args.device,
        dtype=args.dtype,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        limit_input_length=args.limit_input_length,
        position_mode=args.position_mode,
        knockout_generated_tokens=args.knockout_generated_tokens,
    )
    progress = tqdm(total=len(dataset), desc="direct qa knockout")
    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        results = operator.qa_batch(batch, source=args.source_command, **vars(args))
        for result in results:
            append_jsonl(result, out_file)
        progress.update(len(results))
        print(f"Progress: {start + len(results)}/{len(dataset)}")
    progress.close()


if __name__ == "__main__":
    main(parse_args())

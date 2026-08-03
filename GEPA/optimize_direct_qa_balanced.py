import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constant.gepa import DEFAULT_GEPA_REFLECTION_MODEL
from constant.response_level_score import DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL
from data_gen.template import get_template_cls
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from utils.argparse_utils import add_source_subparsers

from GEPA.optimize_direct_qa import (
    DirectQAGEPAEvaluator,
    GeminiReflectionLM,
    disable_few_shot,
    load_jsonl,
    load_seed_prompt,
    save_result,
)


EXAMPLE_TYPE_KEY = "_gepa_question_type"
DATASET_PAIRS = {
    "CancerMyth": {"fpq": "CancerMyth", "tpq": "CancerMythNFP"},
    "QA2": {"fpq": "QA2FPQ", "tpq": "QA2TPQ"},
    "SynQA2": {"fpq": "SynQA2FPQ", "tpq": "SynQA2TPQ"},
    "CREPE": {"fpq": "CREPEFPQ", "tpq": "CREPETPQ"},
}
SCORE_MAX_BY_DATASET = {
    "CancerMyth": 5.0,
    "CancerMythNFP": 5.0,
    "QA2FPQ": 5.0,
    "QA2TPQ": 5.0,
    "SynQA2FPQ": 5.0,
    "SynQA2TPQ": 5.0,
    "CREPEFPQ": 5.0,
    "CREPETPQ": 5.0,
}


def load_tagged_data(
    path: str,
    question_type: str,
    start_idx: int,
    limit: int | None,
) -> List[Dict[str, Any]]:
    data = disable_few_shot(load_jsonl(path, start_idx, limit))
    for dp in data:
        dp[EXAMPLE_TYPE_KEY] = question_type
    return data


def interleave_balanced(
    fpqs: List[Dict[str, Any]],
    tpqs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not fpqs or not tpqs:
        raise ValueError("Both FPQ and TPQ inputs must contain at least one example.")

    size = min(len(fpqs), len(tpqs))
    balanced = []
    for fpq, tpq in zip(fpqs[:size], tpqs[:size]):
        balanced.extend((fpq, tpq))
    return balanced


class BalancedDirectQAGEPAEvaluator(DirectQAGEPAEvaluator):
    def __init__(self, args: argparse.Namespace, backend: str, web_search: bool = False):
        self.web_search = web_search
        self.args = args
        self.backend = backend
        self.template_classes = {
            question_type: get_template_cls(f"{dataset_name}ResponseLevelScoreTemplate")
            for question_type, dataset_name in DATASET_PAIRS[args.dataset_family].items()
        }

    def __call__(
        self,
        candidate: str | Dict[str, str],
        example: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        question_type = example[EXAMPLE_TYPE_KEY]
        self.ScoreTemplateClass = self.template_classes[question_type]
        dataset_name = DATASET_PAIRS[self.args.dataset_family][question_type]
        self.score_max = SCORE_MAX_BY_DATASET[dataset_name]
        score, metadata = super().__call__(candidate, example)
        metadata["question_type"] = question_type
        metadata["score_template"] = self.ScoreTemplateClass.__name__
        return score, metadata


def load_balanced_split(args: argparse.Namespace, split: str) -> List[Dict[str, Any]]:
    fpqs = load_tagged_data(
        getattr(args, f"fpq_{split}_path"),
        "fpq",
        getattr(args, f"{split}_start_idx"),
        getattr(args, f"fpq_{split}_limit"),
    )
    tpqs = load_tagged_data(
        getattr(args, f"tpq_{split}_path"),
        "tpq",
        getattr(args, f"{split}_start_idx"),
        getattr(args, f"tpq_{split}_limit"),
    )
    return interleave_balanced(fpqs, tpqs)


def run_gepa(args: argparse.Namespace, backend: str):
    trainset = load_balanced_split(args, "train")
    valset = load_balanced_split(args, "val")
    evaluator = BalancedDirectQAGEPAEvaluator(
        args=args,
        backend=backend,
        web_search=getattr(args, "web_search", False),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = optimize_anything(
        seed_candidate={"system_prompt": load_seed_prompt(args)},
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=args.objective,
        background=args.background,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=args.run_dir,
                max_metric_calls=args.max_metric_calls,
                max_candidate_proposals=args.max_candidate_proposals,
                display_progress_bar=args.display_progress_bar,
                parallel=False,
                cache_evaluation=args.cache_evaluation,
                seed=args.seed,
            ),
            reflection=ReflectionConfig(
                reflection_lm=GeminiReflectionLM(
                    args.reflection_lm,
                    args.reflection_thinking_level,
                )
            ),
        ),
    )
    save_result(result, out_dir, args)


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--fpq_train_path", type=str, required=True, help="Path to FPQ train JSONL.")
    parser.add_argument("--tpq_train_path", type=str, required=True, help="Path to TPQ train JSONL.")
    parser.add_argument("--fpq_val_path", type=str, required=True, help="Path to FPQ validation JSONL.")
    parser.add_argument("--tpq_val_path", type=str, required=True, help="Path to TPQ validation JSONL.")
    parser.add_argument("--dataset_family", choices=tuple(DATASET_PAIRS), required=True, help="Paired FPQ/TPQ template family.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for GEPA outputs.")
    parser.add_argument("--seed_prompt", type=str, default=None, help="Initial direct-QA system prompt.")
    parser.add_argument("--seed_prompt_file", type=str, default=None, help="File containing the initial direct-QA system prompt.")
    parser.add_argument("--objective", type=str, default="Optimize one direct question-answering system prompt that corrects false presuppositions in FPQs while directly answering TPQs.")
    parser.add_argument("--background", type=str, default="Training and validation are balanced between false-presupposition questions and true-presupposition questions. FPQs reward explicit, accurate correction; TPQs reward direct answers without inventing a misconception.")
    parser.add_argument("--reflection_lm", type=str, default=DEFAULT_GEPA_REFLECTION_MODEL, help="GEPA reflection model.")
    parser.add_argument("--reflection_thinking_level", type=str, default=None, help="Gemini thinking level for reflection calls.")
    parser.add_argument("--evaluator_model_name", type=str, default=DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL, help="Gemini response-level evaluator model.")
    parser.add_argument("--max_metric_calls", type=int, default=500, help="GEPA metric-call budget.")
    parser.add_argument("--max_candidate_proposals", type=int, default=None, help="Optional GEPA proposal budget.")
    parser.add_argument("--run_dir", type=str, default=None, help="Optional GEPA internal run directory.")
    parser.add_argument("--seed", type=int, default=42, help="GEPA random seed.")
    parser.add_argument("--cache_evaluation", action="store_true", help="Enable GEPA evaluation caching.")
    parser.add_argument("--display_progress_bar", action="store_true", help="Show the GEPA progress bar.")
    parser.add_argument("--train_start_idx", type=int, default=0, help="Start index for both train JSONLs.")
    parser.add_argument("--val_start_idx", type=int, default=0, help="Start index for both validation JSONLs.")
    parser.add_argument("--fpq_train_limit", type=int, default=None, help="Optional FPQ train limit before balancing.")
    parser.add_argument("--tpq_train_limit", type=int, default=None, help="Optional TPQ train limit before balancing.")
    parser.add_argument("--fpq_val_limit", type=int, default=None, help="Optional FPQ validation limit before balancing.")
    parser.add_argument("--tpq_val_limit", type=int, default=None, help="Optional TPQ validation limit before balancing.")
    parser.add_argument("--system_role", type=str, default="system", help="Instruction role name.")
    parser.add_argument("--user_role", type=str, default="user", help="User role name.")
    parser.add_argument("--model_role", type=str, default="assistant", help="Assistant role name.")
    parser.add_argument("--thinking_cutoff_token", type=str, default=None, help="Cut evaluator input before this token.")
    parser.add_argument("--evaluator_thinking_level", type=str, default=None, help="Gemini thinking level for evaluator calls.")
    add_source_subparsers(parser)


def main(args: argparse.Namespace):
    if args.command == "transformers":
        import torch

        assert torch.cuda.is_available(), "Transformers models can only be run with CUDA device."
        run_gepa(args, backend="transformers")
    elif args.command == "gemini":
        run_gepa(args, backend="gemini")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a direct-QA prompt on balanced FPQ and TPQ data with GEPA.")
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    transformers_parser = subparsers.add_parser("transformers", help="Tune a transformers task model.")
    transformers_parser.add_argument("--model_name", type=str, required=True, help="Transformers model name or path.")
    transformers_parser.add_argument("--device", type=str, default="auto", help="Device map for transformers loading.")
    transformers_parser.add_argument("--dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"], help="Model dtype.")
    transformers_parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode if supported.")
    transformers_parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum generated task-model tokens.")
    transformers_parser.add_argument("--limit_input_length", type=int, default=3700, help="Maximum task-model input tokens.")
    add_common_args(transformers_parser)

    gemini_parser = subparsers.add_parser("gemini", help="Tune a Gemini task model.")
    gemini_parser.add_argument("--model_name", type=str, required=True, help="Gemini task model name.")
    gemini_parser.add_argument("--thinking_level", type=str, default=None, help="Gemini task-model thinking level.")
    gemini_parser.add_argument("--web_search", action="store_true", help="Enable Gemini web search for task calls.")
    add_common_args(gemini_parser)

    main(parser.parse_args())

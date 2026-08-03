import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constant.constant import (
    MODEL_FINAL_ANSWER_KEY,
    RAG_RETRIEVED_WIKIPEDIA_PASSAGES_KEY,
    RAG_TOP_PASSAGES_KEY,
)
from constant.response_level_score import (
    DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL,
    RESPONSE_LEVEL_SCORE_EXPLANATION_KEY,
    RESPONSE_LEVEL_SCORE_KEY,
)
from constant.gepa import DEFAULT_GEPA_REFLECTION_MODEL
from data_gen.template import get_template_cls
from response import FinalAnswerResponse
from utils.RAG_utils import run_RAG
from utils.argparse_utils import add_source_subparsers
from utils.gemini_utils import call_gemini_one_by_one_api, extract_gemini_response_text, save_gemini_thinking_trace
from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)


GEPA_SUPPORTED_DATASETS = ("CancerMyth", "CancerMythNFP", "QA2FPQ", "SynQA2FPQ", "CREPEFPQ", "CREPETPQ")

DEFAULT_SEED_PROMPT = """
    You are a helpful assistant that answer questions based on your knowledge.
    The user will ask a question, and you need to provide the answer to that question.
"""


def load_jsonl(path: str, start_idx: int = 0, limit: int | None = None) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    data = data[start_idx:]
    if limit is not None:
        data = data[:limit]
    return data


def load_seed_prompt(args: argparse.Namespace) -> str:
    if args.seed_prompt_file is not None:
        with open(args.seed_prompt_file, "r") as f:
            return f.read().strip()
    if args.seed_prompt is not None:
        return args.seed_prompt.strip()

    return DEFAULT_SEED_PROMPT


def disable_few_shot(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data = [copy.deepcopy(dp) for dp in data]
    for dp in data:
        dp["few_shot_data"] = []
    return data


def get_passages(dp: Dict[str, Any], source: str, **kwargs) -> List[str]:
    question = dp["question"]
    if source == "use_RAG":
        dp_passages = "\n\n".join(dp["passages"]).split("\n\n")
        passages = run_RAG(
            query=question,
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            passages=dp_passages,
            **kwargs,
        )
        if RAG_TOP_PASSAGES_KEY not in dp:
            dp[RAG_TOP_PASSAGES_KEY] = [passages]
        else:
            dp[RAG_TOP_PASSAGES_KEY].append(passages)
    elif source == "use_passages":
        passages = dp["passages"]
    elif source == "use_wiki":
        passages = dp[RAG_RETRIEVED_WIKIPEDIA_PASSAGES_KEY]
    elif source == "no_passages":
        passages = []
    else:
        raise ValueError(f"Unknown source {source}")
    return passages


def format_few_shot(few_shot_data: List[Dict[str, Any]], user_role: str, model_role: str) -> str:
    formatted = []
    for dp in few_shot_data:
        formatted.append(f"{user_role}: {dp['question']}\n{model_role}: {dp['answer']}")
    return "\n\n".join(formatted)


def build_direct_qa_prompt(
    candidate_prompt: str,
    dp: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    passages = get_passages(dp, args.source_command, **vars(args))
    few_shot = format_few_shot(dp.get("few_shot_data", []), args.user_role, args.model_role)
    rag_content = ""
    if len(passages) > 0:
        rag_content = (
            "For the following question, you are given this additional information:\n"
            f"Additional Information: {' ||| '.join(passages)}"
        )
    examples_intro = "Below are some examples to help you understand the task." if few_shot else ""
    system_content = "\n\n".join(
        part.strip()
        for part in [candidate_prompt, examples_intro, few_shot, rag_content]
        if part and part.strip()
    )
    return [
        {"role": args.system_role, "content": system_content},
        {"role": args.user_role, "content": dp["question"]},
    ]


class DirectQAGEPAEvaluator:
    def __init__(self, args: argparse.Namespace, backend: str, web_search: bool = False):
        self.web_search = web_search
        self.args = args
        self.backend = backend
        self.ScoreTemplateClass = get_template_cls(f"{args.dataset}ResponseLevelScoreTemplate")
        self.score_max = args.score_max if args.score_max is not None else default_score_max(args.dataset)

    def __call__(self, candidate: str | Dict[str, str], example: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        system_prompt = candidate["system_prompt"] if isinstance(candidate, dict) else candidate
        dp = copy.deepcopy(example)
        messages = build_direct_qa_prompt(system_prompt, dp, self.args)
        raw_answer = self.run_task_model(messages)
        answer = FinalAnswerResponse.model_validate_plain_text(raw_answer, model_role=self.args.model_role).get()
        dp[MODEL_FINAL_ANSWER_KEY] = answer
        score, explanation = self.score_response(dp)
        normalized_score = clamp(score / self.score_max, 0.0, 1.0)
        return normalized_score, {
            "question": dp.get("question"),
            "answer": answer,
            "raw_score": score,
            "normalized_score": normalized_score,
            "score_explanation": explanation,
        }

    def run_task_model(self, messages: List[Dict[str, str]]) -> str:
        if self.backend == "transformers":
            from utils.transformers_utils import run_transformers_model
            return run_transformers_model(
                model_name=self.args.model_name,
                messages=messages,
                enable_thinking=self.args.enable_thinking,
                max_new_tokens=self.args.max_new_tokens,
                limit_input_length=self.args.limit_input_length,
                dtype=self.args.dtype,
                device=self.args.device,
            )
        if self.backend == "gemini":
            response = call_gemini_one_by_one_api(
                messages,
                model=self.args.model_name,
                thinking_level=self.args.thinking_level,
                web_search=self.web_search,
            )
            return extract_gemini_response_text(response)
        raise ValueError(f"Unknown backend {self.backend}")

    def score_response(self, dp: Dict[str, Any]) -> Tuple[float, str]:
        prompt = self.ScoreTemplateClass(
            **dp,
            thinking_cutoff_token=self.args.thinking_cutoff_token,
            system_role=self.args.system_role,
            model_role=self.args.model_role,
            user_role=self.args.user_role,
        ).generate()
        response = call_gemini_one_by_one_api(
            prompt,
            model=self.args.evaluator_model_name,
            thinking_level=self.args.evaluator_thinking_level,
            web_search=self.web_search,
        )
        parsed = self.ScoreTemplateClass.ResponseClass.model_validate_plain_text(extract_gemini_response_text(response))
        dp[RESPONSE_LEVEL_SCORE_KEY] = parsed.score
        dp[RESPONSE_LEVEL_SCORE_EXPLANATION_KEY] = parsed.explanation
        save_gemini_thinking_trace(dp, RESPONSE_LEVEL_SCORE_KEY, response)
        return parsed.score, parsed.explanation
    
class GeminiReflectionLM(LanguageModel):
    def __init__(self, model_name: str, thinking_level: str | None = None):
        self.model_name = model_name
        self.thinking_level = thinking_level

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        response = call_gemini_one_by_one_api(
            messages=[
                {"role": "system", "content": None},
                {"role": "user", "content": prompt}
            ] if isinstance(prompt, str) else prompt,
            model=self.model_name,
            thinking_level=self.thinking_level,
        )
        return extract_gemini_response_text(response)


def default_score_max(dataset: str) -> float:
    return 5.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def run_gepa(args: argparse.Namespace, backend: str):
    trainset = disable_few_shot(
        load_jsonl(args.train_path, args.train_start_idx, args.train_limit)
    )
    valset = disable_few_shot(
        load_jsonl(args.val_path, args.val_start_idx, args.val_limit)
    )
    evaluator = DirectQAGEPAEvaluator(args=args, backend=backend, web_search=args.web_search if hasattr(args, "web_search") else False)
    seed_prompt = load_seed_prompt(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = optimize_anything(
            seed_candidate={"system_prompt": seed_prompt},
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
                reflection=ReflectionConfig(reflection_lm=GeminiReflectionLM(args.reflection_lm, args.reflection_thinking_level)),
            ),
        )
        save_result(result, out_dir, args)
    except Exception as e:
        print(f"GEPA optimization failed with error: {e}")
        exception_save(out_dir)


def exception_save(out_dir: Path):
    with open(out_dir / "gepa_result_summary.json", "w") as f:
        result = json.load(f)
    with open(out_dir / "best_system_prompt.txt", "w") as f:
        f.write(result["best_system_prompt"])


def save_result(result: Any, out_dir: Path, args: argparse.Namespace):
    best_candidate = result.best_candidate
    if isinstance(best_candidate, dict):
        best_system_prompt = best_candidate.get("system_prompt", "")
    else:
        best_system_prompt = best_candidate
    best_score = None
    if getattr(result, "val_aggregate_scores", None) is not None:
        best_score = result.val_aggregate_scores[result.best_idx]

    with open(out_dir / "best_system_prompt.txt", "w") as f:
        f.write(best_system_prompt)
    with open(out_dir / "gepa_result_summary.json", "w") as f:
        json.dump(
            {
                "best_idx": getattr(result, "best_idx", None),
                "best_score": best_score,
                "best_system_prompt": best_system_prompt,
                "args": vars(args),
            },
            f,
            indent=2,
        )


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train_path", type=str, required=True, help="Path to train JSONL.")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation JSONL.")
    parser.add_argument("--dataset", type=str, required=True, choices=GEPA_SUPPORTED_DATASETS, help="Dataset name.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for GEPA outputs.")
    parser.add_argument("--seed_prompt", type=str, default=None, help="Initial direct-QA system prompt.")
    parser.add_argument("--seed_prompt_file", type=str, default=None, help="File containing the initial direct-QA system prompt.")
    parser.add_argument("--objective", type=str, default="Optimize a direct question-answering system prompt to produce responses that score highly under the dataset response-level evaluator.")
    parser.add_argument("--background", type=str, default="The input examples are direct QA datapoints. The optimized prompt should preserve direct answering behavior while correcting false or unsupported presuppositions when the dataset/evaluator expects that.")
    parser.add_argument("--reflection_lm", type=str, default=DEFAULT_GEPA_REFLECTION_MODEL, help="GEPA reflection model, for example openai/gpt-5 or openrouter/google/gemini-3-flash-preview.")
    parser.add_argument("--reflection_thinking_level", type=str, default=None, help="Gemini thinking level for reflection calls.")
    parser.add_argument("--evaluator_model_name", type=str, default=DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL, help="Gemini model used by the response-level evaluator.")
    parser.add_argument("--score_max", type=float, default=None, help="Raw evaluator score corresponding to normalized GEPA score 1.0.")
    parser.add_argument("--max_metric_calls", type=int, default=500, help="GEPA metric-call budget.")
    parser.add_argument("--max_candidate_proposals", type=int, default=None, help="Optional GEPA proposal budget.")
    parser.add_argument("--run_dir", type=str, default=None, help="Optional GEPA internal run directory.")
    parser.add_argument("--seed", type=int, default=42, help="GEPA random seed.")
    parser.add_argument("--cache_evaluation", action="store_true", help="Enable GEPA evaluation caching.")
    parser.add_argument("--display_progress_bar", action="store_true", help="Show GEPA progress bar.")
    parser.add_argument("--train_start_idx", type=int, default=0, help="Start index into train JSONL.")
    parser.add_argument("--val_start_idx", type=int, default=0, help="Start index into val JSONL.")
    parser.add_argument("--train_limit", type=int, default=None, help="Optional max train examples.")
    parser.add_argument("--val_limit", type=int, default=None, help="Optional max validation examples.")
    parser.add_argument("--system_role", type=str, default="system", help="Instruction role name.")
    parser.add_argument("--user_role", type=str, default="user", help="User role name.")
    parser.add_argument("--model_role", type=str, default="assistant", help="Assistant role name.")
    parser.add_argument("--thinking_cutoff_token", type=str, default=None, help="Cut evaluator input before this token when model outputs thinking traces.")
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
    parser = argparse.ArgumentParser(description="Tune direct-QA prompts with GEPA.")
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    transformers_parser = subparsers.add_parser("transformers", help="GEPA prompt tuning for transformers models.")
    transformers_parser.add_argument("--model_name", type=str, required=True, help="Model name or path for transformers.")
    transformers_parser.add_argument("--device", type=str, default="auto", help="Device map for transformers loading.")
    transformers_parser.add_argument("--dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"], help="Model dtype.")
    transformers_parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode if the model supports it.")
    transformers_parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max generated task-model tokens.")
    transformers_parser.add_argument("--limit_input_length", type=int, default=3700, help="Max task-model input tokens.")
    add_common_args(transformers_parser)

    gemini_parser = subparsers.add_parser("gemini", help="GEPA prompt tuning for Gemini API models.")
    gemini_parser.add_argument("--model_name", type=str, required=True, help="Gemini task model name.")
    gemini_parser.add_argument("--thinking_level", type=str, default=None, help="Gemini thinking level for task-model calls.")
    gemini_parser.add_argument("--web_search", action="store_true", help="Enable web search for Gemini task-model calls.")
    add_common_args(gemini_parser)

    main(parser.parse_args())

import argparse, json, os
from evaluator import GeminiScoreBatchedEvaluator
from typing import Type, List, Dict, Any
from constant.gemini_score import GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_TASK_NAME
from constant.gemini_score import (
    GEMINI_SCORE_DIRECT_QA_PRECISION_COVERAGE_KEY,
    GEMINI_SCORE_DIRECT_QA_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_FINAL_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_FINAL_PRECISION_COVERAGE_KEY,
    GEMINI_SCORE_FACT_CHECK_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_FACT_CHECK_PRECISION_COVERAGE_KEY,
    GEMINI_SCORE_PRESUPPOSITION_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_PRESUPPOSITION_PRECISION_COVERAGE_KEY
)

def submit_preprocess(args: argparse.Namespace, evaluator: GeminiScoreBatchedEvaluator):
    """
    Submit batched preprocessing for Gemini-Score evaluation to Gemini API.
    For datapoints that do not require calling API, they will be directly saved to the output file.
    """
    task = GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_TASK_NAME
    with open(args.file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    data = data[args.start_idx:]
    job = evaluator.preprocess(data=data, save_to=args.out_file)
    out_file = f'{args.cache_dir}/{task}/{job.batch_job_name}.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        f.write(json.dumps(job.to_dict(), indent=4))

def submit_gemini_score_final_step(args: argparse.Namespace, EvaluatorClass: Type[GeminiScoreBatchedEvaluator]):
    """Submit batched final step evaluation for Gemini-Score evaluation to Gemini API.
    For datapoints that do not require calling API, they will be directly saved to the output file.
    """
    with open(args.file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    data = data[args.start_idx:]
    evaluator = EvaluatorClass()
    recall_job, precision_job = evaluator.evaluate_batch(
        data=data,
        save_to=args.out_file,
        disable_recall=args.disable_recall,
        disable_precision=args.disable_precision
    )
    if not args.disable_recall and recall_job is not None:
        out_file = f'{args.cache_dir}/{recall_job.batch_job_name}.json'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w') as f:
            f.write(json.dumps(recall_job.to_dict(), indent=4))
    if not args.disable_precision and precision_job is not None:
        out_file = f'{args.cache_dir}/{precision_job.batch_job_name}.json'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w') as f:
            f.write(json.dumps(precision_job.to_dict(), indent=4))

def filter_data_for_printing(data: List[Dict[str, Any]], fp_only: bool, tp_only: bool, enable_undefined: bool) -> List[Dict[str, Any]]:
    if fp_only:
        data = [dp for dp in data if 'labels' in dp and 'false presupposition' in dp['labels']]
    elif tp_only:
        data = [dp for dp in data if 'labels' in dp and 'false presupposition' not in dp['labels']]
    if not enable_undefined:
        data = [dp for dp in data if not gemini_score_undefined(dp)]
    return data

def gemini_score_undefined(dp: Dict[str, Any]) -> bool:
    return (
        (GEMINI_SCORE_DIRECT_QA_RECALL_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_DIRECT_QA_RECALL_COVERAGE_KEY]) == 0) or
        (GEMINI_SCORE_DIRECT_QA_PRECISION_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_DIRECT_QA_PRECISION_COVERAGE_KEY]) == 0) or
        (GEMINI_SCORE_FINAL_RECALL_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_FINAL_RECALL_COVERAGE_KEY]) == 0) or
        (GEMINI_SCORE_FINAL_PRECISION_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_FINAL_PRECISION_COVERAGE_KEY]) == 0) or
        (GEMINI_SCORE_FACT_CHECK_RECALL_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_FACT_CHECK_RECALL_COVERAGE_KEY]) == 0) or
        (GEMINI_SCORE_FACT_CHECK_PRECISION_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_FACT_CHECK_PRECISION_COVERAGE_KEY]) == 0) or
        (GEMINI_SCORE_PRESUPPOSITION_RECALL_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_PRESUPPOSITION_RECALL_COVERAGE_KEY]) == 0) or
        (GEMINI_SCORE_PRESUPPOSITION_PRECISION_COVERAGE_KEY in dp and len(dp[GEMINI_SCORE_PRESUPPOSITION_PRECISION_COVERAGE_KEY]) == 0)
    )
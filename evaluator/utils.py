from typing import List, Dict, Any, Tuple, Type
from google.genai import types
from constant.gemini_score import (
    ANSWER_EXTRACTED_PRESUPPOSITIONS_KEY,
    GEMINI_SCORE_FACT_CHECK_RECALL_TASK_NAME,
    GEMINI_SCORE_FACT_CHECK_PRECISION_TASK_NAME,
    GEMINI_SCORE_FACT_CHECK_RECALL_KEY,
    GEMINI_SCORE_FACT_CHECK_PRECISION_KEY,
    GEMINI_SCORE_FACT_CHECK_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_FACT_CHECK_PRECISION_COVERAGE_KEY,
    GEMINI_SCORE_DIRECT_QA_RECALL_TASK_NAME,
    GEMINI_SCORE_DIRECT_QA_PRECISION_TASK_NAME,
    GEMINI_SCORE_DIRECT_QA_RECALL_KEY,
    GEMINI_SCORE_DIRECT_QA_PRECISION_KEY,
    GEMINI_SCORE_DIRECT_QA_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_DIRECT_QA_PRECISION_COVERAGE_KEY,
    GEMINI_SCORE_FINAL_RECALL_TASK_NAME,
    GEMINI_SCORE_FINAL_PRECISION_TASK_NAME,
    GEMINI_SCORE_FINAL_RECALL_KEY,
    GEMINI_SCORE_FINAL_PRECISION_KEY,
    GEMINI_SCORE_FINAL_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_FINAL_PRECISION_COVERAGE_KEY,
    GEMINI_SCORE_PRESUPPOSITION_RECALL_TASK_NAME,
    GEMINI_SCORE_PRESUPPOSITION_PRECISION_TASK_NAME,
    GEMINI_SCORE_PRESUPPOSITION_RECALL_KEY,
    GEMINI_SCORE_PRESUPPOSITION_PRECISION_KEY,
    GEMINI_SCORE_PRESUPPOSITION_RECALL_COVERAGE_KEY,
    GEMINI_SCORE_PRESUPPOSITION_PRECISION_COVERAGE_KEY,
    GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_TASK_NAME
)
from response import Response, ClaimCoverageResponse
from .batch_job_cache import GeminiScoreBatchJobCache
import numpy as np
import json
from utils.gemini_utils import checkback, extract_gemini_response_text, save_gemini_thinking_trace

def to_cache_jobs(
    recall_job: types.BatchJob | None,
    precision_job: types.BatchJob | None,
    save_to: str,
    recall_task_name: str,
    precision_task_name: str,
    disable_recall: bool,
    disable_precision: bool,
    data: List[Dict[str, Any]],
    EvaluatorClass: Type[Any]
) -> Tuple[GeminiScoreBatchJobCache | None, GeminiScoreBatchJobCache | None]:
    recall_job_cache, precision_job_cache = None, None
    if not disable_recall and recall_job is not None:
        recall_job_cache = GeminiScoreBatchJobCache(
            save_to=save_to,
            batch_job_name=recall_job.name,
            ResponseClass=ClaimCoverageResponse,
            task=recall_task_name,
            EvaluatorClass=EvaluatorClass
        )
    if not disable_precision and precision_job is not None:
        precision_job_cache = GeminiScoreBatchJobCache(
            save_to=save_to,
            batch_job_name=precision_job.name,
            ResponseClass=ClaimCoverageResponse,
            task=precision_task_name,
            EvaluatorClass=EvaluatorClass
        )
    with open(save_to, 'w') as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")
    return recall_job_cache, precision_job_cache

def parse_response_one_sided(
    responses: List[Dict[str, Any]],
    dp: Dict[str, Any],
    ResponseClass: Type[Response],
    coverage_key: str,
    coverage_length: int
):
    for response in responses:
        if dp.get(coverage_key, None) is None:
            dp[coverage_key] = [-1] * coverage_length
        j = int(response.metadata['index'])
        assert dp[coverage_key][j] == -1, f"Coverage for index {j} in data point {dp['id']} has already been filled."
        dp[coverage_key][j] = ResponseClass.model_validate_plain_text(extract_gemini_response_text(response.response)).get()
        save_gemini_thinking_trace(dp, coverage_key, response.response, index=j)

def mean_coverage(dp: Dict[str, Any], coverage_key: str, score_key: str) -> float:
    assert not np.any([cov == -1 for cov in dp[coverage_key]]), f"Some coverage values are not filled in FP Score for data point {dp['id']}"
    dp[score_key] = np.mean(dp[coverage_key]) if len(dp[coverage_key]) > 0 else 1.0
    
def mean_coverage_by_task(dp: Dict[str, Any], task: str):
    if task == GEMINI_SCORE_FACT_CHECK_RECALL_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_FACT_CHECK_RECALL_COVERAGE_KEY, GEMINI_SCORE_FACT_CHECK_RECALL_KEY)
    elif task == GEMINI_SCORE_FACT_CHECK_PRECISION_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_FACT_CHECK_PRECISION_COVERAGE_KEY, GEMINI_SCORE_FACT_CHECK_PRECISION_KEY)
    elif task == GEMINI_SCORE_DIRECT_QA_RECALL_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_DIRECT_QA_RECALL_COVERAGE_KEY, GEMINI_SCORE_DIRECT_QA_RECALL_KEY)
    elif task == GEMINI_SCORE_DIRECT_QA_PRECISION_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_DIRECT_QA_PRECISION_COVERAGE_KEY, GEMINI_SCORE_DIRECT_QA_PRECISION_KEY)
    elif task == GEMINI_SCORE_FINAL_RECALL_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_FINAL_RECALL_COVERAGE_KEY, GEMINI_SCORE_FINAL_RECALL_KEY)
    elif task == GEMINI_SCORE_FINAL_PRECISION_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_FINAL_PRECISION_COVERAGE_KEY, GEMINI_SCORE_FINAL_PRECISION_KEY)
    elif task == GEMINI_SCORE_PRESUPPOSITION_RECALL_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_PRESUPPOSITION_RECALL_COVERAGE_KEY, GEMINI_SCORE_PRESUPPOSITION_RECALL_KEY)
    elif task == GEMINI_SCORE_PRESUPPOSITION_PRECISION_TASK_NAME:
        mean_coverage(dp, GEMINI_SCORE_PRESUPPOSITION_PRECISION_COVERAGE_KEY, GEMINI_SCORE_PRESUPPOSITION_PRECISION_KEY)
        
def checkback_and_parse(
    job: GeminiScoreBatchJobCache,
    recall_task_name: str,
    precision_task_name: str,
    recall_coverage_key: str,
    precision_coverage_key: str,
    gold_presuppositions_key: str,
    detected_or_validated_presuppositions_key: str
) -> List[Dict[str, Any]]:
    batch_job_name = job.batch_job_name
    out_file = job.save_to
    task = job.task
    ResponseClass = job.ResponseClass
    results = checkback(batch_job_name)
    result_ids = {response.metadata['id'] for response in results}
    results = {
        id: [
            r for r in results if r.metadata['id'] == id
        ]
        for id in result_ids
    }
    with open(out_file, 'r') as f:
        target_dataset = [json.loads(line.strip()) for line in f]
        target_dataset = {dp['id']: dp for dp in target_dataset}
    for id, responses in results.items():
        dp = target_dataset[id]
        if task == GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_TASK_NAME:
            for response in responses:
                dp[ANSWER_EXTRACTED_PRESUPPOSITIONS_KEY] = ResponseClass.model_validate_plain_text(extract_gemini_response_text(response.response)).get()
                save_gemini_thinking_trace(dp, ANSWER_EXTRACTED_PRESUPPOSITIONS_KEY, response.response)
        elif task == recall_task_name:
            parse_response_one_sided(
                responses=responses,
                dp=dp,
                ResponseClass=ResponseClass,
                coverage_key=recall_coverage_key,
                coverage_length=len(dp[gold_presuppositions_key])
            )
        elif task == precision_task_name:
            parse_response_one_sided(
                responses=responses,
                dp=dp,
                ResponseClass=ResponseClass,
                coverage_key=precision_coverage_key,
                coverage_length=len(dp[detected_or_validated_presuppositions_key])
            )
    target_dataset = list(target_dataset.values())
    if task != GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_TASK_NAME:
        for dp in target_dataset:
            mean_coverage_by_task(dp, task)
    return target_dataset

def checkback_and_parse_response_level(
    job: GeminiScoreBatchJobCache,
    score_key: str,
    explanation_key: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    batch_job_name = job.batch_job_name
    out_file = job.save_to
    ResponseClass = job.ResponseClass
    results = checkback(batch_job_name)
    results = {r.metadata['id']: r for r in results}
    with open(out_file, 'r') as f:
        target_dataset = [json.loads(line.strip()) for line in f]
        target_dataset = {str(dp['id']): dp for dp in target_dataset}
    unresolved_datapoints = []
    for id, response in results.items():
        if id in target_dataset:
            dp = target_dataset[id]
            if response.response is None:
                unresolved_datapoints.append(dp)
                continue
            parsed_response = ResponseClass.model_validate_plain_text(extract_gemini_response_text(response.response))
            dp[score_key] = parsed_response.score
            dp[explanation_key] = parsed_response.explanation
            save_gemini_thinking_trace(dp, score_key, response.response)
    return list(target_dataset.values()), unresolved_datapoints

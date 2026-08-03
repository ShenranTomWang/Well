from .fact_check_final_response_operator import BatchFactCheckFinalResponseOperator, FactCheckFinalResponseOperator, register
from .batch_job_cache import GeminiRunFactCheckFinalResponseBatchJobCache
from utils.gemini_utils import checkback, message2gemini_request, submit_gemini_job, call_gemini_one_by_one_api, extract_gemini_response_text, save_gemini_thinking_trace
from constant.constant import (
    MODEL_FINAL_ANSWER_KEY,
    MODEL_FP_IDENTIFICATION_KEY,
    FACTCHECK_RESULTS_KEY,
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_FEEDBACK_ACTION_KEY,
    MODEL_EXTRACTED_LOGICAL_FORMS_KEY,
    MODEL_CONVERTED_STATEMENT_KEY
)
from response import FinalAnswerResponse
import json
from data_gen.template import (
    FactCheckFinalAnswerTemplate,
    FinalAnswerTemplate,
    FactCheckFPInterpretationTemplate
)
from typing import Dict, Any
import torch
from utils.RAG_utils import get_passages

@register()
class GeminiFactCheckFinalResponseOperator(FactCheckFinalResponseOperator, BatchFactCheckFinalResponseOperator):
    def __init__(self, pipeline: str, **kwargs):
        self.pipeline = pipeline
        if pipeline in ['fact_check', 'statement']:
            self.TemplateClass = FactCheckFinalAnswerTemplate
        elif pipeline == 'feedback_action':
            self.TemplateClass = FinalAnswerTemplate
        elif pipeline == 'interpretation':
            self.TemplateClass = FactCheckFPInterpretationTemplate
        else:
            raise ValueError(f"Unknown pipeline {pipeline}")

    def _build_prompt(self, dp: Dict[str, Any], source: str, **kwargs):
        passages = get_passages(dp, source=source, **kwargs)
        return self.TemplateClass(
            question=dp['question'],
            few_shot_data=dp["few_shot_data"],
            factcheck_results=dp[FACTCHECK_RESULTS_KEY] if FACTCHECK_RESULTS_KEY in dp else None,
            model_FP_identification=dp[MODEL_FP_IDENTIFICATION_KEY] if MODEL_FP_IDENTIFICATION_KEY in dp else None,
            model_detected_presuppositions=[dp[MODEL_CONVERTED_STATEMENT_KEY]] if self.pipeline == 'statement' else dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY] if MODEL_DETECTED_PRESUPPOSITIONS_KEY in dp else None,
            model_feedback_action=dp[MODEL_FEEDBACK_ACTION_KEY] if MODEL_FEEDBACK_ACTION_KEY in dp else None,
            model_extracted_logical_form=dp[MODEL_EXTRACTED_LOGICAL_FORMS_KEY] if MODEL_EXTRACTED_LOGICAL_FORMS_KEY in dp else None,
            passages=passages,
            **kwargs
        ).generate()

    @torch.inference_mode()
    def respond(
        self,
        dp: Dict[str, Any],
        source: str,
        model_name: str,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(dp, source=source, **kwargs)
        response = call_gemini_one_by_one_api(prompt, model=model_name, web_search=web_search, thinking_level=thinking_level)
        dp[MODEL_FINAL_ANSWER_KEY] = FinalAnswerResponse.model_validate_plain_text(extract_gemini_response_text(response)).get()
        save_gemini_thinking_trace(dp, MODEL_FINAL_ANSWER_KEY, response)
        return dp
    
    @torch.inference_mode()
    def submit_job(
        self,
        dps: Dict[str, Any],
        save_to: str,
        model_name: str,
        source: str,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> GeminiRunFactCheckFinalResponseBatchJobCache:
        requests = []
        for dp in dps:
            prompt = self._build_prompt(dp, source=source, **kwargs)
            request = message2gemini_request(
                metadata={'id': dp['id']},
                messages=prompt,
                model=model_name,
                web_search=web_search,
                thinking_level=thinking_level
            )
            requests.append(request)
        job_info = submit_gemini_job(requests, model=model_name)
        with open(save_to, 'w') as f:
            for dp in dps:
                f.write(json.dumps(dp) + '\n')
        job_cache = GeminiRunFactCheckFinalResponseBatchJobCache(
            save_to=save_to,
            batch_job_name=job_info.name,
            OperatorClass=self.__class__
        )
        return job_cache
    
    def checkback(self, job: GeminiRunFactCheckFinalResponseBatchJobCache, **kwargs):
        job_name = job.batch_job_name
        responses = checkback(job_name, **kwargs)
        with open(job.save_to, 'r') as f:
            dataset = [json.loads(line) for line in f]
            dataset = {
                str(dp['id']): dp for dp in dataset
            }
        for response in responses:
            dp = dataset[response.metadata['id']]
            dp[MODEL_FINAL_ANSWER_KEY] = FinalAnswerResponse.model_validate_plain_text(extract_gemini_response_text(response.response)).get()
            save_gemini_thinking_trace(dp, MODEL_FINAL_ANSWER_KEY, response.response)
        with open(job.save_to, 'w') as f:
            for dp in dataset.values():
                f.write(json.dumps(dp) + '\n')

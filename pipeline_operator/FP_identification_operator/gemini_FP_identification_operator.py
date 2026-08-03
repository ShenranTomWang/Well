from .FP_identification_operator import BatchFPIdentificationOperator, FPIdentificationOperator, register
from .batch_job_cache import GeminiRunFPIdentificationBatchJobCache
from utils.gemini_utils import checkback, message2gemini_request, submit_gemini_job, call_gemini_one_by_one_api, extract_gemini_response_text, save_gemini_thinking_trace
from utils.RAG_utils import get_passages
import json
from typing import Dict, List, Any
import torch
from data_gen.template import FPIdentificationTemplate

@register()
class GeminiFPIdentificationOperator(FPIdentificationOperator, BatchFPIdentificationOperator):
    def _build_prompt(self, dp: Dict[str, Any], passages: List[str], **kwargs):
        return FPIdentificationTemplate(
            question=dp['question'],
            few_shot_data=dp["few_shot_data"],
            passages=passages,
            **kwargs
        ).generate()

    @torch.inference_mode()
    def extract(
        self,
        dp: Dict[str, Any],
        source: str,
        model_name: str,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> Dict[str, Any]:
        passages = get_passages(dp, source=source, **kwargs)
        prompt = self._build_prompt(dp, passages=passages, **kwargs)
        response = call_gemini_one_by_one_api(prompt, model=model_name, web_search=web_search, thinking_level=thinking_level)
        dp[FPIdentificationTemplate.answer_key] = FPIdentificationTemplate.ResponseClass.model_validate_plain_text(extract_gemini_response_text(response)).get()
        save_gemini_thinking_trace(dp, FPIdentificationTemplate.answer_key, response)
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
    ) -> GeminiRunFPIdentificationBatchJobCache:
        requests = []
        for dp in dps:
            passages = get_passages(dp, source=source, **kwargs)
            prompt = self._build_prompt(dp, passages=passages, **kwargs)
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
        job_cache = GeminiRunFPIdentificationBatchJobCache(
            save_to=save_to,
            batch_job_name=job_info.name,
            OperatorClass=self.__class__
        )
        return job_cache

    def checkback(self, job: GeminiRunFPIdentificationBatchJobCache, **kwargs):
        job_name = job.batch_job_name
        responses = checkback(job_name, **kwargs)
        with open(job.save_to, 'r') as f:
            dataset = [json.loads(line) for line in f]
            dataset = {
                str(dp['id']): dp for dp in dataset
            }
        for response in responses:
            dp = dataset[response.metadata['id']]
            dp[FPIdentificationTemplate.answer_key] = FPIdentificationTemplate.ResponseClass.model_validate_plain_text(
                extract_gemini_response_text(response.response)
            ).get()
            save_gemini_thinking_trace(dp, FPIdentificationTemplate.answer_key, response.response)
        with open(job.save_to, 'w') as f:
            for dp in dataset.values():
                f.write(json.dumps(dp) + '\n')

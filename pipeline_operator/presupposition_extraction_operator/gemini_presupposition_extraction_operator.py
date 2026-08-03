from .presupposition_extraction_operator import BatchPresuppositionExtractionOperator, PresuppositionExtractionOperator, register
from .batch_job_cache import GeminiRunPresuppositionExtractionBatchJobCache
from utils.gemini_utils import checkback, message2gemini_request, submit_gemini_job, call_gemini_one_by_one_api, extract_gemini_response_text, save_gemini_thinking_trace
from constant.constant import MODEL_DETECTED_PRESUPPOSITIONS_KEY
from response import PresuppositionExtractionResponse
import json
from typing import Dict
import torch
from data_gen.template import PresuppositionExtractionTemplate
from utils.RAG_utils import get_passages

@register()
class GeminiPresuppositionExtractionOperator(PresuppositionExtractionOperator, BatchPresuppositionExtractionOperator):
    def _build_prompt(self, dp: Dict[str, any], source: str, **kwargs):
        passages = get_passages(dp, source=source, **kwargs)
        return PresuppositionExtractionTemplate(
            question=dp['question'],
            few_shot_data=dp["few_shot_data"],
            passages=passages,
            **kwargs
        ).generate()

    @torch.inference_mode()
    def extract(
        self,
        dp: Dict[str, any],
        model_name: str,
        filter_presuppositions: bool,
        source: str,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> Dict[str, any]:
        prompt = self._build_prompt(dp, source=source, **kwargs)
        response = call_gemini_one_by_one_api(prompt, model=model_name, web_search=web_search, thinking_level=thinking_level)
        dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY] = PresuppositionExtractionResponse.model_validate_plain_text(
            extract_gemini_response_text(response),
            filter_presuppositions=filter_presuppositions
        ).get()
        save_gemini_thinking_trace(dp, MODEL_DETECTED_PRESUPPOSITIONS_KEY, response)
        return dp
    
    @torch.inference_mode()
    def submit_job(
        self,
        dps: Dict[str, any],
        save_to: str,
        model_name: str,
        source: str,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> GeminiRunPresuppositionExtractionBatchJobCache:
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
        job_cache = GeminiRunPresuppositionExtractionBatchJobCache(
            save_to=save_to,
            batch_job_name=job_info.name,
            OperatorClass=self.__class__
        )
        return job_cache
    
    def checkback(self, job: GeminiRunPresuppositionExtractionBatchJobCache, filter_presuppositions: bool, **kwargs):
        job_name = job.batch_job_name
        responses = checkback(job_name, **kwargs)
        with open(job.save_to, 'r') as f:
            dataset = [json.loads(line) for line in f]
            dataset = {
                str(dp['id']): dp for dp in dataset
            }
        for response in responses:
            dp = dataset[response.metadata['id']]
            dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY] = PresuppositionExtractionResponse.model_validate_plain_text(
                extract_gemini_response_text(response.response),
                filter_presuppositions=filter_presuppositions
            ).get()
            save_gemini_thinking_trace(dp, MODEL_DETECTED_PRESUPPOSITIONS_KEY, response.response)
        with open(job.save_to, 'w') as f:
            for dp in dataset.values():
                f.write(json.dumps(dp) + '\n')

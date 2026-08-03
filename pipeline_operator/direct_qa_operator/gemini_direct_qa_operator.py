from .direct_qa_operator import BatchDirectQAOperator, DirectQAOperator, register
from .batch_job_cache import GeminiRunDirectQABatchJobCache
from utils.gemini_utils import checkback, message2gemini_request, submit_gemini_job, call_gemini_one_by_one_api, extract_gemini_response_text, save_gemini_thinking_trace
import json
from data_gen.template import DirectQATemplate
from typing import Dict
import torch
from utils.RAG_utils import get_passages

@register()
class GeminiDirectQAOperator(DirectQAOperator, BatchDirectQAOperator):
    def _build_prompt(self, dp: Dict[str, any], source: str, **kwargs):
        passages = get_passages(dp, source=source, **kwargs)
        return DirectQATemplate(
            question=dp['question'],
            few_shot_data=dp["few_shot_data"],
            passages=passages,
            **kwargs
        ).generate()

    @torch.inference_mode()
    def qa(
        self,
        dp: Dict[str, any],
        gemini_model_name: str,
        source: str,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> Dict[str, any]:
        prompt = self._build_prompt(dp, source=source, **kwargs)
        response = call_gemini_one_by_one_api(prompt, model=gemini_model_name, web_search=web_search, thinking_level=thinking_level)
        dp[DirectQATemplate.answer_key] = DirectQATemplate.ResponseClass.model_validate_plain_text(extract_gemini_response_text(response)).get()
        save_gemini_thinking_trace(dp, DirectQATemplate.answer_key, response)
        return dp
    
    @torch.inference_mode()
    def submit_job(
        self,
        dps: Dict[str, any],
        save_to: str,
        gemini_model_name: str,
        source: str,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> GeminiRunDirectQABatchJobCache:
        requests = []
        for dp in dps:
            prompt = self._build_prompt(dp, source=source, **kwargs)
            request = message2gemini_request(metadata={'id': str(dp['id'])}, messages=prompt, model=gemini_model_name, web_search=web_search, thinking_level=thinking_level)
            requests.append(request)
        job_info = submit_gemini_job(requests, model=gemini_model_name)
        with open(save_to, 'w') as f:
            for dp in dps:
                f.write(json.dumps(dp) + '\n')
        job_cache = GeminiRunDirectQABatchJobCache(
            save_to=save_to,
            batch_job_name=job_info.name,
            OperatorClass=self.__class__
        )
        return job_cache
    
    def checkback(self, job: GeminiRunDirectQABatchJobCache, **kwargs):
        job_name = job.batch_job_name
        responses = checkback(job_name, **kwargs)
        with open(job.save_to, 'r') as f:
            dataset = [json.loads(line) for line in f]
            dataset = {
                str(dp['id']): dp for dp in dataset
            }
        for response in responses:
            dp = dataset[response.metadata['id']]
            dp[DirectQATemplate.answer_key] = DirectQATemplate.ResponseClass.model_validate_plain_text(extract_gemini_response_text(response.response)).get()
            save_gemini_thinking_trace(dp, DirectQATemplate.answer_key, response.response)
        with open(job.save_to, 'w') as f:
            for dp in dataset.values():
                f.write(json.dumps(dp) + '\n')

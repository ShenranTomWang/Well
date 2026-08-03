from typing import Dict, Any, List, Tuple
from data_gen.template import get_template_cls
from utils.gemini_utils import message2gemini_request, submit_gemini_job, call_gemini_one_by_one_api, extract_gemini_response_text, save_gemini_thinking_trace
from .utils import checkback_and_parse_response_level
from .batch_job_cache import ResponseLevelScoreBatchJobCache
from constant.response_level_score import (
    DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL,
    RESPONSE_LEVEL_SCORE_TASK_NAME,
    RESPONSE_LEVEL_SCORE_KEY,
    RESPONSE_LEVEL_SCORE_EXPLANATION_KEY,
)
import json

class ResponseLevelScoreBatchedEvaluator:
    dataset_name: str
    
    def __init__(
        self,
        dataset_name: str,
        evaluator_model_name: str = DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL,
        thinking_cutoff_token: str = None,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self.evaluator_model_name = evaluator_model_name
        self.thinking_cutoff_token = thinking_cutoff_token
        
    def evaluate(
        self,
        data: List[Dict],
        save_to: str,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        thinking_level: str = None,
        **kwargs
    ):
        with open(save_to, 'w') as f:
            for dp in data:
                TemplateClass = get_template_cls(f"{self.dataset_name}ResponseLevelScoreTemplate")
                prompt = TemplateClass(
                    **dp,
                    thinking_cutoff_token=self.thinking_cutoff_token,
                    system_role=system_role,
                    model_role=model_role,
                    user_role=user_role
                ).generate()
                response = call_gemini_one_by_one_api(
                    prompt,
                    model=self.evaluator_model_name,
                    thinking_level=thinking_level
                )
                ResponseClass = TemplateClass.ResponseClass
                response_text = extract_gemini_response_text(response)
                parsed_response = ResponseClass.model_validate_plain_text(response_text)
                dp[RESPONSE_LEVEL_SCORE_KEY] = parsed_response.score
                dp[RESPONSE_LEVEL_SCORE_EXPLANATION_KEY] = parsed_response.explanation
                save_gemini_thinking_trace(dp, RESPONSE_LEVEL_SCORE_KEY, response)
                f.write(json.dumps(dp) + "\n")
    
    def evaluate_batch(
        self,
        data: List[Dict],
        save_to: str,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        thinking_level: str = None,
        **kwargs
    ) -> Tuple[Any]:
        requests = []
        for dp in data:
            TemplateClass = get_template_cls(f"{self.dataset_name}ResponseLevelScoreTemplate")
            prompt = TemplateClass(
                **dp,
                thinking_cutoff_token=self.thinking_cutoff_token,
                system_role=system_role,
                model_role=model_role,
                user_role=user_role
            ).generate()
            request = message2gemini_request(
                metadata={"id": str(dp["id"]), "task": RESPONSE_LEVEL_SCORE_TASK_NAME},
                messages=prompt,
                model=self.evaluator_model_name,
                thinking_level=thinking_level
            )
            requests.append(request)
        if len(requests) > 0:
            job = submit_gemini_job(
                requests=requests,
                model=self.evaluator_model_name,
            )
            job_cache = ResponseLevelScoreBatchJobCache(
                save_to=save_to,
                batch_job_name=job.name if job is not None else None,
                ResponseClass=TemplateClass.ResponseClass,
                EvaluatorClass=self.__class__
            )
        else:
            job_cache = None
        with open(save_to, 'w') as f:
            for dp in data:
                f.write(json.dumps(dp) + "\n")
        return job_cache
    
    def checkback_and_parse(
        self,
        job: ResponseLevelScoreBatchJobCache,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        thinking_level: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        data, unresolved_datapoints = checkback_and_parse_response_level(
            job=job,
            score_key=RESPONSE_LEVEL_SCORE_KEY,
            explanation_key=RESPONSE_LEVEL_SCORE_EXPLANATION_KEY
        )
        for dp in unresolved_datapoints:
            TemplateClass = get_template_cls(f"{self.dataset_name}ResponseLevelScoreTemplate")
            prompt = TemplateClass(
                **dp,
                thinking_cutoff_token=self.thinking_cutoff_token,
                system_role=system_role,
                model_role=model_role,
                user_role=user_role
            ).generate()
            response = call_gemini_one_by_one_api(
                prompt,
                model=self.evaluator_model_name,
                thinking_level=thinking_level
            )
            response_text = extract_gemini_response_text(response)
            parsed_response = TemplateClass.ResponseClass.model_validate_plain_text(response_text)
            dp[RESPONSE_LEVEL_SCORE_KEY] = parsed_response.score
            dp[RESPONSE_LEVEL_SCORE_EXPLANATION_KEY] = parsed_response.explanation
            save_gemini_thinking_trace(dp, RESPONSE_LEVEL_SCORE_KEY, response)
        return data

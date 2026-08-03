import json
from typing import Dict
from constant.constant import (
    FACTCHECK_RESULTS_KEY,
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_CONVERTED_STATEMENT_KEY
)
from utils.gemini_utils import message2gemini_request, submit_gemini_job, checkback, call_gemini_one_by_one_api, extract_gemini_response_text, save_gemini_thinking_trace
from .batch_job_cache import GeminiRunFactCheckBatchJobCache
import torch
from data_gen.template import (
    LLMCheckTemplate,
    FeedbackActionTemplate,
    get_template_cls
)
from utils.RAG_utils import get_passages
from .check_operator import BatchCheckOperator, CheckOperator, register

@register()
class GeminiCheckOperator(CheckOperator, BatchCheckOperator):
    def __init__(self, pipeline: str, template_class: str = None):
        self.pipeline = pipeline
        if template_class is not None:
            self.TemplateClass = get_template_cls(template_class)
        elif pipeline in ['fact_check', 'statement']:
            self.TemplateClass = LLMCheckTemplate
        elif pipeline == 'feedback_action':
            self.TemplateClass = FeedbackActionTemplate
        else:
            raise ValueError(f"Unknown pipeline {pipeline}")

    def _get_presuppositions(self, dp: Dict[str, any], check_gold: bool):
        if check_gold:
            return dp["presuppositions"]
        detected = dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY]
        return detected["presuppositions"] if isinstance(detected, dict) else detected

    def _get_statements(self, dp: Dict[str, any]):
        statement = dp[MODEL_CONVERTED_STATEMENT_KEY]
        statements = [statement] if isinstance(statement, str) else statement
        return [statement for statement in statements if statement.strip() != ""]

    def _build_prompt(
            self,
            dp: Dict[str, any],
            presupposition: str,
            source: str,
            doctor_suggestion: bool = False,
            **kwargs
        ):
        passages = get_passages(
            dp,
            query=presupposition,
            source=source,
            instruction='Given a statement, retrieve relevant passages that validate or refute the statement',
            **kwargs
        )
        return self.TemplateClass(
            model_detected_presupposition=presupposition,
            few_shot_data=dp["few_shot_data"],
            passages=passages,
            doctor_suggestion=doctor_suggestion,
            **kwargs
        ).generate()

    def _build_feedback_action_prompt(self, dp: Dict[str, any], presuppositions: list[str], source: str, **kwargs):
        passages = get_passages(
            dp,
            query='; '.join(presuppositions),
            source=source,
            instruction='Given a list of statements, retrieve relevant passages that validate or refute the statements',
            **kwargs
        )
        return self.TemplateClass(
            question=dp['question'],
            model_detected_presuppositions=presuppositions,
            few_shot_data=dp["few_shot_data"],
            passages=passages,
            **kwargs
        ).generate()

    @torch.inference_mode()
    def check(
        self,
        dp: Dict[str, any],
        model_name: str,
        source: str,
        check_gold: bool,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> Dict[str, any]:
        presuppositions = self._get_statements(dp) if self.pipeline == 'statement' else self._get_presuppositions(dp, check_gold=check_gold)
        if self.pipeline == 'feedback_action':
            prompt = self._build_feedback_action_prompt(dp, presuppositions=presuppositions, source=source, **kwargs)
            response = call_gemini_one_by_one_api(prompt, model=model_name, web_search=web_search, thinking_level=thinking_level)
            dp[self.TemplateClass.answer_key] = self.TemplateClass.ResponseClass.model_validate_plain_text(extract_gemini_response_text(response)).get()
            save_gemini_thinking_trace(dp, self.TemplateClass.answer_key, response)
            return dp

        if FACTCHECK_RESULTS_KEY not in dp or len(dp[FACTCHECK_RESULTS_KEY]) != len(presuppositions):
            dp[FACTCHECK_RESULTS_KEY] = [-1] * len(presuppositions)
        for i, presupposition in enumerate(presuppositions):
            prompt = self._build_prompt(
                dp, presupposition=presupposition,
                source=source,
                doctor_suggestion=dp["doctor_suggestion"][i] if check_gold and "doctor_suggestion" in dp else False,
                **kwargs
            )
            response = call_gemini_one_by_one_api(prompt, model=model_name, web_search=web_search, thinking_level=thinking_level)
            dp[FACTCHECK_RESULTS_KEY][i] = self.TemplateClass.ResponseClass.model_validate_plain_text(extract_gemini_response_text(response)).get()
            save_gemini_thinking_trace(dp, FACTCHECK_RESULTS_KEY, response, index=i)
        return dp
    
    @torch.inference_mode()
    def submit_job(
        self,
        dps: Dict[str, any],
        save_to: str,
        model_name: str,
        source: str,
        check_gold: bool,
        web_search: bool,
        thinking_level: str,
        **kwargs
    ) -> GeminiRunFactCheckBatchJobCache:
        requests = []
        for dp in dps:
            presuppositions = self._get_statements(dp) if self.pipeline == 'statement' else self._get_presuppositions(dp, check_gold=check_gold)
            if self.pipeline == 'feedback_action':
                prompt = self._build_feedback_action_prompt(dp, presuppositions=presuppositions, source=source, **kwargs)
                request = message2gemini_request(
                    metadata={'id': dp['id']},
                    messages=prompt,
                    model=model_name,
                    web_search=web_search,
                    thinking_level=thinking_level
                )
                requests.append(request)
            else:
                for i, presupposition in enumerate(presuppositions):
                    prompt = self._build_prompt(
                        dp,
                        presupposition=presupposition,
                        source=source,
                        doctor_suggestion=dp["doctor_suggestion"][i] if "doctor_suggestion" in dp else False,
                        **kwargs
                    )
                    request = message2gemini_request(
                        metadata={'id': dp['id'], 'index': str(i)},
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
        job_cache = GeminiRunFactCheckBatchJobCache(
            save_to=save_to,
            batch_job_name=job_info.name,
            check_gold=check_gold,
            pipeline=self.pipeline
        )
        return job_cache
    
    def checkback(self, job: GeminiRunFactCheckBatchJobCache, **kwargs):
        job_name = job.batch_job_name
        check_gold = job.check_gold
        responses = checkback(job_name, **kwargs)
        response_keys = set([response.metadata['id'] for response in responses])
        responses = {
            response_key: [
                response for response in responses if response.metadata['id'] == response_key
            ] for response_key in response_keys
        }
        with open(job.save_to, 'r') as f:
            dataset = [json.loads(line) for line in f]
            dataset = {
                str(dp['id']): dp for dp in dataset
            }
        if self.pipeline == 'feedback_action':
            for response_key, grouped_responses in responses.items():
                dp = dataset[response_key]
                dp[self.TemplateClass.answer_key] = self.TemplateClass.ResponseClass.model_validate_plain_text(extract_gemini_response_text(grouped_responses[0].response)).get()
                save_gemini_thinking_trace(dp, self.TemplateClass.answer_key, grouped_responses[0].response)
            assert all(self.TemplateClass.answer_key in dp for dp in dataset.values())
        else:
            for response_key, grouped_responses in responses.items():
                dp = dataset[response_key]
                presuppositions = self._get_presuppositions(dp, check_gold=check_gold)
                if FACTCHECK_RESULTS_KEY not in dp:
                    dp[FACTCHECK_RESULTS_KEY] = [-1] * len(presuppositions)
                for response in grouped_responses:
                    idx = int(response.metadata['index'])
                    dp[FACTCHECK_RESULTS_KEY][idx] = self.TemplateClass.ResponseClass.model_validate_plain_text(extract_gemini_response_text(response.response)).get()
                    save_gemini_thinking_trace(dp, FACTCHECK_RESULTS_KEY, response.response, index=idx)
            if self.pipeline == 'statement':
                assert all(dp[FACTCHECK_RESULTS_KEY][i] != -1 for dp in dataset.values() for i in range(len(self._get_statements(dp))))
            elif check_gold:
                assert all(dp[FACTCHECK_RESULTS_KEY][i] != -1 for dp in dataset.values() for i in range(len(dp["presuppositions"])))
            else:
                assert all(dp[FACTCHECK_RESULTS_KEY][i] != -1 for dp in dataset.values() for i in range(len(dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY]["presuppositions"] if isinstance(dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY], dict) else dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY])))
        with open(job.save_to, 'w') as f:
            for dp in dataset.values():
                f.write(json.dumps(dp) + '\n')

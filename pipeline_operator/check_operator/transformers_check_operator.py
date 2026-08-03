from .check_operator import CheckOperator, register
import torch
from typing import Dict
from constant.constant import (
    FACTCHECK_RESULTS_KEY,
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_CONVERTED_STATEMENT_KEY
)
from data_gen.template import (
    LLMCheckTemplate,
    FeedbackActionTemplate,
    get_template_cls
)
from utils.transformers_utils import run_transformers_model
from utils.RAG_utils import get_passages

@register()
class TransformersCheckOperator(CheckOperator):
    model_name: str
    device: str | torch.DeviceObjType
    dtype: str
    enable_thinking: bool
    
    def __init__(
        self,
        model_name: str,
        device: str | torch.DeviceObjType,
        dtype: str,
        enable_thinking: bool,
        pipeline: str,
        template_class: str = None
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_thinking = enable_thinking
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

    @torch.inference_mode()
    def check(self, dp: Dict[str, any], source: str, check_gold: bool, **kwargs) -> Dict[str, any]:
        presuppositions = self._get_statements(dp) if self.pipeline == 'statement' else self._get_presuppositions(dp, check_gold=check_gold)
        if self.pipeline == 'feedback_action':
            passages = get_passages(
                dp,
                query='; '.join(presuppositions),
                source=source,
                instruction='Given a list of statements, retrieve relevant passages that validate or refute the statements',
                **kwargs
            )
            prompt = self.TemplateClass(
                question=dp['question'],
                model_detected_presuppositions=presuppositions,
                few_shot_data=dp["few_shot_data"],
                passages=passages,
                **kwargs
            ).generate()
            pred = run_transformers_model(
                model_name=self.model_name,
                messages=prompt,
                max_new_tokens=128,
                enable_thinking=self.enable_thinking,
                dtype=self.dtype,
                device=self.device
            )
            dp[self.TemplateClass.answer_key] = self.TemplateClass.ResponseClass.model_validate_plain_text(pred).get()
            return dp

        preds = []
        for i, presupposition in enumerate(presuppositions):
            passages = get_passages(
                dp,
                query=presupposition,
                source=source,
                instruction='Given a statement, retrieve relevant passages that validate or refute the statement',
                **kwargs
            )
            prompt = self.TemplateClass(
                model_detected_presupposition=presupposition,
                few_shot_data=dp["few_shot_data"],
                passages=passages,
                doctor_suggestion=dp["doctor_suggestion"][i] if check_gold and "doctor_suggestion" in dp else False,
                **kwargs
            ).generate()
            pred = run_transformers_model(
                model_name=self.model_name,
                messages=prompt,
                max_new_tokens=36,
                enable_thinking=self.enable_thinking,
                dtype=self.dtype,
                device=self.device
            )
            pred = self.TemplateClass.ResponseClass.model_validate_plain_text(pred).get()
            preds.append(pred)
        dp[FACTCHECK_RESULTS_KEY] = preds
        return dp

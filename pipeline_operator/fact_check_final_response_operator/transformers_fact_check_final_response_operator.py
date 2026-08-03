from .fact_check_final_response_operator import FactCheckFinalResponseOperator, register
import torch
from data_gen.template import (
    FactCheckFinalAnswerTemplate,
    FinalAnswerTemplate,
    FactCheckFPInterpretationTemplate
)
from typing import Dict, Any
from response import FinalAnswerResponse
from constant.constant import (
    MODEL_FP_IDENTIFICATION_KEY,
    MODEL_FINAL_ANSWER_KEY,
    FACTCHECK_RESULTS_KEY,
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_FEEDBACK_ACTION_KEY,
    MODEL_EXTRACTED_LOGICAL_FORMS_KEY,
    MODEL_CONVERTED_STATEMENT_KEY
)
from utils.transformers_utils import run_transformers_model
from utils.RAG_utils import get_passages

@register()
class TransformersFactCheckFinalResponseOperator(FactCheckFinalResponseOperator):
    def __init__(
        self,
        model_name: str,
        device: str | torch.DeviceObjType,
        dtype: str,
        enable_thinking: bool,
        pipeline: str = False,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_thinking = enable_thinking
        self.pipeline = pipeline
        if pipeline in ['fact_check', 'statement']:
            self.TemplateClass = FactCheckFinalAnswerTemplate
        elif pipeline == 'feedback_action':
            self.TemplateClass = FinalAnswerTemplate
        elif pipeline == 'interpretation':
            self.TemplateClass = FactCheckFPInterpretationTemplate
        else:
            raise ValueError(f"Unknown pipeline {pipeline}")
        
    @torch.inference_mode()
    def respond(self, dp: Dict[str, Any], source: str, **kwargs) -> Dict[str, Any]:
        question = dp['question']
        passages = get_passages(dp, source=source, **kwargs)
        prompt = self.TemplateClass(
            question=question,
            few_shot_data=dp["few_shot_data"],
            factcheck_results=dp[FACTCHECK_RESULTS_KEY] if FACTCHECK_RESULTS_KEY in dp else None,
            model_FP_identification=dp[MODEL_FP_IDENTIFICATION_KEY] if MODEL_FP_IDENTIFICATION_KEY in dp else None,
            model_extracted_logical_form=dp[MODEL_EXTRACTED_LOGICAL_FORMS_KEY] if MODEL_EXTRACTED_LOGICAL_FORMS_KEY in dp else None,
            model_detected_presuppositions=[dp[MODEL_CONVERTED_STATEMENT_KEY]] if self.pipeline == 'statement' else dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY] if MODEL_DETECTED_PRESUPPOSITIONS_KEY in dp else None,
            model_feedback_action=dp[MODEL_FEEDBACK_ACTION_KEY] if MODEL_FEEDBACK_ACTION_KEY in dp else None,
            passages=passages,
            **kwargs
        ).generate()
        pred = run_transformers_model(
            model_name=self.model_name,
            messages=prompt,
            enable_thinking=self.enable_thinking,
            dtype=self.dtype,
            device=self.device
        )
        pred = FinalAnswerResponse.model_validate_plain_text(pred).get()
        dp[MODEL_FINAL_ANSWER_KEY] = pred
        return dp
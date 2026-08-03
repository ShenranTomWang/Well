from .question_to_statement_operator import QuestionToStatementOperator, register
import torch
from typing import Dict, Any
from data_gen.template import QuestionToStatementTemplate
from utils.transformers_utils import run_transformers_model
from utils.RAG_utils import get_passages

@register()
class TransformersQuestionToStatementOperator(QuestionToStatementOperator):
    model_name: str
    device: str | torch.DeviceObjType
    dtype: str
    enable_thinking: bool

    def __init__(self, model_name: str, device: str | torch.DeviceObjType, dtype: str, enable_thinking: bool):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_thinking = enable_thinking

    @torch.inference_mode()
    def extract(self, dp: Dict[str, Any], source: str, **kwargs) -> Dict[str, Any]:
        question = dp['question']
        passages = get_passages(dp, source=source, **kwargs)
        prompt = QuestionToStatementTemplate(
            question=question,
            few_shot_data=dp["few_shot_data"],
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
        pred = QuestionToStatementTemplate.ResponseClass.model_validate_plain_text(pred).get()
        dp[QuestionToStatementTemplate.answer_key] = pred
        return dp

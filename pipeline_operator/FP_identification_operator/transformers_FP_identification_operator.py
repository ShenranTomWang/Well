from utils.RAG_utils import get_passages
from .FP_identification_operator import FPIdentificationOperator, register
import torch
from typing import Dict, Any
from data_gen.template import FPIdentificationTemplate
from utils.transformers_utils import run_transformers_model

@register()
class TransformersFPIdentificationOperator(FPIdentificationOperator):
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
    def extract(self, dp: Dict[str, Any], source: str,  **kwargs) -> Dict[str, Any]:
        passages = get_passages(dp, source=source, **kwargs)
        prompt = FPIdentificationTemplate(
            question=dp['question'],
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
        pred = FPIdentificationTemplate.ResponseClass.model_validate_plain_text(pred).get()
        dp[FPIdentificationTemplate.answer_key] = pred
        return dp

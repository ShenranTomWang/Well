from .presupposition_extraction_operator import PresuppositionExtractionOperator, register
import torch
from typing import Dict
from response import PresuppositionExtractionResponse
from constant.constant import MODEL_DETECTED_PRESUPPOSITIONS_KEY
from data_gen.template import PresuppositionExtractionTemplate
from utils.transformers_utils import run_transformers_model
from utils.RAG_utils import get_passages

@register()
class TransformersPresuppositionExtractionOperator(PresuppositionExtractionOperator):
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
    def extract(self, dp: Dict[str, any], source: str, **kwargs) -> Dict[str, any]:
        question = dp['question']
        passages = get_passages(dp, source=source, **kwargs)
        prompt = PresuppositionExtractionTemplate(
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
        pred = PresuppositionExtractionResponse.model_validate_plain_text(pred).get()
        dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY] = pred
        return dp
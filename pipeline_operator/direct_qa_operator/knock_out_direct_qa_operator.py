from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer

from .direct_qa_operator import DirectQAOperator, register
from data_gen.template import DirectQATemplate
from FAITH.identify_heads import attention_o_proj_layers, find_token_range, get_head_dim, load_generation_model, untuple
from utils.RAG_utils import get_passages
from FAITH.utils.nethook import TraceDict


DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    model_name: str,
    device: str | torch.DeviceObjType,
    dtype: str | torch.dtype,
    enable_thinking: bool,
):
    model = load_generation_model(model_name, dtype, device)
    if enable_thinking:
        tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@register()
class KnockOutDirectQAOperator(DirectQAOperator):
    def __init__(
        self,
        model_name: str,
        heads: List[Tuple[int, int]],
        device: str | torch.DeviceObjType = "auto",
        dtype: str | torch.dtype = "auto",
        enable_thinking: bool = False,
        max_new_tokens: int = 2048,
        limit_input_length: int | None = 3700,
        position_mode: str = "question_end",
        knockout_generated_tokens: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.limit_input_length = limit_input_length
        self.position_mode = position_mode
        self.knockout_generated_tokens = knockout_generated_tokens

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            dtype=dtype,
            enable_thinking=enable_thinking,
        )
        self.layers = attention_o_proj_layers(self.model)
        self.head_dim = get_head_dim(self.model, self.layers[0])
        self.layer_to_heads = defaultdict(list)
        for layer_idx, head_idx in heads:
            self.layer_to_heads[self.layers[layer_idx]].append(head_idx)
        self.knockout_layers = list(self.layer_to_heads.keys())

    def _tokenize_chat_batch(self, messages_batch: List[List[Dict[str, str]]]) -> Dict[str, torch.Tensor]:
        prompts = []
        for messages in messages_batch:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **({"enable_thinking": self.enable_thinking} if self.enable_thinking else {}),
            )
            prompts.append(prompt)

        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                truncation=self.limit_input_length is not None,
                max_length=self.limit_input_length,
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side
        return inputs

    def _tokenize_chat(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        return self._tokenize_chat_batch([messages])["input_ids"]

    def _question_position(self, input_ids: torch.Tensor, question: str) -> int:
        if self.position_mode == "last_prompt_token":
            return input_ids.shape[1] - 1
        try:
            _, end_of_question = find_token_range(self.tokenizer, input_ids[0], question)
            return max(0, end_of_question - 2)
        except Exception:
            return input_ids.shape[1] - 1

    def _generate_batch_with_knockout(
        self,
        messages_batch: List[List[Dict[str, str]]],
        questions: List[str],
    ) -> List[str]:
        inputs = self._tokenize_chat_batch(messages_batch)
        input_ids = inputs["input_ids"]
        positions = [
            self._question_position(input_ids[i : i + 1], question)
            for i, question in enumerate(questions)
        ]
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

        def intervene_head(x, layer):
            h = untuple(x)
            for head in self.layer_to_heads[layer]:
                dim_start = head * self.head_dim
                dim_end = (head + 1) * self.head_dim
                if self.position_mode == "all_prompt_tokens" and h.shape[1] > 1:
                    h[:, :, dim_start:dim_end] = 0
                elif h.shape[1] > 1:
                    for batch_idx, position in enumerate(positions):
                        h[batch_idx, position, dim_start:dim_end] = 0
                elif self.knockout_generated_tokens:
                    h[:, :, dim_start:dim_end] = 0
            return x

        with torch.inference_mode(), TraceDict(
            self.model,
            self.knockout_layers,
            retain_output=False,
            edit_input=intervene_head,
        ):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def _generate_with_knockout(self, messages: List[Dict[str, str]], question: str) -> str:
        return self._generate_batch_with_knockout([messages], [question])[0]

    @torch.inference_mode()
    def qa_batch(self, dps: List[Dict[str, Any]], source: str, **kwargs) -> List[Dict[str, Any]]:
        prompts = []
        questions = []
        for dp in dps:
            question = dp["question"]
            dp.setdefault("few_shot_data", [])
            passages = get_passages(dp=dp, source=source, **kwargs)
            prompts.append(
                DirectQATemplate(
                    question=question,
                    few_shot_data=dp["few_shot_data"],
                    passages=passages,
                    **kwargs,
                ).generate()
            )
            questions.append(question)

        predictions = self._generate_batch_with_knockout(prompts, questions)
        knockout_heads = [
            {"layer": layer, "head": head}
            for layer, heads in self.layer_to_heads.items()
            for head in heads
        ]
        for dp, pred in zip(dps, predictions):
            dp[DirectQATemplate.answer_key] = DirectQATemplate.ResponseClass.model_validate_plain_text(pred).get()
            dp["knockout_heads"] = knockout_heads
        return dps

    @torch.inference_mode()
    def qa(self, dp: Dict[str, Any], source: str, **kwargs) -> Dict[str, Any]:
        return self.qa_batch([dp], source=source, **kwargs)[0]

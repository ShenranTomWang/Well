import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, logging
from typing import List, Any
logging.set_verbosity_error()
logging.disable_progress_bar()
__CACHE__ = {}

@torch.inference_mode()
def run_transformers_model(
    model_name: str,
    messages: List[str],
    enable_thinking: bool = False,
    max_new_tokens: int = 2048,
    limit_input_length: int | None = 3700,
    dtype: torch.dtype = 'auto',
    device: torch.DeviceObjType = 'auto',
    **kwargs
) -> Any:
    """
    Run the transformers model given the input messages. Return the parsed response.
    This does not call .get()
    """
    if model_name in __CACHE__:
        model, tokenizer = __CACHE__[model_name]
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, dtype=dtype)
        if enable_thinking:
            tokenizer = AutoProcessor.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        __CACHE__[model_name] = (model, tokenizer)

    if enable_thinking:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
        if not isinstance(prompt, str):
            prompt = tokenizer.decode(prompt.input_ids)
        prompt = tokenizer(text=prompt, return_tensors='pt').input_ids.to(model.device)
    else:
        prompt = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).input_ids.to(model.device)
    prompt = prompt[:, :limit_input_length] if limit_input_length is not None else prompt
    output_ids = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)[:, prompt.shape[1]:]
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


@torch.inference_mode()
def run_transformers_model_batch(
    model_name: str,
    messages_batch: List[List[dict]],
    enable_thinking: bool = False,
    max_new_tokens: int = 2048,
    limit_input_length: int | None = 3700,
    dtype: torch.dtype = "auto",
    device: torch.DeviceObjType = "auto",
) -> List[str]:
    if model_name in __CACHE__:
        model, tokenizer = __CACHE__[model_name]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, dtype=dtype
        )
        tokenizer = (
            AutoProcessor.from_pretrained(model_name)
            if enable_thinking
            else AutoTokenizer.from_pretrained(model_name)
        )
        __CACHE__[model_name] = (model, tokenizer)

    padding_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    padding_tokenizer.padding_side = "left"
    if padding_tokenizer.pad_token_id is None:
        padding_tokenizer.pad_token = padding_tokenizer.eos_token

    prompts = []
    for messages in messages_batch:
        template_kwargs = {"enable_thinking": True} if enable_thinking else {}
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        prompts.append(prompt)

    model_inputs = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=limit_input_length is not None,
        max_length=limit_input_length,
    ).to(model.device)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )[:, model_inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

"""Run direct QA with a PEFT model fine-tuned by FalseQA/train.py."""

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from constant.constant import MODEL_FINAL_ANSWER_KEY
from response import FinalAnswerResponse
from utils.RAG_utils import get_passages


def build_prompt(question: str, passages: list[str]) -> list[dict[str, str]]:
    extra = (
        "\n\nFor the following question, you are given these additional information: "
        f"\nAdditional Information: {' ||| '.join(passages)}"
        if passages
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answer questions based on your knowledge.\n"
                "The user will ask a question, and you need to provide the answer to that question."
                + extra
            ),
        },
        {"role": "user", "content": question},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--RAG", choices=["0", "all", "4"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--RAG_device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--limit_input_length", type=int, default=3700)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be at least 1")
    if args.limit_input_length < 1:
        parser.error("--limit_input_length must be at least 1")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not (args.model_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{args.model_path} is not a saved PEFT adapter directory"
        )

    with args.dataset_path.open() as file:
        data = [dict(json.loads(line), few_shot_data=[]) for line in file]

    source = {"0": "no_passages", "all": "use_passages", "4": "use_RAG"}[
        args.RAG
    ]
    prompts = [
        build_prompt(
            datapoint["question"],
            get_passages(datapoint, source=source, RAG_device=args.RAG_device),
        )
        for datapoint in data
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=args.dtype,
    )
    model.eval()

    for start in range(0, len(data), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.limit_input_length,
        ).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )[:, inputs.input_ids.shape[1] :]
        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for datapoint, answer in zip(
            data[start : start + args.batch_size], answers
        ):
            datapoint[MODEL_FINAL_ANSWER_KEY] = (
                FinalAnswerResponse.model_validate_plain_text(answer).get()
            )
        print(
            f"FalseQA direct qa {args.RAG}: "
            f"{min(start + args.batch_size, len(data))}/{len(data)}",
            flush=True,
        )

    output_path = args.output_dir / f"RAG={args.RAG}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for datapoint in data:
            file.write(json.dumps(datapoint) + "\n")


if __name__ == "__main__":
    main()

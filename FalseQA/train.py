"""PEFT fine-tuning on mixed FPQ/TPQ examples with ARC-DA replay.

FPQ and TPQ records are mixed into the primary training dataset. Each primary
batch is augmented with a periodically refreshed ARC-DA replay batch.
"""

import argparse
import inspect
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    __version__ as transformers_version,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a causal LM on mixed FPQ/TPQ data with ARC-DA replay."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--fpq_path", type=Path, required=True)
    parser.add_argument("--tpq_path", type=Path, required=True)
    parser.add_argument(
        "--arc_da_path",
        type=Path,
        default=Path(__file__).parent.parent / "data_gen/ARC_DA/train.csv",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--question_field", default="question")
    parser.add_argument("--answer_field", default="answer")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--system_prompt", default="You are a helpful assistant that answer questions based on your knowledge.\nThe user will ask a question, and you need to provide the answer to that question.")
    parser.add_argument("--prompt_template", default="{question}\n")
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Format question/answer as user/assistant messages using the tokenizer chat template.",
    )
    parser.add_argument(
        "--update_gate",
        type=int,
        default=30,
        help="Reuse an ARC-DA replay batch for this many collator calls.",
    )
    parser.add_argument(
        "--arc_replay_batch_size",
        type=int,
        default=None,
        help="ARC examples appended per mixed-data batch (default: train batch size).",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["all-linear"],
        help="PEFT target module names; 'all-linear' works across most causal LMs.",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", nargs="*", default=[])
    args = parser.parse_args()

    if args.update_gate <= 0:
        parser.error("--update_gate must be positive")
    if args.arc_replay_batch_size is not None and args.arc_replay_batch_size <= 0:
        parser.error("--arc_replay_batch_size must be positive")
    if args.load_in_4bit and args.load_in_8bit:
        parser.error("choose only one of --load_in_4bit and --load_in_8bit")
    if args.bf16 and args.fp16:
        parser.error("choose only one of --bf16 and --fp16")
    try:
        args.prompt_template.format(question="example")
    except (KeyError, ValueError) as error:
        parser.error(f"--prompt_template must be format-able with {{question}}: {error}")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_columns(dataset: Dataset, path: Path, required_fields: set[str]) -> None:
    missing = required_fields - set(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(sorted(missing))}")


def load_records(path: Path) -> Dataset:
    if path.suffix.lower() == ".csv":
        return load_dataset("csv", data_files=str(path), split="train")
    if path.suffix.lower() in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path), split="train")
    raise ValueError(f"Unsupported dataset format for {path}; use .csv, .json, or .jsonl")


def tokenize_dataset(dataset: Dataset, tokenizer, args: argparse.Namespace, split: str) -> Dataset:
    eos = tokenizer.eos_token or ""

    def tokenize_record(record):
        question = str(record[args.question_field])
        answer = str(record[args.answer_field])

        if args.use_chat_template:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError("--use_chat_template was set, but this tokenizer has no chat template")
            prompt_messages = []
            if args.system_prompt:
                prompt_messages.append({"role": "system", "content": args.system_prompt})
            prompt_messages.append({"role": "user", "content": question})
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                prompt_messages + [{"role": "assistant", "content": answer}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt = args.prompt_template.format(question=question)
            if args.system_prompt:
                prompt = f"{args.system_prompt}\n\n{prompt}"
            full_text = f"{prompt}{answer}{eos}"

        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=args.max_length)[
            "input_ids"
        ]
        encoded = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=args.max_length)
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        if all(label == -100 for label in labels):
            raise ValueError(
                "An example has no answer tokens after truncation; increase --max_length or shorten the prompt."
            )
        encoded["labels"] = labels
        return encoded

    return dataset.map(
        tokenize_record,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split} examples",
    )


class PeriodicARCReplayCollator:
    def __init__(self, tokenizer, arc_dataset, replay_batch_size, update_gate, seed):
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
        )
        self.arc_dataset = arc_dataset
        self.replay_batch_size = replay_batch_size
        self.update_gate = update_gate
        self.rng = random.Random(seed)
        self.indices = list(range(len(arc_dataset)))
        self.rng.shuffle(self.indices)
        self.offset = 0
        self.calls = 0
        self.replay_features = None

    def _next_replay_batch(self):
        selected = []
        while len(selected) < self.replay_batch_size:
            if self.offset == len(self.indices):
                self.rng.shuffle(self.indices)
                self.offset = 0
            take = min(self.replay_batch_size - len(selected), len(self.indices) - self.offset)
            selected.extend(self.indices[self.offset : self.offset + take])
            self.offset += take
        return [self.arc_dataset[index] for index in selected]

    def __call__(self, features):
        if self.calls % self.update_gate == 0:
            self.replay_features = self._next_replay_batch()
        self.calls += 1
        return self.base_collator(features + self.replay_features)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)
    if torch.cuda.device_count() > 1 and world_size == 1:
        raise RuntimeError(
            "Multiple GPUs are visible, but this process was not launched with DDP. "
            "Use `torchrun --standalone --nproc_per_node=<gpu_count> FalseQA/train.py ...` "
            "or restrict CUDA_VISIBLE_DEVICES to one GPU. Legacy DataParallel is unsupported."
        )

    fpq = load_records(args.fpq_path)
    tpq = load_records(args.tpq_path)
    arc = load_records(args.arc_da_path)
    validate_columns(fpq, args.fpq_path, {args.question_field, args.answer_field})
    validate_columns(tpq, args.tpq_path, {args.question_field, args.answer_field})
    validate_columns(
        arc, args.arc_da_path, {args.question_field, args.answer_field, args.label_field}
    )
    if len(fpq) == 0 or len(tpq) == 0 or len(arc) == 0:
        raise ValueError("FPQ, TPQ, and ARC-DA datasets must all contain at least one record")
    for dataset, path in ((fpq, args.fpq_path), (tpq, args.tpq_path), (arc, args.arc_da_path)):
        if not any(str(answer).strip() for answer in dataset[args.answer_field]):
            raise ValueError(f"{path} has no populated {args.answer_field!r} values")
    # The split builders may attach different metadata to FPQs and TPQs.  It is
    # irrelevant to SFT and would otherwise prevent datasets from concatenating.
    training_columns = [args.question_field, args.answer_field]
    fpq = fpq.select_columns(training_columns)
    tpq = tpq.select_columns(training_columns)
    train_dataset = concatenate_datasets([fpq, tpq]).shuffle(seed=args.seed)
    print(f"Primary training examples: {len(fpq)} FPQ + {len(tpq)} TPQ")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args, "mixed FPQ/TPQ")
    arc = tokenize_dataset(arc, tokenizer, args, "ARC-DA")

    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
    }
    transformers_major_version = int(transformers_version.split(".", maxsplit=1)[0])
    dtype_argument = "dtype" if transformers_major_version >= 5 else "torch_dtype"
    if args.bf16:
        model_kwargs[dtype_argument] = torch.bfloat16
    elif args.fp16:
        model_kwargs[dtype_argument] = torch.float16
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        # Under DDP, each process must load its quantized model replica onto its
        # own GPU.  ``device_map="auto"`` may shard a model across devices and
        # is therefore rejected by Accelerate in distributed training.
        model_kwargs["device_map"] = {"": local_rank} if world_size > 1 else "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    # Some architectures (including Qwen) intentionally pad their embedding
    # vocabulary beyond the tokenizer size.  Never shrink that table: special
    # token IDs may occupy the padded range, and shrinking makes them invalid.
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        model.enable_input_require_grads()

    target_modules = args.lora_target_modules
    if target_modules == ["all-linear"]:
        # ``all-linear`` is a special PEFT selector, not a literal module name.
        # PEFT recognizes it only when supplied as a string.
        target_modules = "all-linear"

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=args.seed,
        local_rank=local_rank if world_size > 1 else -1,
        ddp_find_unused_parameters=False,
    )
    # Transformers 5 renamed this setting; supporting both also keeps the script
    # usable on the recent 4.x releases.
    training_parameters = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs["eval_strategy" if "eval_strategy" in training_parameters else "evaluation_strategy"] = "no"
    training_args = TrainingArguments(**training_kwargs)

    replay_batch_size = args.arc_replay_batch_size or args.per_device_train_batch_size
    collator = PeriodicARCReplayCollator(
        tokenizer, arc, replay_batch_size, args.update_gate, args.seed
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_parameters else "tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        barrier_kwargs = {"device_ids": [torch.cuda.current_device()]} if torch.cuda.is_available() else {}
        torch.distributed.barrier(**barrier_kwargs)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

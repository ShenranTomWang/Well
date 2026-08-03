"""Standalone false-presupposition identification and interpretation pipeline."""

import argparse
import json
import time
from pathlib import Path
from constant.constant import MODEL_FP_IDENTIFICATION_KEY, MODEL_FINAL_ANSWER_KEY
from response import FPIdentificationResponse, FinalAnswerResponse
from utils.RAG_utils import get_passages
from utils.gemini_utils import (
    call_gemini_one_by_one_api,
    checkback,
    extract_gemini_response_text,
    message2gemini_request,
    save_gemini_thinking_trace,
    submit_gemini_job,
)
from utils.transformers_utils import run_transformers_model_batch


class FPIdentificationTemplate:
    ResponseClass = FPIdentificationResponse

    def __init__(self, question, passages, few_shot_data, **kw):
        self.question = question
        self.passages = passages
        self.few_shot_data = []
        for dp in few_shot_data:
            answer = "No" if dp["is_normal"] else "Yes"
            self.few_shot_data.append(
                f"user: Input: {dp['question']}\nQuestion: Does the input contain any false assumptions?\nassistant: {answer}"
            )

    def generate(self):
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that helps identify false assumptions.
                    {"Use the information from the evidence to help you identify the false assumption." if self.passages else ""}
                    Output Yes if the question has false assumptions; otherwise, output No.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                """,
            },
            {
                "role": "user",
                "content": (
                    f"Input: {self.question}\n"
                    "Question: Does the input contain any false assumptions?\n"
                    f"Evidence: {' ||| '.join(self.passages)}"
                    if self.passages
                    else ""
                ),
            },
        ]


class FactCheckFPInterpretationTemplate:
    ResponseClass = FinalAnswerResponse

    def __init__(self, question, model_FP_identification, few_shot_data, **kw):
        self.question = question
        self.result = model_FP_identification
        self.few_shot_data = [
            f"user: Question: {dp['question']}\nassistant: {dp['answer']}"
            for dp in few_shot_data
            if not dp["is_normal"]
        ]

    def generate(self):
        has_fp = self.result == 0
        hint = (
            """
            You will be provided with a question that contains at least 1 false assumption.
            Your task is to help me understand what are the false assumptions.
            Write an explanation to pinpoint the false assumptions.
        """
            if has_fp
            else ""
        )
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that answers questions.                     The user will ask a question, and you need to provide the answer to that question.
                    {hint}
                    {"Below are some examples to help you understand the task." if self.few_shot_data and has_fp else ""}
                    {"\n\n".join(self.few_shot_data) if self.few_shot_data and has_fp else ""}
                """,
            },
            {"role": "user", "content": f"Question: {self.question}\n"},
        ]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")


def wait(job, a):
    while True:
        try:
            return checkback(job.name)
        except RuntimeError as e:
            if "not completed yet" not in str(e):
                raise
            print(e, flush=True)
            time.sleep(a.poll_interval)


def infer(data, prompt, parser, key, a, out, web=False):
    if a.backend == "transformers":
        for start in range(0, len(data), a.batch_size):
            batch = data[start : start + a.batch_size]
            texts = run_transformers_model_batch(
                a.model_name,
                [prompt(dp) for dp in batch],
                device=a.device,
                dtype=a.dtype,
                enable_thinking=a.thinking == "true",
            )
            for dp, text in zip(batch, texts):
                dp[key] = parser(text).get()
    elif a.batching == "false":
        for dp in data:
            r = call_gemini_one_by_one_api(
                prompt(dp),
                model=a.model_name,
                web_search=web,
                thinking_level=a.thinking_level if a.thinking == "true" else None,
            )
            dp[key] = parser(extract_gemini_response_text(r)).get()
            save_gemini_thinking_trace(dp, key, r)
    else:
        job = submit_gemini_job(
            [
                message2gemini_request(
                    {"id": str(dp["id"])},
                    prompt(dp),
                    a.model_name,
                    web_search=web,
                    thinking_level=a.thinking_level if a.thinking == "true" else None,
                )
                for dp in data
            ],
            a.model_name,
        )
        a.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = a.cache_dir / f"{job.name.replace('/', '_')}.json"
        cache.write_text(json.dumps({"job": job.name, "out": str(out)}))
        by = {str(dp["id"]): dp for dp in data}
        for item in wait(job, a):
            dp = by[item.metadata["id"]]
            dp[key] = parser(extract_gemini_response_text(item.response)).get()
            save_gemini_thinking_trace(dp, key, item.response)
        cache.unlink(missing_ok=True)
    save(out, data)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["transformers", "gemini"], required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--identification_output_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--RAG_device", default="cuda")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--RAG", choices=["0", "all", "4", "web"], required=True)
    p.add_argument("--thinking", choices=["true", "false"], required=True)
    p.add_argument("--batching", choices=["true", "false"], default="false")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--cache_dir", type=Path, default=Path("tmp/fp_identification"))
    p.add_argument("--poll_interval", type=int, default=60)
    p.add_argument("--thinking_level")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch_size must be at least 1")
    if a.backend == "transformers" and a.batching == "false":
        a.batch_size = 1
    if a.RAG == "web" and a.backend != "gemini":
        p.error("--RAG web is only supported with --backend gemini")
    if a.backend == "gemini" and a.thinking == "true" and not a.thinking_level:
        p.error("--thinking_level is required for Gemini thinking")
    original = [json.loads(x) for x in open(a.dataset_path)]
    settings = [
        {
            "0": ("0", "no_passages", False),
            "all": ("all", "use_passages", False),
            "4": ("4", "use_RAG", False),
            "web": ("web", "no_passages", True),
        }[a.RAG]
    ]
    for rag, source, web in settings:
        data = [dict(dp) for dp in original]
        identified = a.identification_output_dir / f"RAG={rag}.jsonl"
        data = infer(
            data,
            lambda dp: FPIdentificationTemplate(
                question=dp["question"],
                few_shot_data=dp["few_shot_data"],
                passages=get_passages(dp, source=source, RAG_device=a.RAG_device),
            ).generate(),
            FPIdentificationTemplate.ResponseClass.model_validate_plain_text,
            MODEL_FP_IDENTIFICATION_KEY,
            a,
            identified,
            web,
        )
        infer(
            data,
            lambda dp: FactCheckFPInterpretationTemplate(
                question=dp["question"],
                few_shot_data=dp["few_shot_data"],
                model_FP_identification=dp[MODEL_FP_IDENTIFICATION_KEY],
            ).generate(),
            FactCheckFPInterpretationTemplate.ResponseClass.model_validate_plain_text,
            MODEL_FINAL_ANSWER_KEY,
            a,
            a.output_dir / f"RAG={rag}.jsonl",
        )


if __name__ == "__main__":
    main()

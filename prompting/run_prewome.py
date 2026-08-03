"""Standalone PreWoMe extraction/feedback/final-answer research pipeline."""

import argparse
import json
import time
from pathlib import Path
from constant.constant import (
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_FEEDBACK_ACTION_KEY,
    MODEL_FINAL_ANSWER_KEY,
)
from response import (
    PresuppositionExtractionResponse,
    FeedbackActionResponse,
    FinalAnswerResponse,
)
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


class PresuppositionExtractionTemplate:
    ResponseClass = PresuppositionExtractionResponse

    def __init__(self, question, few_shot_data, passages=None, **kw):
        self.question = question
        self.passages = passages or []
        self.few_shot_data = [
            f"user: {dp['question']}\nassistant: {'\n'.join(dp['presuppositions'])}\n"
            for dp in few_shot_data
        ]

    def generate(self):
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that analyzes the given question.
                    Your task is to extract presuppositions in the given question.
                    Notice that the presuppositions in a question could be true or false, and may be explicit or implicit.
                    There could be multiple presuppositions in a question, but there will always be at least one presupposition in the question.
                    Format your response as a list of presuppositions, separated by newlines.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                    {f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}" if len(self.passages) > 0 else ""}
                """,
            },
            {"role": "user", "content": self.question},
        ]


class FeedbackActionTemplate:
    ResponseClass = FeedbackActionResponse

    def __init__(
        self, question, model_detected_presuppositions, passages, few_shot_data, **kw
    ):
        self.question = question
        self.claims = model_detected_presuppositions
        self.passages = passages
        self.few_shot_data = []
        for dp in few_shot_data:
            presuppositions = list(dp["presuppositions"]) + [
                "There is a clear and single answer to the question."
            ]
            if dp["is_normal"]:
                feedback = (
                    "The question is valid and does not contain false presuppositions."
                )
                action = "Answer the question directly based on the presuppositions."
            else:
                joined = "; ".join(presuppositions)
                feedback = f"The question contains false presuppositions that {joined}."
                action = f"Correct the false assumptions that {joined} and respond based on the corrected assumption."
            self.few_shot_data.append(
                f"user: Question: {dp['question']}\nPresuppositions: {'; '.join(presuppositions)}\nassistant: Feedback: {feedback}\nAction: {action}"
            )

    def generate(self):
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that provides feedback on the question and a guideline for answering the question.
                    You will be given a question and the assumptions that are implicit in the question.
                    Your task is to first, provide feedback on the question based on whether it contains any false assumptions and then provide a guideline for answering the question.
                    Separate your feedback and action with a newline, and format your response as:
                    Feedback: <your feedback>\nAction: <your action>.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                    {f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}" if self.passages else ""}
                """,
            },
            {
                "role": "user",
                "content": f"Question: {self.question}\nPresuppositions: {'; '.join(self.claims)}; There is a clear and single answer to the question",
            },
        ]


class FinalAnswerTemplate:
    ResponseClass = FinalAnswerResponse

    def __init__(self, question, model_feedback_action, few_shot_data, **kw):
        self.question = question
        self.feedback = model_feedback_action
        self.few_shot_data = []
        for dp in few_shot_data:
            if dp["is_normal"]:
                feedback = (
                    "The question is valid and does not contain false presuppositions."
                )
                action = "Answer the question directly based on the presuppositions."
            else:
                joined = "; ".join(dp["presuppositions"])
                feedback = f"The question contains false presuppositions that {joined}."
                action = f"Correct the false assumptions that {joined} and respond based on the corrected assumption."
            self.few_shot_data.append(
                f"user: Question: {dp['question']}\nFeedback: {feedback}\nAction: {action}\nassistant: {dp['answer']}"
            )

    def generate(self):
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that provides a response to a question based on the feedback and action guideline.
                    You will be given a question and feedback and action guideline on how to answer the question.
                    Your task is to provide a final answer to the question based on the feedback and action guideline.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                """,
            },
            {
                "role": "user",
                "content": f"Question: {self.question}\n{self.feedback}\n",
            },
        ]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")


def assumptions(dp):
    x = dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY]
    return x["presuppositions"] if isinstance(x, dict) else x


def wait(job, a):
    while True:
        try:
            return checkback(job.name)
        except RuntimeError as e:
            if "not completed yet" not in str(e):
                raise
            print(e, flush=True)
            time.sleep(a.poll_interval)


def infer(data, prompt, parser, key, a, out, web=False, max_tokens=None):
    if a.backend == "transformers":
        for start in range(0, len(data), a.batch_size):
            batch = data[start : start + a.batch_size]
            kw = {} if max_tokens is None else {"max_new_tokens": max_tokens}
            texts = run_transformers_model_batch(
                a.model_name,
                [prompt(dp) for dp in batch],
                device=a.device,
                dtype=a.dtype,
                enable_thinking=a.thinking == "true",
                **kw,
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
                    {"id": dp["id"]},
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
        by = {dp["id"]: dp for dp in data}
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
    p.add_argument("--dataset_path", type=Path)
    p.add_argument(
        "--presupposition_file",
        type=Path,
        help="Existing presupposition-extracted JSONL; skips extraction.",
    )
    p.add_argument(
        "--feedback_action_file",
        type=Path,
        help="Existing feedback/action JSONL; skips extraction and feedback/action generation.",
    )
    p.add_argument("--presupposition_output_dir", type=Path, required=True)
    p.add_argument("--feedback_action_output_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--RAG_device", default="cuda")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--RAG", choices=["0", "all", "4", "web"], required=True)
    p.add_argument("--thinking", choices=["true", "false"], required=True)
    p.add_argument("--batching", choices=["true", "false"], default="false")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--cache_dir", type=Path, default=Path("tmp/prewome"))
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
    feedback_action_exists = (
        a.feedback_action_file is not None and a.feedback_action_file.is_file()
    )
    presupposition_exists = (
        a.presupposition_file is not None and a.presupposition_file.is_file()
    )
    if feedback_action_exists:
        data = [json.loads(x) for x in open(a.feedback_action_file)]
    elif presupposition_exists:
        data = [json.loads(x) for x in open(a.presupposition_file)]
    else:
        if a.dataset_path is None:
            p.error("--dataset_path is required when no intermediate input file exists")
        data = [json.loads(x) for x in open(a.dataset_path)]
        data = infer(
            data,
            lambda dp: PresuppositionExtractionTemplate(
                question=dp["question"],
                few_shot_data=dp["few_shot_data"],
                passages=[],
            ).generate(),
            PresuppositionExtractionTemplate.ResponseClass.model_validate_plain_text,
            MODEL_DETECTED_PRESUPPOSITIONS_KEY,
            a,
            a.presupposition_output_dir / "presuppositions.jsonl",
        )
    settings = [
        {
            "0": ("0", "no_passages", False),
            "all": ("all", "use_passages", False),
            "4": ("4", "use_RAG", False),
            "web": ("web", "no_passages", True),
        }[a.RAG]
    ]
    suffix = "gemini_checked" if a.backend == "gemini" else "transformers_checked"
    for rag, source, web in settings:
        current = [dict(dp) for dp in data]
        if not feedback_action_exists:
            current = infer(
                current,
                lambda dp: FeedbackActionTemplate(
                    question=dp["question"],
                    few_shot_data=dp["few_shot_data"],
                    model_detected_presuppositions=assumptions(dp),
                    passages=get_passages(
                        dp,
                        query="; ".join(assumptions(dp)),
                        source=source,
                        RAG_device=a.RAG_device,
                        instruction="Given a list of statements, retrieve relevant passages that validate or refute the statements",
                    ),
                ).generate(),
                FeedbackActionTemplate.ResponseClass.model_validate_plain_text,
                MODEL_FEEDBACK_ACTION_KEY,
                a,
                a.feedback_action_output_dir / f"RAG={rag}_{suffix}.jsonl",
                web,
                128,
            )
        infer(
            current,
            lambda dp: FinalAnswerTemplate(
                question=dp["question"],
                few_shot_data=dp["few_shot_data"],
                model_feedback_action=dp[MODEL_FEEDBACK_ACTION_KEY],
            ).generate(),
            FinalAnswerTemplate.ResponseClass.model_validate_plain_text,
            MODEL_FINAL_ANSWER_KEY,
            a,
            a.output_dir / f"RAG={rag}_{suffix}.jsonl",
        )


if __name__ == "__main__":
    main()

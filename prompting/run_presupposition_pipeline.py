"""Standalone presupposition extraction/check/final-answer research pipeline."""

import argparse
import json
import os
import time
from pathlib import Path
from constant.constant import (
    FACTCHECK_RESULTS_KEY,
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_FINAL_ANSWER_KEY,
)
from response import (
    PresuppositionExtractionResponse,
    LLMCheckResponse,
    FinalAnswerResponse,
)
from minicheck.minicheck import MiniCheck
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
        rag = (
            f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}"
            if len(self.passages) > 0
            else ""
        )
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
                    {rag}
                """,
            },
            {"role": "user", "content": self.question},
        ]


class LLMCheckTemplate:
    ResponseClass = LLMCheckResponse

    def __init__(self, model_detected_presupposition, passages, few_shot_data, **kw):
        self.claim = model_detected_presupposition
        self.passages = passages
        self.few_shot_data = []
        for dp in few_shot_data:
            feedback = "true" if dp["is_normal"] else "false"
            self.few_shot_data.append(
                f"user: Presupposition: {dp['presuppositions'][0]}\nassistant: {feedback}"
            )

    def generate(self):
        rag = (
            f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}"
            if self.passages
            else ""
        )
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that fact-checks a presupposition.
                    You will be given a presupposition.
                    Your task is to determine whether it is true or false.
                    You should just return one word "true" or "false" as your answer, without any additional explanation.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                    {rag}
                """,
            },
            {"role": "user", "content": f"Presupposition: {self.claim}"},
        ]


class FactCheckFinalAnswerTemplate:
    ResponseClass = FinalAnswerResponse

    def __init__(
        self,
        question,
        factcheck_results,
        model_detected_presuppositions,
        few_shot_data,
        **kw,
    ):
        self.question = question
        self.results = factcheck_results
        self.claims = model_detected_presuppositions
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
        false = [self.claims[i] for i, x in enumerate(self.results) if x == 0]
        if false:
            joined = "; ".join(false)
            guideline = f"Feedback: The question contains false presuppositions that {joined}.\nAction: Correct the false assumptions that {joined} and respond based on the corrected assumption."
        else:
            guideline = "Feedback: The question is valid and does not contain false presuppositions.\nAction: Answer the question directly based on the presuppositions."
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
                "content": f"Question: {self.question}\n{guideline}\n",
            },
        ]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")


def claims(dp):
    x = dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY]
    return x["presuppositions"] if isinstance(x, dict) else x


def prompt_extract(dp):
    return PresuppositionExtractionTemplate(
        question=dp["question"], few_shot_data=dp["few_shot_data"], passages=[]
    ).generate()


def prompt_check(dp, claim, source, a):
    return LLMCheckTemplate(
        model_detected_presupposition=claim,
        few_shot_data=dp["few_shot_data"],
        passages=get_passages(
            dp,
            query=claim,
            source=source,
            RAG_device=a.RAG_device,
            instruction="Given a statement, retrieve relevant passages that validate or refute the statement",
        ),
    ).generate()


def prompt_final(dp):
    return FactCheckFinalAnswerTemplate(
        question=dp["question"],
        few_shot_data=dp["few_shot_data"],
        factcheck_results=dp[FACTCHECK_RESULTS_KEY],
        model_detected_presuppositions=claims(dp),
    ).generate()


def wait(job, a):
    while True:
        try:
            return checkback(job.name)
        except RuntimeError as e:
            if "not completed yet" not in str(e):
                raise
            print(e, flush=True)
            time.sleep(a.poll_interval)


def simple(data, prompt, parser, key, a, out, web=False):
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
        c = a.cache_dir / f"{job.name.replace('/', '_')}.json"
        c.write_text(json.dumps({"job": job.name, "out": str(out)}))
        by = {str(dp["id"]): dp for dp in data}
        for item in wait(job, a):
            dp = by[item.metadata["id"]]
            dp[key] = parser(extract_gemini_response_text(item.response)).get()
            save_gemini_thinking_trace(dp, key, item.response)
        c.unlink(missing_ok=True)
    save(out, data)
    return data


def llm_check(data, source, a, out, web=False):
    if a.backend == "transformers":
        for dp in data:
            dp[FACTCHECK_RESULTS_KEY] = [-1] * len(claims(dp))
        items = [
            (dp, i, x)
            for dp in data
            for i, x in enumerate(claims(dp))
        ]
        for start in range(0, len(items), a.batch_size):
            batch = items[start : start + a.batch_size]
            texts = run_transformers_model_batch(
                a.model_name,
                [prompt_check(dp, x, source, a) for dp, _, x in batch],
                device=a.device,
                dtype=a.dtype,
                enable_thinking=a.thinking == "true",
                max_new_tokens=36,
            )
            for (dp, i, _), text in zip(batch, texts):
                dp[FACTCHECK_RESULTS_KEY][i] = (
                    LLMCheckTemplate.ResponseClass.model_validate_plain_text(
                        text
                    ).get()
                )
    elif a.batching == "false":
        for dp in data:
            dp[FACTCHECK_RESULTS_KEY] = []
            for i, x in enumerate(claims(dp)):
                r = call_gemini_one_by_one_api(
                    prompt_check(dp, x, source, a),
                    model=a.model_name,
                    web_search=web,
                    thinking_level=a.thinking_level if a.thinking == "true" else None,
                )
                dp[FACTCHECK_RESULTS_KEY].append(
                    LLMCheckTemplate.ResponseClass.model_validate_plain_text(
                        extract_gemini_response_text(r)
                    ).get()
                )
                save_gemini_thinking_trace(dp, FACTCHECK_RESULTS_KEY, r, index=i)
    else:
        req = [
            message2gemini_request(
                {"id": str(dp["id"]), "index": str(i)},
                prompt_check(dp, x, source, a),
                a.model_name,
                web_search=web,
                thinking_level=a.thinking_level if a.thinking == "true" else None,
            )
            for dp in data
            for i, x in enumerate(claims(dp))
        ]
        job = submit_gemini_job(req, a.model_name)
        by = {str(dp["id"]): dp for dp in data}
        for dp in data:
            dp[FACTCHECK_RESULTS_KEY] = [-1] * len(claims(dp))
        for item in wait(job, a):
            dp = by[item.metadata["id"]]
            i = int(item.metadata["index"])
            dp[FACTCHECK_RESULTS_KEY][i] = (
                LLMCheckTemplate.ResponseClass.model_validate_plain_text(
                    extract_gemini_response_text(item.response)
                ).get()
            )
            save_gemini_thinking_trace(
                dp, FACTCHECK_RESULTS_KEY, item.response, index=i
            )
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
        "--factchecked_file",
        type=Path,
        help="Existing fact-checked JSONL; skips extraction and fact-checking.",
    )
    p.add_argument("--presupposition_output_dir", type=Path, required=True)
    p.add_argument("--factcheck_output_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--RAG_device", default="cuda")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--RAG", choices=["0", "all", "4", "web"], required=True)
    p.add_argument("--factchecker", choices=["llm", "minicheck"], required=True)
    p.add_argument("--thinking", choices=["true", "false"], required=True)
    p.add_argument("--batching", choices=["true", "false"], default="false")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--cache_dir", type=Path, default=Path("tmp/presupposition"))
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
    if a.factchecker == "minicheck" and a.RAG not in {"all", "4"}:
        p.error("--factchecker minicheck requires --RAG all or --RAG 4")
    factchecked_exists = (
        a.factchecked_file is not None and a.factchecked_file.is_file()
    )
    presupposition_exists = (
        a.presupposition_file is not None and a.presupposition_file.is_file()
    )
    if factchecked_exists:
        checked_input = [json.loads(x) for x in open(a.factchecked_file)]
        data = None
    elif presupposition_exists:
        data = [json.loads(x) for x in open(a.presupposition_file)]
    else:
        if a.dataset_path is None:
            p.error("--dataset_path is required when no intermediate input file exists")
        data = [json.loads(x) for x in open(a.dataset_path)]
        extract = a.presupposition_output_dir / "RAG=0.jsonl"
        data = simple(
            data,
            prompt_extract,
            PresuppositionExtractionTemplate.ResponseClass.model_validate_plain_text,
            MODEL_DETECTED_PRESUPPOSITIONS_KEY,
            a,
            extract,
        )
    settings = [
        {
            "0": ("0", "no_passages", False),
            "all": ("all", "use_passages", False),
            "4": ("4", "use_RAG", False),
            "web": ("web", "no_passages", True),
        }[a.RAG]
    ]
    for rag, source, web in settings:
        if a.factchecker == "llm":
            suffix = (
                "gemini_checked" if a.backend == "gemini" else "transformers_checked"
            )
        else:
            suffix = "minichecked"

        if factchecked_exists:
            checked = [dict(dp) for dp in checked_input]
        elif a.factchecker == "llm":
            checked = llm_check(
                [dict(dp) for dp in data],
                source,
                a,
                a.factcheck_output_dir / f"RAG={rag}_{suffix}.jsonl",
                web,
            )
        else:
            model = MiniCheck("flan-t5-large", cache_dir=os.getenv("HF_HOME"))
            checked = [dict(dp) for dp in data]
            for dp in checked:
                ps = claims(dp)
                docs = get_passages(
                    dp,
                    claims=ps,
                    source=source,
                    RAG_device=a.RAG_device,
                    instruction="Given a list of statements, retrieve relevant passages that validate or refute the statements",
                )
                dp[FACTCHECK_RESULTS_KEY] = [
                    0
                    if not doc.strip()
                    else model.score(docs=[doc], claims=[x])[0][0]
                    for doc, x in zip(docs, ps)
                ]
            save(a.factcheck_output_dir / f"RAG={rag}_{suffix}.jsonl", checked)

        simple(
            checked,
            prompt_final,
            FactCheckFinalAnswerTemplate.ResponseClass.model_validate_plain_text,
            MODEL_FINAL_ANSWER_KEY,
            a,
            a.output_dir / f"RAG={rag}_{suffix}.jsonl",
        )


if __name__ == "__main__":
    main()

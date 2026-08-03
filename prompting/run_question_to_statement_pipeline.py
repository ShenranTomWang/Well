"""Standalone question-to-statement/check/final-answer research pipeline."""

import argparse
import json
import os
import time
from pathlib import Path
from constant.constant import (
    FACTCHECK_RESULTS_KEY,
    MODEL_CONVERTED_STATEMENT_KEY,
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_FINAL_ANSWER_KEY,
)
from response import QuestionToStatementResponse, LLMCheckResponse, FinalAnswerResponse, PresuppositionExtractionResponse
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


class QuestionToStatementTemplate:
    ResponseClass = QuestionToStatementResponse

    def __init__(self, question, few_shot_data, passages=None, **kw):
        self.question = question
        self.passages = passages or []
        self.few_shot_data = [
            f"user: Question: {dp['question']}\nassistant: Statement: {dp['statement']}"
            for dp in few_shot_data
        ]

    def generate(self):
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that analyzes the given question.
                    You will be provided with a question. Your task is to transform the question into a statement and keep its original meaning.
                    Return exactly one statement. Do not answer the question or add explanations.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                """,
            },
            {"role": "user", "content": f"Question: {self.question}"},
        ]


class KnowledgeGenerationTemplate:
    ResponseClass = FinalAnswerResponse

    def __init__(self, statement, few_shot_data, **kw):
        self.statement = statement
        self.few_shot_data = [
            f"user: Input: {dp['statement']}\nassistant: {dp['knowledge']}"
            for dp in few_shot_data
        ]

    def generate(self):
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that generate some knowledge about the user input.
                    You will be provided with a statement. Your task is to generate relevant knowledge for the statement.
                    Return exactly one piece of knowledge. Do not answer the question or add explanations.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                """,
            },
            {"role": "user", "content": f"Input: {self.statement}"},
        ]


class PresuppositionExtractionTemplate:
    ResponseClass = PresuppositionExtractionResponse

    def __init__(self, statement, few_shot_data, passages=None, **kw):
        self.statement = statement
        self.passages = passages or []
        self.few_shot_data = [
            f"user: {dp['question']}\nassistant: {'\n'.join(dp['presuppositions'])}\n"
            for dp in few_shot_data
        ]

    def generate(self):
        rag = (
            f"For the following statement, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}"
            if len(self.passages) > 0
            else ""
        )
        return [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant. Help me understand the question by extracting both explicit and implicit atomic assumptions. You must notice that considering the intention of the question asker is helpful for extracting a hidden assumption. Output every atomic assumption in a complete sentence.
                    There could be multiple presuppositions in a statement, but there will always be at least one presupposition in the statement.
                    Format your response as a list of presuppositions, separated by newlines.
                    {"Below are some examples to help you understand the task." if self.few_shot_data else ""}
                    {"\n\n".join(self.few_shot_data)}
                    {rag}
                """,
            },
            {"role": "user", "content": self.statement},
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
                    {f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}" if self.passages else ""}
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


def presuppositions(dp):
    x = dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY]
    return [x] if isinstance(x, str) else x


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


def check(data, source, a, out, web=False):
    def prompt(dp, x):
        passages = (
            [dp["knowledge"]]
            if source == "generated_knowledge"
            else get_passages(
                dp,
                query=x,
                source=source,
                RAG_device=a.RAG_device,
                instruction="Given a presupposition, retrieve relevant passages that validate or refute the presupposition",
            )
        )
        return LLMCheckTemplate(
            model_detected_presupposition=x,
            few_shot_data=dp["few_shot_data"],
            passages=passages,
        ).generate()

    if a.backend == "transformers":
        for dp in data:
            dp[FACTCHECK_RESULTS_KEY] = [-1] * len(presuppositions(dp))
        items = [
            (dp, i, x)
            for dp in data
            for i, x in enumerate(presuppositions(dp))
        ]
        for start in range(0, len(items), a.batch_size):
            batch = items[start : start + a.batch_size]
            texts = run_transformers_model_batch(
                a.model_name,
                [prompt(dp, x) for dp, _, x in batch],
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
            for i, x in enumerate(presuppositions(dp)):
                r = call_gemini_one_by_one_api(
                    prompt(dp, x),
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
        job = submit_gemini_job(
            [
                message2gemini_request(
                    {"id": str(dp["id"]), "index": str(i)},
                    prompt(dp, x),
                    a.model_name,
                    web_search=web,
                    thinking_level=a.thinking_level if a.thinking == "true" else None,
                )
                for dp in data
                for i, x in enumerate(presuppositions(dp))
            ],
            a.model_name,
        )
        by = {str(dp["id"]): dp for dp in data}
        for dp in data:
            dp[FACTCHECK_RESULTS_KEY] = [-1] * len(presuppositions(dp))
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
        "--question_decomposed_file",
        type=Path,
        help="Existing question-to-statement JSONL; skips question conversion.",
    )
    p.add_argument(
        "--presupposition_file",
        type=Path,
        help="Existing statement-to-presupposition JSONL; skips both decomposition stages.",
    )
    p.add_argument(
        "--knowledge_file",
        type=Path,
        help="Existing generated-knowledge JSONL; used for RAG=0 when available.",
    )
    p.add_argument(
        "--factchecked_file",
        type=Path,
        help="Existing fact-checked JSONL; skips all stages before final-answer generation.",
    )
    p.add_argument("--question_decomposition_output_dir", type=Path, required=True)
    p.add_argument("--presupposition_output_dir", type=Path, required=True)
    p.add_argument("--knowledge_output_dir", type=Path, required=True)
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
    p.add_argument("--cache_dir", type=Path, default=Path("tmp/statement"))
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
    question_decomposed_exists = (
        a.question_decomposed_file is not None
        and a.question_decomposed_file.is_file()
    )
    knowledge_exists = a.knowledge_file is not None and a.knowledge_file.is_file()
    if factchecked_exists:
        checked_input = [json.loads(x) for x in open(a.factchecked_file)]
        decomposed = None
    elif a.RAG == "0" and knowledge_exists:
        decomposed = [json.loads(x) for x in open(a.knowledge_file)]
    elif presupposition_exists:
        decomposed = [json.loads(x) for x in open(a.presupposition_file)]
    else:
        if question_decomposed_exists:
            converted = [json.loads(x) for x in open(a.question_decomposed_file)]
        else:
            if a.dataset_path is None:
                p.error("--dataset_path is required when no intermediate input file exists")
            data = [json.loads(x) for x in open(a.dataset_path)]
            converted = infer(
                data,
                lambda dp: QuestionToStatementTemplate(
                    question=dp["question"], few_shot_data=dp["few_shot_data"]
                ).generate(),
                QuestionToStatementTemplate.ResponseClass.model_validate_plain_text,
                MODEL_CONVERTED_STATEMENT_KEY,
                a,
                a.question_decomposition_output_dir / "RAG=0.jsonl",
            )
        decomposed = infer(
            converted,
            lambda dp: PresuppositionExtractionTemplate(
                statement=dp[MODEL_CONVERTED_STATEMENT_KEY],
                few_shot_data=dp["few_shot_data"],
            ).generate(),
            PresuppositionExtractionTemplate.ResponseClass.model_validate_plain_text,
            MODEL_DETECTED_PRESUPPOSITIONS_KEY,
            a,
            a.presupposition_output_dir / "RAG=0.jsonl",
        )
    if not factchecked_exists and a.RAG == "0" and not knowledge_exists:
        decomposed = infer(
            [dict(dp) for dp in decomposed],
            lambda dp: KnowledgeGenerationTemplate(
                statement=dp[MODEL_CONVERTED_STATEMENT_KEY],
                few_shot_data=dp["few_shot_data"],
            ).generate(),
            KnowledgeGenerationTemplate.ResponseClass.model_validate_plain_text,
            "knowledge",
            a,
            a.knowledge_output_dir / "RAG=0.jsonl",
        )
    settings = [
        {
            "0": ("0", "generated_knowledge", False),
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
            checked = check(
                [dict(dp) for dp in decomposed],
                source,
                a,
                a.factcheck_output_dir / f"RAG={rag}_{suffix}.jsonl",
                web,
            )
        else:
            model = MiniCheck("flan-t5-large", cache_dir=os.getenv("HF_HOME"))
            checked = [dict(dp) for dp in decomposed]
            for dp in checked:
                ss = presuppositions(dp)
                docs = get_passages(
                    dp,
                    claims=ss,
                    source=source,
                    RAG_device=a.RAG_device,
                    instruction="Given a presupposition, retrieve relevant passages that validate or refute the presupposition",
                )
                dp[FACTCHECK_RESULTS_KEY] = [
                    0
                    if not doc.strip()
                    else model.score(docs=[doc], claims=[x])[0][0]
                    for doc, x in zip(docs, ss)
                ]
            save(a.factcheck_output_dir / f"RAG={rag}_{suffix}.jsonl", checked)

        infer(
            checked,
            lambda dp: FactCheckFinalAnswerTemplate(
                question=dp["question"],
                few_shot_data=dp["few_shot_data"],
                factcheck_results=dp[FACTCHECK_RESULTS_KEY],
                model_detected_presuppositions=presuppositions(dp),
            ).generate(),
            FactCheckFinalAnswerTemplate.ResponseClass.model_validate_plain_text,
            MODEL_FINAL_ANSWER_KEY,
            a,
            a.output_dir / f"RAG={rag}_{suffix}.jsonl",
        )


if __name__ == "__main__":
    main()

from typing import List, Iterable, Dict, Type, Tuple, Any
import evaluate
from bert_score import BERTScorer
from ignite.metrics import RougeL, RougeN
from google import genai
from google.genai import types
from data_gen.template import Template
from response import Response
import numpy as np

__CACHE__ = {
    "bleurt_models": None,
    "bert_scorer": {},
}

def _avg_report(
    data: List[Dict],
    measure: str = '',
    run_rouge: bool = False,
    run_bleurt: bool = False,
    run_bert_score: bool = False,
    run_fp_score: bool = False,
    presupposition_extraction: bool = False,
    fp_only: bool = False,
    tp_only: bool = False
):
    _measure = f'_{measure}' if measure != '' else ''
    if run_rouge:
        rouge1_f1_key = f'rouge1_f1{_measure}'
        rougeL_f1_key = f'rougeL_f1{_measure}'
        avg_rouge1 = np.mean([dp[rouge1_f1_key] for dp in data if dp.get(rouge1_f1_key) is not None])
        avg_rougeL = np.mean([dp[rougeL_f1_key] for dp in data if dp.get(rougeL_f1_key) is not None])
        print(f'Average ROUGE-1 F1 {measure.capitalize()}: {avg_rouge1:.4f}')
        print(f'Average ROUGE-L F1 {measure.capitalize()}: {avg_rougeL:.4f}')
    if run_bleurt:
        bleurt_key = f'bleurt_f1{_measure}'
        avg_bleurt = np.mean([dp[bleurt_key] for dp in data if dp.get(bleurt_key) is not None])
        print(f'Average BLEURT F1 {measure.capitalize()}: {avg_bleurt:.4f}')
    if run_bert_score:
        bert_score_key = f'bert_score_f1{_measure}'
        avg_bert_score = np.mean([dp[bert_score_key] for dp in data if dp.get(bert_score_key) is not None])
        print(f'Average BERTScore F1 {measure.capitalize()}: {avg_bert_score:.4f}')
    if run_fp_score:
        if presupposition_extraction:
            if fp_only:
                avg_fp_score = np.mean([dp[GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_RECALL_KEY] for dp in data if dp.get(GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_RECALL_KEY) is not None and "false presupposition" in dp['labels']])
                print(f'Average FP Score Recall {measure.capitalize()}: {avg_fp_score:.4f}')
                avg_fp_score = np.mean([dp[GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_PRECISION_KEY] for dp in data if dp.get(GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_PRECISION_KEY) is not None and "false presupposition" in dp['labels']])
                print(f'Average FP Score Precision {measure.capitalize()}: {avg_fp_score:.4f}')
            elif tp_only:
                avg_fp_score = np.mean([dp[GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_RECALL_KEY] for dp in data if dp.get(GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_RECALL_KEY) is not None and "false presupposition" not in dp['labels']])
                print(f'Average FP Score Recall {measure.capitalize()}: {avg_fp_score:.4f}')
                avg_fp_score = np.mean([dp[GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_PRECISION_KEY] for dp in data if dp.get(GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_PRECISION_KEY) is not None and "false presupposition" not in dp['labels']])
                print(f'Average FP Score Precision {measure.capitalize()}: {avg_fp_score:.4f}')
            else:
                avg_fp_score = np.mean([dp[GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_RECALL_KEY] for dp in data if dp.get(GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_RECALL_KEY) is not None])
                print(f'Average FP Score Recall {measure.capitalize()}: {avg_fp_score:.4f}')
                avg_fp_score = np.mean([dp[GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_PRECISION_KEY] for dp in data if dp.get(GEMINI_SCORE_PRESUPPOSITION_EXTRACTION_PRECISION_KEY) is not None])
                print(f'Average FP Score Precision {measure.capitalize()}: {avg_fp_score:.4f}')
        else:
            if fp_only:
                fp_score_key = f'{GEMINI_SCORE_RECALL_KEY}'
                avg_fp_score = np.mean([dp[fp_score_key] for dp in data if dp.get(fp_score_key) is not None and "false presupposition" in dp['labels']])
                print(f'Average FP Score Recall {measure.capitalize()}: {avg_fp_score:.4f}')
                fp_score_key = f'{GEMINI_SCORE_PRECISION_KEY}'
                avg_fp_score = np.mean([dp[fp_score_key] for dp in data if dp.get(fp_score_key) is not None and "false presupposition" in dp['labels']])
                print(f'Average FP Score Precision {measure.capitalize()}: {avg_fp_score:.4f}')
            elif tp_only:
                fp_score_key = f'{GEMINI_SCORE_RECALL_KEY}'
                avg_fp_score = np.mean([dp[fp_score_key] for dp in data if dp.get(fp_score_key) is not None and "false presupposition" not in dp['labels']])
                print(f'Average FP Score Recall {measure.capitalize()}: {avg_fp_score:.4f}')
                fp_score_key = f'{GEMINI_SCORE_PRECISION_KEY}'
                avg_fp_score = np.mean([dp[fp_score_key] for dp in data if dp.get(fp_score_key) is not None and "false presupposition" not in dp['labels']])
                print(f'Average FP Score Precision {measure.capitalize()}: {avg_fp_score:.4f}')
            else:
                fp_score_key = f'{GEMINI_SCORE_RECALL_KEY}'
                avg_fp_score = np.mean([dp[fp_score_key] for dp in data if dp.get(fp_score_key) is not None])
                print(f'Average FP Score Recall {measure.capitalize()}: {avg_fp_score:.4f}')
                fp_score_key = f'{GEMINI_SCORE_PRECISION_KEY}'
                avg_fp_score = np.mean([dp[fp_score_key] for dp in data if dp.get(fp_score_key) is not None])
                print(f'Average FP Score Precision {measure.capitalize()}: {avg_fp_score:.4f}')

def _parse_response_gemini(response_cls: Type[Response], response: Dict | str | None) -> Response:
    if isinstance(response, Dict):
        response = response['response']['text']
    return response_cls.model_validate_plain_text(response)

def f1(prec: float, rec: float, eps: float = 1e-12) -> float:
    return 2 * prec * rec / (prec + rec + eps)

def rouge1_f1(
    candidate: str,
    references: Iterable[str],
) -> float:
    """
    ROUGE-1 F1 over possibly multiple references (take the best ref).
    """
    m = RougeN(ngram=1, multiref='best', alpha=0.5)
    m.update(([candidate.split()], [[r.split() for r in references]]))
    return m.compute()['Rouge-1-F']

def rougeL_f1(
    candidate: str,
    references: Iterable[str]
) -> float:
    """
    ROUGE-L F1 using Lin (2004) LCS-based precision/recall.
    Multiple refs: take the best ref.
    """
    m = RougeL(multiref='best', alpha=0.5)
    m.update(([candidate.split()], [[r.split() for r in references]]))
    return m.compute()['Rouge-L-F']

def bleurt_score(
    candidates: Iterable[str],
    references: Iterable[str],
    model_name: str = "Elron/bleurt-large-512",
    reduction: str = "max"
) -> float:
    """
    Compute BLEURT for candidate vs multiple references.
    reduction: "max" or "mean"
    """
    bleurt = __CACHE__.get("bleurt_models")
    if bleurt is None:
        __CACHE__["bleurt_models"] = evaluate.load("bleurt", module_type="metric", config_name=model_name)
        bleurt = __CACHE__["bleurt_models"]
    scores = []
    for ref in references:
        _scores = []
        for candidate in candidates:
            res = bleurt.compute(predictions=[candidate], references=[ref], return_details=True)
            _scores.append(res["scores"][0])
        scores.append(max(_scores))
    if reduction == "mean":
        return sum(scores) / len(scores)
    else:
        return max(scores)

def bert_score_f1(
    candidates: List[str],
    references: List[str],
    model_type: str = None
) -> float:
    """
    Compute BERTScore F1 for lists of candidates and references.
    """
    bert_scorer = __CACHE__.get("bert_scorer", {}).get(model_type, None)
    if bert_scorer is None:
        bert_scorer = BERTScorer(model_type=model_type, lang="en")
        __CACHE__["bert_scorer"][model_type] = bert_scorer
    _, _, F1 = bert_scorer.score(candidates, references)
    if F1.item() < -1: breakpoint()
    return F1.mean().item()

def fp_score_presupposition_extraction(
    messages: List[Dict],
    response_type: Type[Response]
) -> Response:
    client = __CACHE__.get("genai_client", None)
    if not client:
        client = genai.Client()
        __CACHE__["genai_client"] = client
    response1 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[{"role": message["role"], "parts": [{"text": message["content"]}]} for message in messages[1:]],
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=messages[0]['content']
        )
    )
    response1 = _parse_response_gemini(response_type, response1.text)
    return response1

def _run_fp_score_coverage(
    messages: List[List[Dict[str, str]]],
    response_type: Type[Response],
) -> List[int]:
    client = __CACHE__.get("genai_client", None)
    if not client:
        client = genai.Client()
        __CACHE__["genai_client"] = client
    coverages = []
    for _messages in messages:
        response2 = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[{"role": message["role"], "parts": [{"text": message["content"]}]} for message in _messages[1:]],
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=_messages[0]['content']
            )
        )
        coverage = _parse_response_gemini(response_type, response2.text).get()
        coverages.append(coverage)
    return coverages

def fp_score_recall_generate(
    presuppositions: List[str],
    template_type: Type[Template],
    answer_extracted_presuppositions: List[str],
    system_role: str = "system",
    model_role: str = "assistant",
    user_role: str = "user"
) -> List[List[Dict[str, str]]]:
    messages = []
    for presupposition in presuppositions:
        _messages = _fp_score_template_generate(
            template_type=template_type,
            answer_extracted_presuppositions=answer_extracted_presuppositions,
            presupposition=presupposition,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )
        messages.append(_messages)
    return messages

def fp_score_precision_generate(
    presuppositions: List[str],
    template_type: Type[Template],
    answer_extracted_presuppositions: List[str],
    system_role: str = "system",
    model_role: str = "assistant",
    user_role: str = "user"
) -> List[List[Dict[str, str]]]:
    messages = []
    for answer_extracted_presupposition in answer_extracted_presuppositions:
        _messages = _fp_score_template_generate(
            template_type=template_type,
            answer_extracted_presupposition=answer_extracted_presupposition,
            presuppositions=presuppositions,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )
        messages.append(_messages)
    return messages

def fp_score_presupposition_extraction_generate(
    question: str,
    few_shot_data: List[Dict[str, str]],
    model_final_answer: str,
    template_type: Type[Template],
    system_role: str = "system",
    model_role: str = "assistant",
    user_role: str = "user"
) -> List[Dict[str, str]]:
    messages = _fp_score_template_generate(
        template_type=template_type,
        question=question,
        few_shot_data=few_shot_data,
        model_final_answer=model_final_answer,
        system_role=system_role,
        model_role=model_role,
        user_role=user_role
    )
    return messages

def fp_score_precision(
    messages: List[List[Dict[str, str]]],
    response_type: Type[Response],
) -> Tuple[List[int], float]:
    """
    Compute the percentage of false presuppositions being identified by the model final answer that are also covered by the ground truths.
    Args:
        messages: list of messages for each extracted presupposition from the answer
        response_type: response type for coverage evaluation
    Returns:
        - list of coverage (1/0) for each presupposition by the extracted presuppositions from response
        - precision score
    """
    coverages = _run_fp_score_coverage(
        messages=messages,
        response_type=response_type,
    )
    return (coverages, np.mean(coverages) if len(coverages) > 0 else 0)
        
def fp_score_recall(
    messages: List[List[Dict[str, str]]],
    response_type: Type[Response],
) -> Tuple[List[int], float]:
    """
    Compute the percentage of false presuppositions being identified by the model final answer.
    Args:
        messages: list of messages for each presupposition
        response_type: response type for coverage evaluation
    Returns:
        - list of coverage (1/0) for each presupposition by the extracted presuppositions from response
        - recall score
    """
    coverages = _run_fp_score_coverage(
        messages=messages,
        response_type=response_type,
    )
    return (coverages, np.mean(coverages) if len(coverages) > 0 else 0)

def _fp_score_template_generate(
    template_type: Type[Template],
    system_role: str = "system",
    model_role: str = "assistant",
    user_role: str = "user",
    **kwargs
) -> List[Dict[str, str]]:
    template = template_type(
        system_role=system_role,
        model_role=model_role,
        user_role=user_role,
        **kwargs
    )
    return template.generate()
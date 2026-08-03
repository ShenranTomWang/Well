from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
import re
from constant.constant import RAG_TOP_PASSAGES_KEY

__CACHE__ = {
    "embedding_models": {}
}
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

def keep_normal_english(batch: List[str]) -> str:
    control_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    byte_escape_re = re.compile(r"(?:\\x[0-9a-fA-F]{2}){2,}")
    cleaned_batch = []
    for item in batch:
        if control_re.search(item) or byte_escape_re.search(item) or "\uFFFD" in item:
            continue
        cleaned_batch.append(item)
    return cleaned_batch

def truncate_item(item: str, length_truncation: int) -> List[str]:
    """Truncate a long passage into multiple passages that are shorter than the length truncation threshold.
    Args:
        item (str): The input passage to be truncated.
        length_truncation (int): The maximum length of the passage after truncation. Passages longer than this will be split.
    Returns:
        List[str]: A list of truncated passages that are shorter than the length truncation threshold.
    """
    items = []
    item = item.split()
    items.append(item[:length_truncation])
    remaining = item[length_truncation:]
    while len(remaining) > 0:
        items.append(remaining[:length_truncation])
        remaining = remaining[length_truncation:]
    return [" ".join(item) for item in items]

def get_passages(
    dp: Dict[str, Any],
    source: str,
    instruction: str = DEFAULT_INSTRUCTION,
    query: str = None,
    k: int = 4,
    model: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 16,
    RAG_device: torch.DeviceObjType = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    **kwargs
):
    if not query:
        query = dp['question']
    if source == 'use_RAG':
        dp_passages = '\n\n'.join(dp['passages'])
        dp_passages = dp_passages.split('\n\n')
        passages = run_RAG(
            query=query,
            instruction=instruction,
            passages=dp_passages,
            k=k,
            RAG_model=model if not kwargs.get("RAG_model", None) else kwargs.get("RAG_model"),
            RAG_device=RAG_device,
            batch_size=batch_size
        )
        dp[RAG_TOP_PASSAGES_KEY] = passages
    elif source == 'use_passages':
        passages = keep_normal_english(dp['passages'])
    elif source == 'no_passages':
        passages = []
    else:
        raise ValueError(f"Unknown source {source}")
    return passages

def run_RAG(
    query: str,
    instruction: str,
    passages: List[str],
    k: int,
    RAG_model: str,
    RAG_device: torch.DeviceObjType,
    batch_size: int,
    **kwargs
) -> List[str]:
    """Operator-level interface for running RAG
    Args:
        query (str): The input query for which to retrieve relevant passages.
        instruction (str): The instruction to guide the retrieval process.
        passages (List[str]): The list of passages to search from.
        k (int): The number of top passages to retrieve.
        RAG_model (str): The name of the sentence transformer model to use for encoding the query and passages.
        RAG_device (torch.DeviceObjType): The device (CPU or GPU) to use for encoding the query and passages and computing similarities.
        batch_size (int): The number of passages to process in each batch for similarity computation.
    Returns:
        List[str]: top-k passages retrieved by RAG.
    """
    retrieved_passages = RAG(
        query=query,
        instruction=instruction,
        passages=passages,
        k=k,
        model_name=RAG_model,
        device=RAG_device,
        batch_size=batch_size
    )
    return retrieved_passages

@torch.inference_mode()
def RAG(
    query: str,
    instruction: str,
    k: int,
    passages: List[str],
    model_name: str,
    device: torch.DeviceObjType | str,
    batch_size: int,
    length_truncation: int = 1024,
    **kwargs
) -> List[str]:
    """
    Retrieve top-k passages from the given list of passages based on the query embedding.
    Args:
        query (str): The input query for which to retrieve relevant passages.
        instruction (str): The instruction to guide the retrieval process.
        k (int): The number of top passages to retrieve.
        passages (List[str]): The list of passages to search from.
        model_name (str): The name of the transformer model to use for encoding the query and passages.
        device (torch.DeviceObjType): The device (CPU, GPU or 'auto') to use for encoding the query and passages and computing similarities.
        batch_size (int): The number of passages to process in each batch for similarity computation.
        length_truncation (int): The maximum length of the passages after tokenization. Passages longer than this will be split. Default is 1024
    Returns:
        List[str]: A list of the top-k retrieved passages corresponding to the query.
    """
    model = __CACHE__["embedding_models"].get(model_name, None)
    if model is None:
        model = SentenceTransformer(model_name, device=device)
        __CACHE__["embedding_models"][model_name] = model
    if not passages or k <= 0:
        return []
    query_embedding = model.encode([query], prompt=instruction, convert_to_tensor=True)
    passages = [
        split_item
        for item in keep_normal_english(passages)
        for split_item in truncate_item(item, length_truncation)
    ]
    if not passages:
        return []
    best_similarities = None
    best_indices = None
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        batch_similarities = model.similarity(query_embedding, batch_embeddings)
        batch_indices = torch.arange(i, i + len(batch), device=batch_similarities.device, dtype=torch.long)
        if best_similarities is None:
            best_similarities = batch_similarities
            best_indices = batch_indices
        else:
            best_similarities = torch.cat([best_similarities, batch_similarities], dim=-1)
            best_indices = torch.cat([best_indices, batch_indices], dim=-1)
        keep = min(k, best_similarities.shape[-1])
        best_similarities, top_positions = torch.topk(best_similarities, k=keep, largest=True)
        best_indices = best_indices[top_positions][0]
    retrieved_passages = [passages[i] for i in best_indices.tolist()]
    return retrieved_passages

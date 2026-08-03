"""
Run path patching and select false-premise attention heads.

This script generalizes the original two-step workflow:

1. compute one path-patch influence matrix per sample;
2. rank heads by how often they receive high influence scores.

Examples:
    python identify_nobel_prize_heads.py \
        --dataset movies \
        --model-name-or-path /path/to/Qwen2.5-7B-Instruct \
        --output-dir results/movie_heads/qwen25_7b

    python identify_nobel_prize_heads.py \
        --dataset nobel_prize \
        --model-name-or-path /path/to/Meta-Llama-3.2-3B-Instruct \
        --output-dir results/nobel_heads/llama32_3b

    python identify_nobel_prize_heads.py \
        --dataset movies \
        --model-name-or-path /path/to/Olmo-3-7B-Instruct \
        --output-dir results/movie_heads/olmo3_7b
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from FAITH.utils.nethook import TraceDict


SUPPORTED_FAMILIES = ("llama3.2", "qwen2.5", "qwen3", "gemma4", "llama3", "olmo3")
SYS_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
}
LLAMA_SYS_PROMPT = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""
LLAMA_PROMPT_TEMPLATE = LLAMA_SYS_PROMPT + " {} [/INST] {}"

NOBEL_DATASET_CANDIDATES = (
    "dataset/ToyDataset/Awards/llama2-13b-chat_on_nobel_prize.json",
    "dataset/ToyDataset/Awards/llama2-7b-chat_on_nobel_prize.json",
    "dataset/ToyDataset/Awards/nobel-prize-laureates-simple.json",
    "dataset/ToyDataset/Awards/nobal_prizes.csv",
)
MOVIE_DATASET_CANDIDATES = (
    "/ubc/cs/home/s/shenranw/scratch/datasets/Movies/wikidata_movies.json",
    "/ubc/cs/home/s/shenranw/scratch/datasets/Movies/film_release.json",
    "dataset/ToyDataset/Movies/llama2-13b-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json",
    "dataset/ToyDataset/Movies/llama2-7b-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json",
)


def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([int(t)]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, token in enumerate(toks):
        loc += len(token)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return tok_start, tok_end


def render_prompt(tokenizer, question, answer, prompt_style):
    if prompt_style == "plain":
        return f"{question}\n{answer}"
    if prompt_style == "faith":
        return LLAMA_PROMPT_TEMPLATE.format(question, answer)

    messages = [SYS_PROMPT, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return LLAMA_PROMPT_TEMPLATE.format(question, answer)


def read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def read_json_or_jsonl(path: Path):
    with path.open("r") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", str(value).strip())
    return value[:120] or "sample"


def load_csv_dataset(path: Path):
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def candidate_paths(candidates):
    paths = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        paths.append(path)
    env_base = os.environ.get("FAITH_BASE_DIR") or os.environ.get("HALLUINDUCING_BASE_DIR")
    if env_base:
        for candidate in candidates:
            if not Path(candidate).is_absolute():
                paths.append(Path(env_base).expanduser() / candidate)
    return paths


def find_dataset_file(dataset, explicit_path):
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
        return path

    candidates = NOBEL_DATASET_CANDIDATES if dataset == "nobel_prize" else MOVIE_DATASET_CANDIDATES
    paths = candidate_paths(candidates)
    for path in paths:
        if path.is_file():
            return path

    tried = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not auto-locate a {dataset} dataset file. Pass --dataset-file. Tried:\n{tried}")


def normalize_award_sample(sample):
    normalized = dict(sample)
    if "awardYear" not in normalized:
        for key in ("award_year", "year", "prize_year"):
            if key in normalized:
                normalized["awardYear"] = normalized[key]
                break
    if "categoryFullName" not in normalized:
        for key in ("category_full_name", "categoryFullname", "category_name", "category"):
            if key in normalized:
                normalized["categoryFullName"] = normalized[key]
                break
    if "name" not in normalized:
        for key in ("knownName", "fullName", "laureate", "laureate_name"):
            if key in normalized:
                normalized["name"] = normalized[key]
                break
    return normalized


def construct_nobel_questions(samples):
    for sample in samples:
        if "awardYear" not in sample:
            continue
        false_year = int(sample["awardYear"]) + 1
        name = sample.get("name") or sample.get("fullName") or sample.get("knownName")
        category = sample.get("categoryFullName") or sample.get("category")
        if name and category:
            sample.setdefault(
                "when_fp_question",
                f"For what specific contribution was {name} awarded {category}in {false_year}?",
            )
            sample.setdefault(
                "when_fp_question2",
                f"{name} was awarded {category} in {false_year} for what specific reason?",
            )
            sample.setdefault(
                "when_fp_question3",
                f"{category} in {false_year} was awarded to {name} for what specific reason?",
            )
            sample.setdefault(
                "when_fp_question4",
                f"Why was {name} awarded the {false_year} {category}?",
            )
    return samples


def load_nobel_prize_samples(dataset_file, question_key):
    if dataset_file:
        path = find_dataset_file("nobel_prize", dataset_file)
        raw = load_csv_dataset(path) if path.suffix == ".csv" else read_json_or_jsonl(path)
        source = str(path)
    else:
        try:
            from datasets import load_dataset

            raw = list(load_dataset("HongbangYuan/FAITH", split="Prize_7b"))
            source = "HongbangYuan/FAITH:Prize_7b"
        except Exception:
            path = find_dataset_file("nobel_prize", None)
            raw = load_csv_dataset(path) if path.suffix == ".csv" else read_json_or_jsonl(path)
            source = str(path)

    if isinstance(raw, dict):
        for key in ("train", "test", "validation", "data", "samples"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    samples = construct_nobel_questions([normalize_award_sample(sample) for sample in raw])
    required = ("name", "categoryFullName", "awardYear", question_key)
    usable = []
    for sample in samples:
        if all(sample.get(key) not in (None, "") for key in required):
            sample["awardYear"] = int(sample["awardYear"])
            usable.append(sample)
    if not usable:
        raise ValueError(f"No usable Nobel Prize samples found. Required keys: {required}")
    return usable, source


def normalize_movie_sample(sample, question_key):
    normalized = dict(sample)
    movie = normalized.get("movie") or normalized.get("original_title") or normalized.get("title")
    times = normalized.get("time") or normalized.get("release_year") or normalized.get("year")
    if isinstance(times, str):
        times = [int(times[:4])]
    elif isinstance(times, int):
        times = [times]
    elif isinstance(times, list):
        times = [int(str(t)[:4]) for t in times]
    else:
        times = []

    if not movie or not times:
        return None
    true_year = min(times)
    false_year = int(normalized.get("false_year", true_year - 1))
    normalized["movie"] = movie
    normalized["name"] = movie
    normalized["time"] = times
    normalized["true_year"] = true_year
    normalized["false_year"] = false_year
    normalized.setdefault("when_question", f"When was the film {movie} released?")
    normalized.setdefault("why_fp_question", f"Why was the film {movie} released in {false_year}?")
    normalized.setdefault("fp_question_1", normalized["why_fp_question"])
    normalized.setdefault("fp_question_2", f"What was the film {movie} released in {false_year} describing about?")
    normalized.setdefault("fp_question_3", f"What was the {false_year} film {movie} about?")
    normalized.setdefault("fp_question_4", f"Who are the main characters in the {false_year} film {movie}?")
    if question_key not in normalized:
        normalized[question_key] = normalized["why_fp_question"]
    return normalized


def load_movie_samples(dataset_file, question_key, require_shared_year_prefix):
    path = find_dataset_file("movies", dataset_file)
    raw = read_json_or_jsonl(path)
    if isinstance(raw, dict):
        for key in ("train", "test", "validation", "data", "samples"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    samples = []
    for sample in raw:
        normalized = normalize_movie_sample(sample, question_key)
        if normalized is None:
            continue
        if len(normalized["time"]) != 1:
            continue
        if require_shared_year_prefix and str(normalized["true_year"])[:-1] != str(normalized["false_year"])[:-1]:
            continue
        samples.append(normalized)
    if not samples:
        raise ValueError(f"No usable movie samples found in {path}.")
    return samples, str(path)


def load_samples(args):
    question_key = args.question_key
    if question_key is None:
        question_key = "why_fp_question" if args.dataset == "movies" else "when_fp_question"
    if args.dataset == "movies":
        samples, source = load_movie_samples(args.dataset_file, question_key, args.movies_require_shared_year_prefix)
    else:
        samples, source = load_nobel_prize_samples(args.dataset_file, question_key)
    return samples, source, question_key


def infer_model_family(model, model_name_or_path: str, requested_family: str) -> str:
    if requested_family != "auto":
        family = requested_family
    else:
        model_type = getattr(model.config, "model_type", "") or ""
        text = f"{model_type} {model.__class__.__name__} {model_name_or_path}".lower()
        if "gemma4" in text or "gemma-4" in text:
            family = "gemma4"
        elif "olmo3" in text or "olmo-3" in text or "olmo_3" in text:
            family = "olmo3"
        elif "qwen3" in text or "qwen-3" in text:
            family = "qwen3"
        elif "qwen2.5" in text or "qwen2-5" in text or "qwen2_5" in text or "qwen2" in model_type:
            family = "qwen2.5"
        elif "llama-3.2" in text or "llama3.2" in text or ("llama" in model_type and "3.2" in text):
            family = "llama3.2"
        elif "meta-llama-3" in text or "meta_llama_3" in text or ("llama" in model_type and "3" in text):
            family = "llama3"
        else:
            raise ValueError(
                "Could not infer a supported model family. "
                f"Use --model-family with one of: {', '.join(SUPPORTED_FAMILIES)}."
            )

    validate_model_family(model, family)
    return family


def validate_model_family(model, family: str) -> None:
    model_type = getattr(model.config, "model_type", "") or ""
    class_name = model.__class__.__name__.lower()
    if family == "llama3.2" and "llama" not in model_type:
        text_model_type = getattr(get_text_config(model), "model_type", "") or ""
        if "llama" in text_model_type:
            return
        raise ValueError(f"--model-family llama3.2 requires a Llama config, got model_type={model_type!r}.")
    if family == "qwen2.5" and "qwen2" not in model_type:
        raise ValueError(f"--model-family qwen2.5 requires a Qwen2/Qwen2.5 config, got {model_type!r}.")
    if family == "qwen3" and "qwen3" not in model_type:
        raise ValueError(f"--model-family qwen3 requires a Qwen3 config, got {model_type!r}.")
    if family == "gemma4" and "gemma4" not in f"{model_type} {class_name}":
        raise ValueError(f"--model-family gemma4 requires a Gemma4 config, got {model_type!r}.")
    if family == "olmo3" and "olmo3" not in f"{model_type} {class_name}".replace("_", ""):
        raise ValueError(f"--model-family olmo3 requires an OLMo 3 config, got {model_type!r}.")


def get_module(model, module_name: str):
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"Could not find module {module_name!r} in {model.__class__.__name__}.")
    return modules[module_name]


def get_text_config(model):
    return getattr(model.config, "text_config", None) or model.config


def get_num_hidden_layers(model) -> int:
    return int(get_text_config(model).num_hidden_layers)


def get_num_attention_heads(model) -> int:
    return int(get_text_config(model).num_attention_heads)


def attention_o_proj_layers(model):
    num_layers = get_num_hidden_layers(model)
    candidates = [
        [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)],
        [f"language_model.layers.{i}.self_attn.o_proj" for i in range(num_layers)],
        [f"language_model.model.layers.{i}.self_attn.o_proj" for i in range(num_layers)],
        [f"model.language_model.layers.{i}.self_attn.o_proj" for i in range(num_layers)],
    ]
    modules = dict(model.named_modules())
    for layers in candidates:
        if all(layer in modules for layer in layers):
            return layers

    sparse_candidates = []
    for layers in candidates:
        existing = [layer for layer in layers if layer in modules]
        if existing:
            sparse_candidates.append(existing)
    if sparse_candidates:
        return max(sparse_candidates, key=len)

    suffix = ".self_attn.o_proj"
    discovered = [name for name in modules if name.endswith(suffix) and ".layers." in name]
    if discovered:
        return sorted(discovered, key=layer_index_from_hook)
    raise ValueError("Could not resolve attention o_proj layers for this model.")


def layer_index_from_hook(layer_name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", layer_name)
    if match is None:
        raise ValueError(f"Could not parse layer index from hook name {layer_name!r}.")
    return int(match.group(1))


def get_head_dim(model, first_layer_name: str) -> int:
    attn_name = first_layer_name.rsplit(".o_proj", 1)[0]
    attn = get_module(model, attn_name)
    if hasattr(attn, "head_dim"):
        return int(attn.head_dim)
    text_config = get_text_config(model)
    return int(getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads))


def load_generation_model(model_name_or_path: str, dtype: str, device_map: str):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs = {
        "torch_dtype": dtype_map[dtype],
        "device_map": device_map,
        "trust_remote_code": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs).eval()
    except ValueError as causal_error:
        for auto_cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            try:
                module = __import__("transformers", fromlist=[auto_cls_name])
                auto_cls = getattr(module, auto_cls_name)
            except (ImportError, AttributeError):
                continue
            try:
                return auto_cls.from_pretrained(model_name_or_path, **kwargs).eval()
            except ValueError:
                continue
        raise causal_error


def load_model_and_tokenizer(model_name_or_path: str, dtype: str, device_map: str):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    model = load_generation_model(model_name_or_path, dtype, device_map)
    return model, tokenizer


def predict_probs(model, inp):
    out = model(**inp)["logits"]
    return torch.softmax(out[:, -1], dim=1)


def first_divergent_year_token(tokenizer, true_year: int, false_year: int):
    true_token_ids = tokenizer([str(true_year)], return_tensors="pt")["input_ids"][0]
    false_token_ids = tokenizer([str(false_year)], return_tensors="pt")["input_ids"][0]

    if tokenizer.bos_token_id is not None and len(true_token_ids) and true_token_ids[0].item() == tokenizer.bos_token_id:
        true_token_ids = true_token_ids[1:]
    if tokenizer.bos_token_id is not None and len(false_token_ids) and false_token_ids[0].item() == tokenizer.bos_token_id:
        false_token_ids = false_token_ids[1:]

    common = []
    true_token_id = None
    for true_id, false_id in zip(true_token_ids, false_token_ids):
        if true_id.item() != false_id.item():
            true_token_id = int(true_id.item())
            break
        common.append(true_id)
    if true_token_id is None:
        raise ValueError(f"Could not find divergent token between {true_year} and {false_year}.")
    return tokenizer.decode([int(token.item()) for token in common]), true_token_id


def prepare_sample_for_dataset(sample, dataset, tokenizer):
    if dataset == "movies":
        true_year = int(sample["true_year"])
        false_year = int(sample["false_year"])
        common_prefix, ground_truth_token_id = first_divergent_year_token(tokenizer, true_year, false_year)
        return {
            "name": sample["movie"],
            "false_year": false_year,
            "true_year": true_year,
            "uncompleted_answer": f'Hello! I\'m here to help you answer your question. The film "{sample["movie"]}" was released in {common_prefix}',
            "ground_truth_token_id": ground_truth_token_id,
            "ground_truth_token": tokenizer.decode([int(ground_truth_token_id)]),
            "question_counter_factual": f"Why was the film {sample['movie']} released in XXXX?",
        }

    true_year = int(sample["awardYear"])
    false_year = true_year + 1
    common_prefix, ground_truth_token_id = first_divergent_year_token(tokenizer, true_year, false_year)
    return {
        "name": sample["name"],
        "false_year": false_year,
        "true_year": true_year,
        "uncompleted_answer": f"According to my knowledge, {sample['name']} won the {sample['categoryFullName']} in {common_prefix}",
        "ground_truth_token_id": ground_truth_token_id,
        "ground_truth_token": tokenizer.decode([int(ground_truth_token_id)]),
        "question_counter_factual": (
            f'For what specific contribution was {sample["name"]} '
            f'awarded {sample["categoryFullName"]} in XXXX?'
        ),
    }


def make_batch_input(tokenizer, prompts, device):
    batch_input = tokenizer(prompts, return_tensors="pt", padding=True)
    return {key: value.to(device) for key, value in batch_input.items()}


def calculate_path_patch(
    model,
    tokenizer,
    question_reference,
    question_counter_factual,
    uncompleted_answer,
    ground_truth_token,
    ground_truth_token_id,
    layers,
    threshold,
    prompt_style,
):
    num_heads = get_num_attention_heads(model)
    head_dim = get_head_dim(model, layers[0])

    prompts = [
        render_prompt(tokenizer, question_reference, uncompleted_answer, prompt_style),
        render_prompt(tokenizer, question_counter_factual, uncompleted_answer, prompt_style),
        render_prompt(tokenizer, question_reference, uncompleted_answer, prompt_style),
    ]
    batch_input = make_batch_input(tokenizer, prompts, model.device)

    with torch.no_grad():
        probs = predict_probs(model, batch_input)
        base_scores, preds = torch.max(probs, dim=1)
        answers_t = [preds[0], preds[1]]
        scores_t = [base_scores[0], base_scores[1]]

    predicted_token, counter_factual_token = decode_tokens(tokenizer, answers_t)
    if predicted_token == ground_truth_token:
        return None, f"reference already predicts the ground-truth token {ground_truth_token!r}"

    if counter_factual_token != ground_truth_token:
        init_ground_truth_prob = probs[0, ground_truth_token_id].item()
        counter_factual_ground_truth_prob = probs[1, ground_truth_token_id].item()
        if counter_factual_ground_truth_prob - init_ground_truth_prob < threshold:
            return None, (
                "counterfactual did not sufficiently raise ground-truth probability "
                f"({init_ground_truth_prob:.6f} -> {counter_factual_ground_truth_prob:.6f})"
            )

    _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0], question_reference)
    pos = end_of_question - 2

    differences = []
    for selected_layer in tqdm(layers, desc="layers"):
        row = []
        for selected_head in tqdm(range(num_heads), desc="heads", leave=False):
            dim_start = selected_head * head_dim
            dim_end = (selected_head + 1) * head_dim

            def patch_rep(x, layer):
                h = untuple(x)
                h[2, pos, :] = h[0, pos, :]
                if layer == selected_layer:
                    h[2, pos, dim_start:dim_end] = h[1, pos, dim_start:dim_end]
                return x

            with torch.no_grad(), TraceDict(model, layers, edit_input=patch_rep):
                out = model(**batch_input, output_hidden_states=True)
                patched_probs = torch.softmax(out["logits"][:, -1], dim=1)
                target_token_t = answers_t[1]
                init_score = patched_probs[0, target_token_t]
                after_score = patched_probs[2, target_token_t]
                row.append((after_score - init_score).item())
        differences.append(row)

    return {
        "differences": np.array(differences),
        "layer_indices": np.array([layer_index_from_hook(layer) for layer in layers], dtype=np.int64),
        "predicted_token": predicted_token,
        "counter_factual_token": counter_factual_token,
        "answers_t": np.array([elem.item() for elem in answers_t]),
        "base_scores": np.array([elem.item() for elem in scores_t]),
        "ground_truth_token": ground_truth_token,
        "ground_truth_token_id": int(ground_truth_token_id),
        "position": pos,
    }, None


def top_k_elements(matrix, k):
    flattened = matrix.flatten()
    sorted_indices = np.argsort(flattened)[::-1][:k]
    rows, cols = np.unravel_index(sorted_indices, matrix.shape)
    return [(int(row), int(col), float(flattened[idx])) for row, col, idx in zip(rows, cols, sorted_indices)]


def identify_heads(influence_dir: Path, top_k_per_sample: int, score_threshold: float, selected_top_k: int):
    important_heads = []
    skipped_low_score = 0
    files = sorted(influence_dir.glob("*.npz"))
    for path in files:
        result = dict(np.load(path, allow_pickle=True))
        score = result["differences"]
        if score.max() < score_threshold:
            skipped_low_score += 1
            continue
        layer_indices = result.get("layer_indices")
        if layer_indices is None:
            layer_indices = np.arange(score.shape[0])
        top_heads = [elem for elem in top_k_elements(score, top_k_per_sample) if elem[-1] > score_threshold]
        important_heads.extend(f"{int(layer_indices[layer])}-{head}" for layer, head, _ in top_heads)

    counts = Counter(important_heads)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], tuple(int(x) for x in item[0].split("-"))))
    selected = [
        {"layer": int(head.split("-")[0]), "head": int(head.split("-")[1]), "frequency": int(freq)}
        for head, freq in ranked[:selected_top_k]
    ]
    return {
        "num_influence_files": len(files),
        "num_low_score_files": skipped_low_score,
        "score_threshold": score_threshold,
        "top_k_per_sample": top_k_per_sample,
        "selected_top_k": selected_top_k,
        "ranked_heads": [
            {"layer": int(head.split("-")[0]), "head": int(head.split("-")[1]), "frequency": int(freq)}
            for head, freq in ranked
        ],
        "selected_heads": selected,
        "selected_heads_tuples": [[item["layer"], item["head"]] for item in selected],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="movies", choices=("movies", "nobel_prize"))
    parser.add_argument("--dataset-file", default=None, help="Optional dataset json/jsonl/csv file. Auto-detected by default.")
    parser.add_argument("--model-name-or-path", required=True, help="Transformers model name or local path.")
    parser.add_argument("--model-family", default="auto", choices=("auto",) + SUPPORTED_FAMILIES)
    parser.add_argument("--output-dir", required=True, help="Directory for influence files and selected heads.")
    parser.add_argument("--question-key", default=None)
    parser.add_argument("--prompt-style", default="chat_template", choices=("chat_template", "faith", "plain"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--torch-dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-counterfactual-increase", type=float, default=0.03)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--top-k-per-sample", type=int, default=20)
    parser.add_argument("--selected-top-k", type=int, default=20)
    parser.add_argument("--movies-require-shared-year-prefix", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    influence_dir = output_dir / "influences"
    influence_dir.mkdir(parents=True, exist_ok=True)

    samples, dataset_source, question_key = load_samples(args)
    if args.max_samples is not None:
        samples = samples[args.start_index : args.start_index + args.max_samples]
    else:
        samples = samples[args.start_index :]

    print(f"Loaded {len(samples)} {args.dataset} samples from {dataset_source}")
    print(f"Using question key: {question_key}")
    print(f"Loading model from {args.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.torch_dtype, args.device_map)
    family = infer_model_family(model, args.model_name_or_path, args.model_family)
    layers = attention_o_proj_layers(model)
    print(f"Using model family {family}; resolved {len(layers)} attention o_proj hooks.")

    metadata = {
        "dataset": args.dataset,
        "dataset_source": dataset_source,
        "model_name_or_path": args.model_name_or_path,
        "model_family": family,
        "question_key": question_key,
        "prompt_style": args.prompt_style,
        "num_samples_requested": len(samples),
        "num_layers": get_num_hidden_layers(model),
        "num_heads": get_num_attention_heads(model),
        "head_dim": get_head_dim(model, layers[0]),
        "layer_hooks": layers,
    }
    write_json(metadata, output_dir / "run_metadata.json")

    processed = 0
    skipped = []
    for local_idx, sample in enumerate(tqdm(samples, desc="samples")):
        idx = args.start_index + local_idx
        try:
            prepared = prepare_sample_for_dataset(sample, args.dataset, tokenizer)
        except Exception as exc:
            skipped.append({"index": idx, "name": sample.get("name") or sample.get("movie"), "reason": f"prepare: {exc}"})
            continue

        sample_name = sanitize_filename(prepared["name"])
        influence_file = influence_dir / f"{idx}_{sample_name}.npz"
        if influence_file.is_file() and not args.overwrite:
            processed += 1
            continue

        try:
            result, reason = calculate_path_patch(
                model=model,
                tokenizer=tokenizer,
                question_reference=sample[question_key],
                question_counter_factual=prepared["question_counter_factual"],
                uncompleted_answer=prepared["uncompleted_answer"],
                ground_truth_token=prepared["ground_truth_token"],
                ground_truth_token_id=prepared["ground_truth_token_id"],
                layers=layers,
                threshold=args.min_counterfactual_increase,
                prompt_style=args.prompt_style,
            )
        except Exception as exc:
            result = None
            reason = f"{type(exc).__name__}: {exc}"

        if result is None:
            skipped.append({"index": idx, "name": prepared["name"], "reason": reason})
            print(f"Skipping sample {idx} {prepared['name']}: {reason}")
            continue

        np.savez(influence_file, **result)
        processed += 1

    summary = identify_heads(
        influence_dir=influence_dir,
        top_k_per_sample=args.top_k_per_sample,
        score_threshold=args.score_threshold,
        selected_top_k=args.selected_top_k,
    )
    summary["num_processed_or_cached"] = processed
    summary["num_skipped"] = len(skipped)
    summary["skipped_samples"] = skipped
    write_json(summary, output_dir / "identified_false_premise_heads.json")

    print(f"Processed or reused {processed} influence files; skipped {len(skipped)} samples.")
    print(f"Wrote selected heads to {output_dir / 'identified_false_premise_heads.json'}")


if __name__ == "__main__":
    main()

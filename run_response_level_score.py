from evaluator import get_evaluator_cls
from constant.response_level_score import (
    DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL,
    RESPONSE_LEVEL_SCORE_KEY,
)
from constant.constant import SUPPORTED_DATASETS
import argparse, os, json
from collections import Counter
from evaluator import ResponseLevelScoreBatchJobCache
import pandas as pd

def run_print_examples(args: argparse.Namespace):
    dataset = pd.read_json(args.file, lines=True)
    dataset = dataset.query(args.query) if args.query else dataset
    out_path = os.path.join(args.out_dir, 'printed_examples.json')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        counter = 0
        data = []
        for _, dp in dataset.iterrows():
            if args.k > 0 and counter > args.k:
                break
            dp = dp.to_dict()
            dp.pop("few_shot_data", None)
            dp.pop("eval_few_shot_data", None)
            data.append(dp)
        json.dump(data, f, indent=4)
        counter += 1

def submit_response_level_score_batched(args: argparse.Namespace):
    """
    Submit batched Response Level Score evaluation job to Gemini API.
    For datapoints that do not require calling API, they will be directly saved to the output file.
    """
    EvaluatorClass = get_evaluator_cls(f'{args.dataset}ResponseLevelScoreBatchedEvaluator')
    evaluator = EvaluatorClass(
        evaluator_model_name=args.evaluator_model_name,
        thinking_cutoff_token=args.thinking_cutoff_token
    )
    with open(args.file, 'r') as f:
        dataset = [json.loads(line.strip()) for line in f]
    job_cache = evaluator.evaluate_batch(
        data=dataset,
        save_to=args.out_file,
        thinking_level=args.thinking_level
    )
    path = os.path.join(args.cache_dir, f'{job_cache.batch_job_name}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(job_cache.to_dict(), f, indent=4)

def submit_response_level_score(args: argparse.Namespace):
    """
    Submit Response Level Score evaluation job to Gemini API without batching.
    For datapoints that do not require calling API, they will be directly saved to the output file.
    """
    EvaluatorClass = get_evaluator_cls(f'{args.dataset}ResponseLevelScoreBatchedEvaluator')
    evaluator = EvaluatorClass(
        evaluator_model_name=args.evaluator_model_name,
        thinking_cutoff_token=args.thinking_cutoff_token
    )
    with open(args.file, 'r') as f:
        dataset = [json.loads(line.strip()) for line in f]
    evaluator.evaluate(
        data=dataset,
        save_to=args.out_file,
        thinking_level=args.thinking_level
    )

def run_gemini_model_check(args: argparse.Namespace):
    for root, _, filenames in os.walk(args.dir):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
            job_file = os.path.join(root, filename)
            with open(job_file, 'r') as f:
                job_info = json.load(f)
            try:
                job_info = ResponseLevelScoreBatchJobCache.from_dict(job_info)
                out_file = job_info.save_to
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                evaluator = job_info.EvaluatorClass()
                results = evaluator.checkback_and_parse(job_info)
                with open(out_file, 'w') as f:
                    for data in results:
                        f.write(json.dumps(data) + '\n')
                os.remove(job_file)
            except Exception as err:
                import traceback
                traceback.print_exc()
                print(f'file {job_file} error: {err}')
                continue

def run_print_results(args: argparse.Namespace):
    data = pd.read_json(args.file, lines=True)
    data = data.query(args.query) if args.query else data
    results = [dp[RESPONSE_LEVEL_SCORE_KEY] for _, dp in data.iterrows()]
    total = len(results)
    dist = Counter(results)
    proportions = {score: count / total for score, count in dist.items()}
    print(f'Proportions of response level scores: {proportions}')

def run_top_bottom_k(args: argparse.Namespace):
    with open(args.file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    sorted_data = sorted([dp for dp in data if RESPONSE_LEVEL_SCORE_KEY in dp], key=lambda x: x[RESPONSE_LEVEL_SCORE_KEY])
    with open(args.top_file, 'w') as f:
        for dp in sorted_data[-args.k:]:
            for key, value in dp.items():
                f.write(f'{key}: {value}\n')
            f.write('-' * 40 + '\n')
    with open(args.bottom_file, 'w') as f:
        for dp in sorted_data[:args.k]:
            for key, value in dp.items():
                f.write(f'{key}: {value}\n')
            f.write('-' * 40 + '\n')

def main(args: argparse.Namespace):
    if args.command == 'response_level_score_submit':
        if args.thinking_level == 'none':
            args.thinking_level = None
        if args.out_file is None:
            fnames = args.file.split('.')
            args.out_file = f"{'.'.join(fnames[:-1])}_response_level_score_evaluated.{fnames[-1]}"
        if args.disable_batching:
            submit_response_level_score(args)
        else:
            submit_response_level_score_batched(args)
    elif args.command == 'checkback':
        run_gemini_model_check(args)
    elif args.command == 'print_results':
        run_print_results(args)
    elif args.command == 'print_examples':
        run_print_examples(args)
    elif args.command == 'top_bottom_k':
        run_top_bottom_k(args)
    else:
        raise ValueError(f'Unknown command {args.command}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model outputs for answer.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    response_level_score_submit_parser = subparsers.add_parser('response_level_score_submit', help='Evaluate Score for model outputs (batched) using Gemini API. This will submit a batch job to Gemini API, need to manually check status later')
    response_level_score_submit_parser.add_argument('--file', type=str, required=True, help='File containing model outputs to evaluate')
    response_level_score_submit_parser.add_argument('--dataset', type=str, required=True, choices=SUPPORTED_DATASETS, help='Dataset name to use for evaluation')
    response_level_score_submit_parser.add_argument('--evaluator_model_name', type=str, default=DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL, help='Gemini model name used for response-level evaluation')
    response_level_score_submit_parser.add_argument('--thinking_level', type=str, default='minimal', choices=['minimal', 'low', 'medium', 'high', 'none'], help='Thinking level for evaluation, default to None to disable thinking')
    response_level_score_submit_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    response_level_score_submit_parser.add_argument('--cache_dir', type=str, default='tmp/response_level_score', help='Directory to save temporary batch job info')
    response_level_score_submit_parser.add_argument('--out_file', type=str, default=None, help='Output file to save the evaluated results, defaults to {--file}_response_level_score_evaluated.jsonl')
    response_level_score_submit_parser.add_argument('--thinking_cutoff_token', type=str, default=None, help='If specified, it will be used as the cutoff token for thinking, and only the content after this token will be used for evaluation.')
    response_level_score_submit_parser.add_argument('--disable_batching', action='store_true', help='Disable batching and directly evaluate all data points (not recommended, may cause timeout for large datasets)')
    
    checkback_parser = subparsers.add_parser('checkback', help='Check batch job status and download results for response level score batched evaluation')
    checkback_parser.add_argument('--dir', type=str, default='tmp/response_level_score', help='Check all batch jobs (stored as json format) in this directory')
    
    print_results_parser = subparsers.add_parser('print_results', help='Print evaluation results for model outputs')
    print_results_parser.add_argument('--file', type=str, required=True, help='File containing model outputs to evaluate')
    print_results_parser.add_argument('--query', type=str, default=None, help='Only print results that satisfy the query condition, e.g., "response_level_score == 0" to only print examples with response level score of 0')
    
    print_parser = subparsers.add_parser('print_examples', help='Print examples from dataset')
    print_parser.add_argument('--file', type=str, default=None, help='Path to the dataset file (JSONL format)')
    print_parser.add_argument('--k', type=int, default=-1, help='Number of examples to print, default to -1 to print all examples')
    print_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the printed examples')
    print_parser.add_argument('--query', type=str, default=None, help='Only print examples that satisfy the query condition, e.g., "response_level_score == 0" to only print examples with response level score of 0')
    
    top_bottom_k_parser = subparsers.add_parser('top_bottom_k', help='Print top and bottom k examples based on evaluation results')
    top_bottom_k_parser.add_argument('--file', type=str, required=True, help='File containing model outputs to evaluate')
    top_bottom_k_parser.add_argument('--k', type=int, default=5, help='Number of top and bottom examples to print')
    top_bottom_k_parser.add_argument('--top_file', type=str, default='top_k_examples.txt', help='Output file to save top k examples')
    top_bottom_k_parser.add_argument('--bottom_file', type=str, default='bottom_k_examples.txt', help='Output file to save bottom k examples')
    args = parser.parse_args()
    
    main(args)

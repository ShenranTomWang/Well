import argparse, json, os
from pathlib import Path
import torch
from pipeline_operator.check_operator.batch_job_cache import GeminiRunFactCheckBatchJobCache
from pipeline_operator.check_operator import MiniCheckOperator, GeminiCheckOperator, TransformersCheckOperator
from pipeline_operator import check_operator
import tqdm
from utils.argparse_utils import add_source_subparsers

def local_check(args: argparse.Namespace, operator: check_operator.CheckOperator):
    with open(args.file, 'r') as f:
        data = [json.loads(line) for line in f]
    data = data[args.start_idx:]
    if hasattr(args, 'disable_few_shot') and args.disable_few_shot:
        for dp in data:
            dp['few_shot_data'] = []
    for i, dp in tqdm.tqdm(enumerate(data), total=len(data)):
        dp = operator.check(
            dp,
            source=args.source_command,
            **vars(args)
        )
        data[i] = dp
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        for dp in data:
            f.write(json.dumps(dp) + '\n')

def minicheck(args: argparse.Namespace):
    assert torch.cuda.is_available(), "MiniCheck can only be run with CUDA device. Please check your environment and try again."
    operator = MiniCheckOperator(model_name=args.model_name, cache_dir=args.cache_dir, statement=args.statement)
    if args.out_file is None:
        fnames = args.file.split('.')
        args.out_file = f"{'.'.join(fnames[:-1])}_minichecked.{fnames[-1]}"
    path = Path(args.out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.start_idx == 0:
        open(path, 'w').close()
    local_check(args, operator)

def transformers_check(args: argparse.Namespace):
    assert torch.cuda.is_available(), "Transformers models can only be run with CUDA device. Please check your environment and try again."
    operator = TransformersCheckOperator(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        enable_thinking=args.enable_thinking,
        pipeline=args.pipeline,
        template_class=args.template_class
    )
    if args.out_file is None:
        fnames = args.file.split('.')
        args.out_file = f"{'.'.join(fnames[:-1])}_{args.model_name}_transformers_checked.{fnames[-1]}"
    path = Path(args.out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.start_idx == 0:
        open(path, 'w').close()
    local_check(args, operator)

def gemini_check_submit(args: argparse.Namespace):
    with open(args.file, 'r') as f:
        data = [json.loads(line) for line in f]
    data = data[args.start_idx:]
    if args.disable_few_shot:
        for dp in data:
            dp['few_shot_data'] = []
    operator = GeminiCheckOperator(pipeline=args.pipeline, template_class=args.template_class)
    if args.out_file is None:
        fnames = args.file.split('.')
        args.out_file = f"{'.'.join(fnames[:-1])}_{args.model_name}_gemini_checked.{fnames[-1]}"
    path = Path(args.out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.start_idx == 0:
        open(path, 'w').close()
    if args.disable_batching:
        local_check(args, operator)
    else:
        job_cache = operator.submit_job(
            data,
            save_to=args.out_file,
            source=args.source_command,
            **vars(args)
        )
        cache_file = f'{args.cache_dir}/{job_cache.batch_job_name}.json'
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(job_cache.to_dict(), f, indent=4)
        
def gemini_check_checkback(args: argparse.Namespace):
    for path, _, files in os.walk(args.cache_dir):
        for file in files:
            if file.endswith('.json'):
                try:
                    file_path = os.path.join(path, file)
                    with open(file_path, 'r') as f:
                        job_cache_dict = json.load(f)
                    job_cache = GeminiRunFactCheckBatchJobCache.from_dict(job_cache_dict)
                    operator = GeminiCheckOperator(pipeline=job_cache.pipeline)
                    operator.checkback(job_cache)
                    os.remove(file_path)
                except Exception as err:
                    print(f"Error processing file {file}: {err}")
                    continue
            
def main(args: argparse.Namespace):
    if args.command == "minicheck":
        minicheck(args)
    elif args.command == "transformers":
        transformers_check(args)
    elif args.command == "gemini_submit":
        gemini_check_submit(args)
    elif args.command == "checkback":
        gemini_check_checkback(args)
    else:
        raise ValueError(f"Unknown command {args.command}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    commands = parser.add_subparsers(dest="command", required=True, help="The operation to do.")
    minicheck_parser = commands.add_parser("minicheck", help="Run MiniCheck on the data.")
    minicheck_parser.add_argument("--file", type=str, required=True, help="Path to the file.jsonl to check.")
    minicheck_parser.add_argument('--out_file', type=str, default=None, help='Output file to save the minichecked results, defaults to {--file}_minichecked.jsonl')
    minicheck_parser.add_argument("--model_name", type=str, default="flan-t5-large", help="The MiniCheck model name.")
    minicheck_parser.add_argument("--cache_dir", type=str, default=os.getenv("HF_HOME"), help="The cache directory for the MiniCheck model.")
    minicheck_parser.add_argument("--start_idx", type=int, default=0, help="The start index of the data to check, used for splitting the data into multiple jobs.")
    minicheck_parser.add_argument("--check_gold", action='store_true', help="Whether to check the gold presuppositions instead of the model predictions. This requires the dataset to have gold presuppositions.")
    minicheck_parser.add_argument("--statement", action='store_true', help="Whether to check the converted question statement instead of presuppositions.")
    add_source_subparsers(minicheck_parser)
    
    transformers_check_parser = commands.add_parser("transformers", help="Run verification with transformers model on the data.")
    transformers_check_parser.add_argument("--file", type=str, required=True, help="Path to the file.jsonl to check.")
    transformers_check_parser.add_argument('--out_file', type=str, default=None, help='Output file to save the llmcheck results, defaults to {--file}_{--model}_transformers_checked.jsonl')
    transformers_check_parser.add_argument("--model_name", type=str, required=True, help="The transformers model name.")
    transformers_check_parser.add_argument("--device", type=str, default='auto', help="The device to run the model on.")
    transformers_check_parser.add_argument("--dtype", type=str, default='auto', choices=['float16', 'bfloat16', 'float32', 'auto'], help="The dtype to run the model with, e.g., float16, bfloat16, float32.")
    transformers_check_parser.add_argument("--start_idx", type=int, default=0, help="The start index of the data to check, used for splitting the data into multiple jobs.")
    transformers_check_parser.add_argument("--check_gold", action='store_true', help="Whether to check the gold presuppositions instead of the model predictions. This requires the dataset to have gold presuppositions.")
    transformers_check_parser.add_argument("--disable_few_shot", action='store_true', help='Whether to disable few-shot examples in the prompt')
    transformers_check_parser.add_argument("--enable_thinking", action='store_true', help='Whether to enable thinking (only available if the model supports it)')
    transformers_check_parser.add_argument("--pipeline", type=str, choices=['fact_check', 'feedback_action', 'statement'], default='fact_check', help="The pipeline to use for checking presuppositions.")
    transformers_check_parser.add_argument("--template_class", type=str, default=None, help="Template class to use for the transformers check, default to None to use the default template class for the pipeline")
    add_source_subparsers(transformers_check_parser)
    
    gemini_check_parser = commands.add_parser("gemini_submit", help="Run verification with gemini model on the data.")
    gemini_check_parser.add_argument("--file", type=str, required=True, help="Path to the file.jsonl to check.")
    gemini_check_parser.add_argument('--out_file', type=str, default=None, help='Output file to save the llmcheck results, defaults to {--file}_{--model_name}_gemini_checked.jsonl')
    gemini_check_parser.add_argument("--model_name", type=str, required=True, help="The gemini model name.")
    gemini_check_parser.add_argument("--cache_dir", type=str, default="./tmp/run_fact_check", help="The directory to save the gemini check job cache.")
    gemini_check_parser.add_argument("--start_idx", type=int, default=0, help="The start index of the data to check, used for splitting the data into multiple jobs.")
    gemini_check_parser.add_argument("--check_gold", action='store_true', help="Whether to check the gold presuppositions instead of the model predictions. This requires the dataset to have gold presuppositions.")
    gemini_check_parser.add_argument("--disable_few_shot", action='store_true', help='Whether to disable few-shot examples in the prompt')
    gemini_check_parser.add_argument("--disable_batching", action='store_true', help='Disable batching and directly run Gemini inference one example at a time')
    gemini_check_parser.add_argument('--web_search', action='store_true', help='Enable Gemini Google Search grounding')
    gemini_check_parser.add_argument('--thinking_level', type=str, default=None, help='Thinking level for Gemini model')
    gemini_check_parser.add_argument("--pipeline", type=str, choices=['fact_check', 'feedback_action', 'statement'], default='fact_check', help="The pipeline to use for checking presuppositions.")
    gemini_check_parser.add_argument("--template_class", type=str, default=None, help="Template class to use for the gemini check, default to None to use the default template class for the pipeline")
    add_source_subparsers(gemini_check_parser)
    
    gemini_check_checkback_parser = commands.add_parser("checkback", help="Checkback the gemini check job and save the results.")
    gemini_check_checkback_parser.add_argument("--cache_dir", type=str, default="./tmp/run_fact_check", help="The directory to search the gemini check job cache.")
    
    args = parser.parse_args()
    main(args)

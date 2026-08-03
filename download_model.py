from huggingface_hub import snapshot_download
import argparse, os

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, required=True, help='Name of the model to download from Hugging Face Hub')
argparser.add_argument('--directory', type=str, default=os.getenv('HF_HOME'), help='Local directory to save the downloaded model file')
args = argparser.parse_args()

full_path = os.path.join(args.directory, args.model_name.split('/')[-1])
os.makedirs(full_path, exist_ok=True)

path = snapshot_download(
    repo_id=args.model_name,
    local_dir=full_path,
    local_dir_use_symlinks=False,
    revision='main'
)
print(path)

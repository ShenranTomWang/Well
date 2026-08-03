import argparse

def add_source_subparsers(parser: argparse.ArgumentParser):
    source_command = parser.add_subparsers(dest="source_command", required=True, help=f"Whether to use RAG in this step.")
    passages_parser = source_command.add_parser("use_passages", help="Use all passages provided in the input data.")
    no_passages_parser = source_command.add_parser("no_passages", help="Do not use any passages for checking.")
    RAG_parser = source_command.add_parser("use_RAG", help="Use RAG to retrieve from passages provided in the input data.")
    RAG_parser.add_argument("--k", type=int, default=4, help="The number of passages to use.")
    RAG_parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use.")
    RAG_parser.add_argument("--RAG_model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="The model name to use for RAG embedding.")
    RAG_parser.add_argument("--RAG_device", type=str, default='cuda', help="The device to run RAG model on (for multi-GPU processing).")
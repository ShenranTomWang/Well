from google.genai import Client, types
import argparse

def main(args: argparse.Namespace):
    client = Client()
    for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=50)):
        if job.name in args.exceptions:
            print(f'Skipping batch job: {job.name}')
            continue
        client.batches.delete(name=job.name)
        print(f'Deleted batch job: {job.name}')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge all batch jobs.")
    parser.add_argument("--exceptions", default="", help="Comma-separated list of batch job names to exclude from deletion.")
    args = parser.parse_args()
    args.exceptions = set(args.exceptions.split(",")) if args.exceptions else set()
    main(args)
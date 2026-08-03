from google.genai import Client, types

if __name__ == "__main__":
    client = Client()
    for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
        print(job.name)
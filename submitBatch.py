from openai import OpenAI
import os
import anthropic
import argparse
import json

# Argument parser for directory
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory")
parser.add_argument("-m", "--model", choices={"gpt-4o", "gpt-4-turbo", "claude-sonnet", "claude-haiku"}, required=True)
parser.add_argument("-f", "--finetuning", action='store_true', help="Batch is for fine tuning")
args = parser.parse_args()

directory=("finetuning/" if args.finetuning else "results/")+args.directory

# Initialize client
if args.model in ["gpt-4o", "gpt-4-turbo"]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif args.model in ["claude-sonnet", "claude-haiku"]:
    client = anthropic.Anthropic(api_key = os.getenv("ANTHROPIC_API_KEY"))
else:
    raise ValueError("Invalid model type!")

if args.finetuning:
    if args.model != "gpt-4o":
        raise ValueError("Finetuning for models other than GPT 4o not supported")


# Get all batch_requests_*.jsonl files in the directory
batch_files = [f for f in os.listdir(directory) if f.startswith("batch_requests_"+args.model) and f.endswith(".jsonl")]

# Sort batch files to ensure proper order
batch_files.sort()

# Open a file to store all batch IDs
batchid_file_path = os.path.join(directory, "batchid_"+args.model)
with open(batchid_file_path, "w") as batchid_file:
    for batch_file in batch_files:
        # Full path to the batch file
        batch_file_path = os.path.join(directory, batch_file)
        
        if args.finetuning:
            # Create input file in OpenAI
            batch_input_file = client.files.create(
                file=open(batch_file_path, "rb"),
                purpose="fine-tune"
            )

            # Extract file ID from the response
            batch_input_file_id = batch_input_file.id

            # Create the fine tuning job
            created_batch = client.fine_tuning.jobs.create(
                training_file=batch_input_file_id,
                model="gpt-4o-2024-08-06",
            )

            # Check the job was created successfully
            print("-- Fine tuning jobs --")
            print(client.fine_tuning.jobs.list(limit=10))
        elif args.model in ["gpt-4o", "gpt-4-turbo"]:
            # Create input file in OpenAI
            batch_input_file = client.files.create(
                file=open(batch_file_path, "rb"),
                purpose="batch"
            )
            
            # Extract file ID from the response
            batch_input_file_id = batch_input_file.id
            
            # Create a batch using the input file
            created_batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "Visual Search"
                }
            )
        elif args.model in ["claude-sonnet", "claude-haiku"]:
            with open(batch_file_path, 'r') as file:
                batch=[]
                for line in file:
                    # Parse each line as JSON and append to the list
                    try:
                        data = json.loads(line.strip())
                        batch.append(data)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON line: {line.strip()}")
            created_batch = client.beta.messages.batches.create(requests=batch)

        # Write the batch ID to the batchid file
        batchid_file.write(created_batch.id + "\n")
        print(f"Batch submitted for {batch_file}, batch ID: {created_batch.id}")

print(f"All batch IDs saved to '{batchid_file_path}'.")

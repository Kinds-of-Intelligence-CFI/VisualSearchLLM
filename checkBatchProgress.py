from openai import OpenAI
import os
import argparse
import anthropic
import json


def extract_relevant_data(result):
    """
    Extracts only the relevant parts of the BetaMessageBatchIndividualResponse object.
    """
    try:
        return {
            "custom_id": result.custom_id,
            "result": {
                "type": result.result.type,
                "message": {
                    "id": result.result.message.id,
                    "content": [
                        {"text": block.text, "type": block.type}
                        for block in result.result.message.content
                    ],
                    "model": result.result.message.model,
                    "role": result.result.message.role,
                    "stop_reason": result.result.message.stop_reason,
                }
            }
        }
    except:
        return {
            "custom_id": result.custom_id,
            "result": {
            "type": result.result.type,
            "message": "Error"
            }
        }

# Argument parser for directory
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory")
parser.add_argument("-m", "--model", choices={"gpt-4o", "claude-sonnet", "gpt-4-turbo", "claude-haiku"}, required=True)
args = parser.parse_args()

directory="results/"+args.directory

# Initialize OpenAI client
if args.model in ["gpt-4o", "gpt-4-turbo"]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif args.model in ["claude-sonnet", "claude-haiku"]:
    client = anthropic.Anthropic(api_key = os.getenv("ANTHROPIC_API_KEY"))
else:
    raise ValueError("Invalid model type!")

# Read batch IDs from the batchid file
batchid_file_path = os.path.join(directory, "batchid_"+args.model)
with open(batchid_file_path, "r") as batchid_file:
    batch_ids = [line.strip() for line in batchid_file.readlines()]

# Check the status of all batches
all_batches_completed = True

for batch_id in batch_ids:
    if args.model in ["gpt-4o", "gpt-4-turbo"]:
        batch_info = client.batches.retrieve(batch_id)
        if not batch_info.status == "completed":
            print(f"Batch {batch_id} is still in progress: Completed {batch_info.request_counts.completed}/{batch_info.request_counts.total}")
            all_batches_completed = False
        elif batch_info.status == "completed":
            print(f"Batch {batch_id} is completed.")
        else:
            print(f"Batch {batch_id} status: {batch_info.status}")
            print(batch_info)

    elif args.model in ["claude-sonnet", "claude-haiku"]:
        batch_info=client.beta.messages.batches.retrieve(batch_id)
        if not batch_info.processing_status=="ended":
            all_batches_completed = False
            print(f"Batch {batch_id} is still in progress: Completed {batch_info.request_counts.succeeded}")
        elif batch_info.processing_status == "ended":
            print(f"Batch {batch_id} is completed.")
        else:
            print(f"Batch {batch_id} status: {batch_info.processing_status}")
            print(batch_info)

if not all_batches_completed:
    print("Not all batches are completed. Exiting the program.")
    exit(1)

# If all batches are completed, download and combine results
output_file_path = os.path.join(directory, args.model+"_combined_batch_responses.jsonl")

with open(output_file_path, "w", encoding="utf-8") as combined_file:
    for batch_id in batch_ids:
        # Retrieve batch info
        if args.model in ["gpt-4o", "gpt-4-turbo"]:
            batch_info = client.batches.retrieve(batch_id)
            # Check the output file ID and download the content
            file_id = batch_info.output_file_id
            file_response = client.files.content(file_id)
            
            # Write the content to the combined output file
            combined_file.write(file_response.text + "\n")
            print(f"Results from batch {batch_id} appended to {output_file_path}")

        elif args.model in ["claude-sonnet", "claude-haiku"]:

            results = client.beta.messages.batches.results(batch_id)
            for result in results:
                serialized_result = extract_relevant_data(result)
               # Convert the serialized object to a JSON string
                json_string = json.dumps(serialized_result, indent=None)
               # Write the JSON string to the file, followed by a newline
                combined_file.write(json_string + '\n')

    print(f"All batch results combined into '{output_file_path}'.")

# Optional: Validate the first few lines of the combined file
num_lines_to_print = 10
with open(output_file_path, "r", encoding="utf-8") as infile:
    print("First few lines of the combined file:")
    for i in range(num_lines_to_print):
        line = infile.readline()
        if not line:
            break
        print(f"Line {i+1}: {line.strip()}")

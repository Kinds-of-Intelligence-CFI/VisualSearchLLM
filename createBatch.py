import os
import base64
import json
from tqdm import tqdm
from constructMessage import constructMessage
import argparse

# Paths
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory")
parser.add_argument("-p", "--prompt", type=int)
parser.add_argument("-m", "--model", choices={"gpt-4o", "claude-sonnet"}, required=True)
args = parser.parse_args()

directory="results/"+args.directory

annotations_file = os.path.join(directory, 'annotations.csv')

# Function to encode images in Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Get the list of image filenames
image_filenames = [f for f in os.listdir(directory) if f.endswith('.png') and f.startswith('image_')]

# Sort the filenames numerically to ensure correct ordering
image_filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

batch_requests = []


if args.model == "gpt-4o":
    batch_limit=2500
else:
    batch_limit = 500
#batch_limit = 2500  # Maximum number of requests per file
file_count = 0  # Counter for file numbering
current_batch_count = 0  # Counter for current batch size



# Iterate over each image in the dataset with a progress bar
for filename in tqdm(image_filenames, desc='Processing images'):
    image_path = os.path.join(directory, filename)

    # Encode the image to Base64
    base64_image = encode_image(image_path)

    # Construct messages using the constructMessage function
    
    # Create the batch request JSON object
    if args.model == "gpt-4o":
        batch_request = {
            "custom_id": filename,  # Use the filename as the custom_id
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": constructMessage(args.prompt, "2", base64_image, args.model),
                "temperature": 0.0,
                "max_tokens": 1000
            }
        }
    elif args.model == "claude-sonnet":
        batch_request = {
            "custom_id": filename[0:-4],
            "params": {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 128,
            "system": "You are an AI assistant that can analyze images and answer questions about them.",
            "messages": constructMessage(args.prompt, "2", base64_image, args.model)
            }

        }


    # Add the batch request to the list
    batch_requests.append(batch_request)
    current_batch_count += 1

    # Check if the batch limit is reached
    if current_batch_count >= batch_limit:
        # Write the current batch to a file
        batch_file_path = os.path.join(directory, f"batch_requests_{args.model}_{file_count}.jsonl")
        with open(batch_file_path, 'w', encoding='utf-8') as outfile:
            for request in batch_requests:
                json_line = json.dumps(request)
                outfile.write(json_line + '\n')
        print(f"Batch file saved: {batch_file_path}")

        # Reset for the next batch
        batch_requests = []
        current_batch_count = 0
        file_count += 1

# Write remaining requests to a final batch file (if any)
if batch_requests:
    batch_file_path = os.path.join(directory, f"batch_requests_{args.model}_{file_count}.jsonl")
    with open(batch_file_path, 'w', encoding='utf-8') as outfile:
        for request in batch_requests:
            json_line = json.dumps(request)
            outfile.write(json_line + '\n')
    print(f"Final batch file saved: {batch_file_path}")

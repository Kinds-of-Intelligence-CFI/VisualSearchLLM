import os
import base64
import json
from tqdm import tqdm
from constructMessage import constructMessage
import argparse
import pandas as pd

colourMap = {"#FF0000": "red", "#00FF00": "green", "#0000FF": "blue", "#000000":"black"}

def encode_image(image_path):
    """Encodes the image at image_path to Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-f", "--finetuning", action='store_true', help="Create Finetuning batch")
    parser.add_argument("-m", "--model",
                        choices={"gpt-4o", "claude-sonnet", "llama11B", "llama90B",
                                 "llamaLocal", "gpt-4-turbo", "claude-haiku"},
                        required=True)
    parser.add_argument("-fmn", "--finetuned_model_name", default=None)
    args = parser.parse_args()

    # Decide on the directory (finetuning vs. results)
    if args.finetuning:
        directory = os.path.join("finetuning", args.directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        directory = os.path.join("results", args.directory)

    # Read the annotations CSV
    annotations_file = os.path.join(directory, 'annotations.csv')
    if not os.path.isfile(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    annotations = pd.read_csv(annotations_file)

    # -------------------------------------------------------------------------
    # Build a dictionary: filename -> a single target row's data
    # We replicate the original logic of using the FIRST row in 'targets'
    # or, if multiple exist, effectively the last one we see. But typically
    # you'd expect one row with target==True. If none exists, we'll store None.
    # -------------------------------------------------------------------------
    targets_by_filename = {}
    for _, row in annotations.iterrows():
        if row["target"] == True:
            fname = row["filename"]
            # If multiple target rows exist for the same file, this will
            # overwrite the previous one, matching the original .iloc[0] usage.
            targets_by_filename[fname] = {
                "color": row["color"],
                "shape_type": row["shape_type"],
                "row": row["row"],
                "column": row["column"]
            }

    # Gather the image filenames
    image_filenames = [
        f for f in os.listdir(directory)
        if f.endswith('.png') and f.startswith('image_')
    ]
    # Sort them numerically so image_1.png, image_2.png, etc. are in order
    image_filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Set batch limits
    if args.model in {"gpt-4o", "gpt-4-turbo"}:
        batch_limit = 2500
    elif args.model == "claude-sonnet":
        batch_limit = 500
    else:
        batch_limit = 5000

    batch_requests = []
    file_count = 0
    current_batch_count = 0

    if args.finetuned_model_name is not None:
        print(f"Using finetuned model: {args.finetuned_model_name}")

    # Loop over each image
    for filename in tqdm(image_filenames, desc='Processing images'):
        image_path = os.path.join(directory, filename)

        # Look up our precomputed target info in O(1)
        target_data = targets_by_filename.get(filename, None)

        # If no target data, replicate original logic: empty .iloc
        if target_data is None:
            targetColour = None
            targetShape = None
            # For finetuning, there's no row/col
            cell = None
        else:
            # We replicate "targets['color'].iloc[0]" => target_data["color"]
            raw_color = target_data["color"]
            targetColour = colourMap[raw_color]
            targetShape = target_data["shape_type"]
            # For finetuning, we want row/column
            cell = (target_data["row"], target_data["column"])

        # Encode image
        base64_image = encode_image(image_path)

        # If finetuning but there's no target row, cell remains None, same as original
        if args.finetuning:
            # Because in the old code, they'd do:
            #    solution = annotations[(filename, target==True)]
            #    if not empty, cell = ...
            # If it's empty, cell = None
            # We already have that logic above. So do nothing extra here.
            pass
        else:
            cell = None

        # Now construct the request exactly as you did before
        if args.finetuning and args.model == "gpt-4o":
            batch_request = {
                "messages": constructMessage(
                    args.prompt,
                    targetShape,
                    targetColour,
                    base64_image,
                    args.model,
                    finetuning=args.finetuning,
                    cell=cell
                )
            }
        elif args.finetuning:
            # The original code raises an error if finetuning & not GPT-4o
            raise ValueError("Fine tuning not implemented for models other than GPT-4o")

        elif args.model == "gpt-4o":
            batch_request = {
                "custom_id": filename,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": ("gpt-4o-2024-08-06"
                              if args.finetuned_model_name is None
                              else args.finetuned_model_name),
                    "messages": constructMessage(
                        args.prompt,
                        targetShape,
                        targetColour,
                        base64_image,
                        args.model
                    ),
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            }


        elif args.model == "gpt-4-turbo":
            batch_request = {
                "custom_id": filename,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4-turbo-2024-04-09",
                    "messages": constructMessage(
                        args.prompt,
                        targetShape,
                        targetColour,
                        base64_image,
                        args.model
                    ),
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            }
        elif args.model == "claude-sonnet":
            batch_request = {
                "custom_id": filename[:-4],  # drop ".png"
                "params": {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 128,
                    "system": (
                        "You are an AI assistant that can analyze images "
                        "and answer questions about them."
                    ),
                    "messages": constructMessage(
                        args.prompt,
                        targetShape,
                        targetColour,
                        base64_image,
                        args.model
                    )
                }
            }
        elif args.model == "claude-haiku":
            batch_request = {
                "custom_id": filename[:-4],  # drop ".png"
                "params": {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 128,
                    "system": (
                        "You are an AI assistant that can analyze images "
                        "and answer questions about them."
                    ),
                    "messages": constructMessage(
                        args.prompt,
                        targetShape,
                        targetColour,
                        base64_image,
                        args.model
                    )
                }
            }
        elif args.model in {"llama11B", "llama90B", "llamaLocal"}:
            batch_request = {
                "custom_id": filename,
                "messages": constructMessage(
                    args.prompt,
                    targetShape,
                    targetColour,
                    base64_image,
                    args.model
                )
            }
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        # Add this request to the batch
        batch_requests.append(batch_request)
        current_batch_count += 1

        # If we hit batch_limit, write out a chunk
        if current_batch_count >= batch_limit:
            batch_file_path = os.path.join(directory,
                                           f"batch_requests_{args.model}_{file_count}.jsonl")
            with open(batch_file_path, 'w', encoding='utf-8') as outfile:
                for request in batch_requests:
                    json_line = json.dumps(request)
                    outfile.write(json_line + '\n')

            print(f"Batch file saved: {batch_file_path}")
            batch_requests = []
            current_batch_count = 0
            file_count += 1

    # Write any leftover requests
    if batch_requests:
        batch_file_path = os.path.join(directory,
                                       f"batch_requests_{args.model}_{file_count}.jsonl")
        with open(batch_file_path, 'w', encoding='utf-8') as outfile:
            for request in batch_requests:
                json_line = json.dumps(request)
                outfile.write(json_line + '\n')

        print(f"Final batch file saved: {batch_file_path}")

if __name__ == "__main__":
    main()

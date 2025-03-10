#!/usr/bin/env python3
import argparse
import glob
import os

# Parse command-line arguments for the model name and the directory where the output files are.
parser = argparse.ArgumentParser(description="Combine JSONL response files into one output file.")
parser.add_argument("-m", "--model", required=True, help="Model name, e.g., Llama11B")
parser.add_argument("-d ", "--directory", default=".", help="Directory containing the response files")
args = parser.parse_args()

model = args.model
directory = args.directory

# Build a pattern to find all files with the format <model>Responses*.jsonl
pattern = os.path.join("results/"+directory, f"{model}_responses*.jsonl")
response_files = sorted(glob.glob(pattern))

if not response_files:
    print(f"No response files found with pattern {pattern}")
    exit(1)

combined_filename = os.path.join("results/"+directory, f"{model}_combined_batch_responses.jsonl")

# Concatenate the files line-by-line. This works well for JSONL files.
with open(combined_filename, "w") as outfile:
    for filename in response_files:
        with open(filename, "r") as infile:
            for line in infile:
                outfile.write(line)

print(f"Combined file created: {combined_filename}")

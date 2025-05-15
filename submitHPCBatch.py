#!/usr/bin/env python3
import subprocess
import os
import glob
import sys
import argparse
import re

# Parse command-line arguments for DIRECTORY and MODEL.
parser = argparse.ArgumentParser(
    description="Submit SLURM jobs for each batch request file found in DIRECTORY."
)
parser.add_argument("-d","--directory", required=True,
                    help="Directory containing the batch request files")
parser.add_argument("-m", "--model", required=True,
                    help="Model name to pass to the job submission")
args = parser.parse_args()

directory = args.directory
model = args.model

# Automatically find all batch request files in the given directory matching the pattern.
pattern = os.path.join("results/"+directory, "batch_requests_llamaLocal_*.jsonl")

request_files = sorted(glob.glob(pattern))

if not request_files:
    print(f"No batch request files found in {directory} matching {pattern}")
    sys.exit(1)

# Build the list of parameter sets.
param_sets = []
for request_file in request_files:
    base = os.path.basename(request_file)
    # Extract the number from the batch request filename.
    match = re.search(r'batch_requests_llamaLocal_(\d+)\.jsonl', base)
    if match:
        output_file = f"{model}_responses{match.group(1)}.jsonl"
    else:
        output_file = "responses.jsonl"  # Fallback if pattern doesn't match.
    param_sets.append({
        "DIRECTORY": directory,
        "MODEL": model,
        "REQUESTS": base,  # Use the base filename; change to full path if necessary.
        "OUTPUT": output_file
    })

# Submit a SLURM job for each parameter set.
for params in param_sets:
    export_str = ",".join([f"{key}={value}" for key, value in params.items()])
    print(f"Submitting job with: {export_str}")
    subprocess.run(["sbatch", "--export", export_str, "llamaSlurm.sh"])

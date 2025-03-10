#!/usr/bin/env python3
import subprocess
import os
import glob
import sys
import argparse

# Parse command-line arguments for DIRECTORY and MODEL.
parser = argparse.ArgumentParser(
    description="Submit SLURM jobs for each batch request file found in DIRECTORY."
)
parser.add_argument("--directory", required=True,
                    help="Directory containing the batch request files")
parser.add_argument("--model", required=True,
                    help="Model name to pass to the job submission")
args = parser.parse_args()

directory = args.directory
model = args.model

# Automatically find all batch request files in the given directory matching the pattern.
pattern = os.path.join(directory, "batch_requests_llamaLocal_*.jsonl")
request_files = sorted(glob.glob(pattern))

if not request_files:
    print(f"No batch request files found in {directory} matching {pattern}")
    sys.exit(1)

# Build the list of parameter sets.
param_sets = []
for request_file in request_files:
    param_sets.append({
        "DIRECTORY": directory,
        "MODEL": model,
        "REQUESTS": os.path.basename(request_file)  # or use full path if needed
    })

# Submit a SLURM job for each parameter set.
for params in param_sets:
    export_str = ",".join([f"{key}={value}" for key, value in params.items()])
    print(f"Submitting job with: {export_str}")
    subprocess.run(["sbatch", "--export", export_str, "batch_script.sh"])

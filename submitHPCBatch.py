#!/usr/bin/env python3
import subprocess

param_sets = [
    {"DIRECTORY": "2Among5-rc-10k", "MODEL": "llama11B", "REQUESTS": "batch_requests_llamaLocal_0.jsonl"},
    {"DIRECTORY": "2Among5-rc-10k", "MODEL": "llama11B", "REQUESTS": "batch_requests_llamaLocal_1.jsonl"},
    # More parameter sets...
]

for params in param_sets:
    # Build the export string: e.g. DATASET=2Among5-rc-10k,MODEL=llama90B
    export_str = ",".join([f"{key}={value}" for key, value in params.items()])
    # Submit the job, exporting the environment variables.
    subprocess.run(["sbatch", "--export", export_str, "batch_script.sh"])

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
parser.add_argument("-m", "--model", choices={"llama11B", "llama90B"}, required=True,
                    help="Model name to pass to the job submission")
parser.add_argument("--dry-run", action="store_true",
                    help="Show what would be submitted without actually submitting")
args = parser.parse_args()

directory = args.directory
model = args.model
dry_run = args.dry_run

print(f"🔍 Looking for batch files in directory: {directory}")
print(f"🎯 Model: {model}")
print(f"📁 Pattern: results/{directory}/batch_requests_llamaLocal_*.jsonl")

# Automatically find all batch request files in the given directory matching the pattern.
pattern = os.path.join("results/"+directory, f"batch_requests_llamaLocal_*.jsonl")

request_files = sorted(glob.glob(pattern))

if not request_files:
    print(f" No batch request files found in {directory} matching {pattern}")
    print(f" Make sure you've run: python createBatch.py -d {directory} -m {model} -p <prompt>")
    sys.exit(1)

print(f"✅ Found {len(request_files)} batch file(s):")
for f in request_files:
    print(f"   📄 {os.path.basename(f)}")

# Build the list of parameter sets.
param_sets = []
for request_file in request_files:
    base = os.path.basename(request_file)
    # Extract the number from the batch request filename.
    match = re.search(f'batch_requests_{model}_(\\d+)\\.jsonl', base)
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

print(f"\n🚀 Preparing to submit {len(param_sets)} SLURM job(s)...")

if dry_run:
    print("\n🔍 DRY RUN - No jobs will be submitted")
    print("To actually submit, run without --dry-run")

# Submit a SLURM job for each parameter set.
for i, params in enumerate(param_sets):
    export_str = ",".join([f"{key}={value}" for key, value in params.items()])
    
    print(f"\n📋 Job {i+1}/{len(param_sets)}:")
    print(f"   Directory: {params['DIRECTORY']}")
    print(f"   Model: {params['MODEL']}")
    print(f"   Batch file: {params['REQUESTS']}")
    print(f"   Output file: {params['OUTPUT']}")
    
    if dry_run:
        print(f"   Would submit: sbatch --export {export_str} llamaSlurm.sh")
    else:
        print(f"   Submitting to SLURM...")
        try:
            result = subprocess.run(
                ["sbatch", "--export", export_str, "llamaSlurm.sh"], 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # Extract job ID from SLURM output
            job_id = None
            for line in result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    break
            
            if job_id:
                print(f"   ✅ Job submitted successfully! Job ID: {job_id}")
                print(f"   📊 Check status with: squeue -j {job_id}")
                print(f"   📝 Monitor logs with: tail -f logs/llama_{job_id}.out")
            else:
                print(f"   ⚠️  Job submitted but couldn't extract job ID")
                print(f"   📝 SLURM output: {result.stdout.strip()}")
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to submit job: {e}")
            print(f"   📝 Error output: {e.stderr.strip()}")
        except FileNotFoundError:
            print(f"   ❌ SLURM not available (sbatch command not found)")
            print(f"   💡 Make sure you're on a compute node or have SLURM installed")

if not dry_run:
    print(f"\n🎉 All {len(param_sets)} job(s) submitted!")
    print(f"📊 Monitor your jobs with: squeue -u $USER")
    print(f"📝 Check logs in the logs/ directory")
    print(f"💾 Activations will be saved to RDS6: /rds-d6/user/mm2833/hpc-work/VisualSearchLLM/checkpoints/{directory}/")
else:
    print(f"\n🔍 DRY RUN complete. {len(param_sets)} job(s) would be submitted.")

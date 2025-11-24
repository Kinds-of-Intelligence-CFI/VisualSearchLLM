import argparse
import subprocess
import os
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="File containing list of directories to process")
    parser.add_argument("-m", "--model", required=True, help="Model to use (e.g., Qwen, llama11B)")
    parser.add_argument("-p", "--prompt", default="std2x2-2Among5", help="Prompt type for createBatch.py")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    with open(args.file, 'r') as f:
        # Parse lines as "directory,prompt" or just "directory"
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                entries.append((parts[0].strip(), parts[1].strip()))
            else:
                entries.append((parts[0].strip(), args.prompt))

    print(f"Found {len(entries)} directories to process.")

    for i, (directory, prompt) in enumerate(entries):
        print(f"\n[{i+1}/{len(entries)}] Processing directory: {directory} with prompt: {prompt}")
        
        # Step 1: Create Batch File
        # We assume we need to create it. If it already exists, createBatch.py typically overwrites or we can check.
        # createBatch.py logic: it generates batch_requests_{model}_{count}.jsonl
        # We'll run it to be safe.
        print(f"  Generating batch file for {directory}...")
        try:
            subprocess.run(
                ["python3.13", "createBatch.py", "-d", directory, "-p", prompt, "-m", args.model],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  Error creating batch for {directory}: {e}")
            continue

        # Step 2: Run Fireworks
        # We need to know the batch file name. createBatch.py generates 'batch_requests_{model}_0.jsonl' usually.
        # If there are multiple (limit 5000), we might need to handle that.
        # For now, let's assume one batch file or iterate.
        # createBatch.py output: "Batch file saved: results/{directory}/batch_requests_{model}_{count}.jsonl"
        
        # Let's look for files matching the pattern
        results_dir = os.path.join("results", directory)
        batch_files = [f for f in os.listdir(results_dir) if f.startswith(f"batch_requests_{args.model}_") and f.endswith(".jsonl")]
        batch_files.sort() # Ensure order 0, 1, 2...
        
        if not batch_files:
            print(f"  No batch files found for {directory} and model {args.model}")
            continue
            
        for batch_file in batch_files:
            print(f"  Running inference for {batch_file}...")
            output_file = batch_file.replace("batch_requests_", "responses_")
            
            try:
                subprocess.run(
                    ["python3.13", "runFireworks.py", "-d", directory, "-m", args.model, "-b", batch_file, "-o", output_file],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"  Error running inference for {batch_file}: {e}")
                # We continue to next batch file or directory?
                # Probably continue to next batch file.
                continue
                
    print("\nAll jobs completed.")

if __name__ == "__main__":
    main()

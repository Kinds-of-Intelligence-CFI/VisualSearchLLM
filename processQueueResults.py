import argparse
import subprocess
import os

def get_mode_flag(prompt):
    if prompt.startswith("std2x2") or prompt == "circle-sizes" or prompt == "LightPriorsOOO":
        return "-rc"
    elif prompt.startswith("coords"):
        return "-c"
    else:
        print(f"Warning: Unknown prompt type '{prompt}', defaulting to -rc")
        return "-rc"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="File containing list of directories to process")
    parser.add_argument("-m", "--model", required=True, help="Model used (e.g., Qwen7B)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    with open(args.file, 'r') as f:
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                entries.append((parts[0].strip(), parts[1].strip()))
            else:
                print(f"Warning: No prompt found for {parts[0]}, skipping mode inference. Using default -rc.")
                entries.append((parts[0].strip(), "std2x2-default"))

    print(f"Found {len(entries)} directories to process for model {args.model}.")

    for i, (directory, prompt) in enumerate(entries):
        print(f"\n[{i+1}/{len(entries)}] Processing results for: {directory}")
        
        mode_flag = get_mode_flag(prompt)
        batch_responses_file = f"responses_{args.model}_0.jsonl"
        
        full_path = os.path.join("results", directory, batch_responses_file)
        if not os.path.exists(full_path):
            print(f"  File not found: {full_path}. Skipping.")
            continue
            
        print(f"  Running processBatchResults.py with mode {mode_flag}...")
        try:
            subprocess.run(
                ["python3.13", "processBatchResults.py", 
                 "-d", directory, 
                 "-m", args.model, 
                 "-b", batch_responses_file,
                 "--exact_filename",
                 mode_flag],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  Error processing results for {directory}: {e}")
            continue
            
    print("\nAll processing jobs completed.")

if __name__ == "__main__":
    main()

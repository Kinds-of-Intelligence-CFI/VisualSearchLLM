import subprocess
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-n", "--number", type=int, default=1000)
    parser.add_argument("-f", "--finetuning", action="store_true")
    parser.add_argument("-pr", "--preset", required=True)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-g", "--gpt4o", action="store_true")
    parser.add_argument("-c" "--claude", action="store_true")
    parser.add_argument("-l", "llama", action="store_true")
    parser.parse_args()


 	subprocess.run(["python3", "generate_images.py", "--d", parser.directory, "-n", parser.number, "-pr", parser.preset])
    if parser.gpt4o:
        subprocess.run(["python3", "createBatch.py", "-d", parser.directory, "-m", "gpt-4o", "-p", args.prompt])
        subprocess.run(["python3", "submitBatch.py", "-d", parser.directory, "-m", "gpt-4o"])
    if parser.claude:
        subprocess.run(["python3", "createBatch.py", "-d", parser.directory, "-m", "claude-sonnet", "-p", args.prompt])
        subprocess.run(["python3", "createBatch.py", "-d", parser.directory, "-m", "claude-sonnet"])
    if parser.llama:
        subprocess.run(["python3", "createBatch.py", "-d", parser.directory, "-m", "llamaLocal", "-p", args.prompt])
        subprocess.run(["python3", "submitHPCbatch.py", "-d", parser.directory, "-m", "llama11B"])
        subprocess.run(["python3", "submitHPCbatch.py", "-d", parser.directory, "-m", "llama90B"])
    print("Finished")


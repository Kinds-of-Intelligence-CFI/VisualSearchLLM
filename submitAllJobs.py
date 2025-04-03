import subprocess
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-dn", "--distractors", type=int, default=-1)
    parser.add_argument("-n", "--number", type=int, default=1000)
    parser.add_argument("-f", "--finetuning", action="store_true")
    parser.add_argument("-pr", "--preset", required=True)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-g", "--gpt4o", action="store_true")
    parser.add_argument("-c", "--claude", action="store_true")
    parser.add_argument("-l", "--llama", action="store_true")
    args=parser.parse_args()

    print(args.prompt)

    if not args.distractors == -1:
        distractorArgs = ["-dn", str(args.distractors)]
    else:
        distractorArgs = []
    subprocess.run(["python3", "generateImages.py", "-d", args.directory, "-n", str(args.number), "-p", args.preset]+distractorArgs)
    if args.gpt4o:
        subprocess.run(["python3", "createBatch.py", "-d", args.directory, "-m", "gpt-4o", "-p", args.prompt])
        subprocess.run(["python3", "submitBatch.py", "-d", args.directory, "-m", "gpt-4o"])
    if args.claude:
        subprocess.run(["python3", "createBatch.py", "-d", args.directory, "-m", "claude-sonnet", "-p", args.prompt])
        subprocess.run(["python3", "submitBatch.py", "-d", args.directory, "-m", "claude-sonnet"])
    if args.llama:
        subprocess.run(["python3", "createBatch.py", "-d", args.directory, "-m", "llamaLocal", "-p", args.prompt])
        subprocess.run(["python3", "submitHPCBatch.py", "-d", args.directory, "-m", "llama11B"])
        subprocess.run(["python3", "submitHPCBatch.py", "-d", args.directory, "-m", "llama90B"])
    print("Finished")


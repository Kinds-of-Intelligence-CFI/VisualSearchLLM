import subprocess
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpt4o", action="store_true")
    parser.add_argument("-c" "--claude", action="store_true")
    parser.add_argument("-l", "--llama", action="store_true")
    parser.add_argument("-rc", "--rowsandcolumns", action="store_true")
    parser.add_argument("-q", "--quadrants", action="store_true")
    parser.add_argument("-c", "--coordinates", action="store_true")
    args=parser.parse_args()


    processFlag = ""
    if args.rowsandcolumns:
        processFlag="-rc"
    elif args.quadrants:
        processFlag="-q"
    elif args.coordinates:
        processFlag="-c"
 
    if args.gpt4o:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "gpt-4o", processFlag])
    if parser.claude:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "claude-sonnet", processFlag])
  
    if parser.llama:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "llama11B", processFlag])
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "llama90B", processFlag])
    print("Finished")


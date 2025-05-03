import subprocess
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-g", "--gpt4o", action="store_true")
    parser.add_argument("-g4", "--gpt4turbo", action="store_true")
    parser.add_argument("-cs", "--claudesonnet", action="store_true")
    parser.add_argument("-cs37", "--claudesonnet37", action="store_true")
    parser.add_argument("-ch", "--claudehaiku", action="store_true")
    parser.add_argument("-l", "--llama", action="store_true")
    parser.add_argument("-rc", "--rowsandcolumns", action="store_true")
    parser.add_argument("-q", "--quadrants", action="store_true")
    parser.add_argument("-co", "--coordinates", action="store_true")
    parser.add_argument("-p", "--presence", action="store_true")
    args=parser.parse_args()


    processFlag = ""
    if args.rowsandcolumns:
        processFlag="-rc"
    elif args.quadrants:
        processFlag="-q"
    elif args.coordinates:
        processFlag="-c"
    elif args.presence:
        processFlag="-p"
 
    if args.gpt4o:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "gpt-4o", processFlag])
    if args.gpt4turbo:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "gpt-4-turbo", processFlag])
    if args.claudesonnet:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "claude-sonnet", processFlag])
    if args.claudesonnet37:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "claude-sonnet37", processFlag])
  
    if args.claudehaiku:
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "claude-haiku", processFlag])

    if args.llama:
        subprocess.run(["python3", "combineLlama.py", "-d", args.directory, "-m", "llama11B"])
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "llama11B", processFlag])
        subprocess.run(["python3", "combineLlama.py", "-d", args.directory, "-m", "llama90B"])
        subprocess.run(["python3", "processBatchResults.py", "-d", args.directory, "-m", "llama90B", processFlag])
    print("Finished")


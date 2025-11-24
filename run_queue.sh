#!/bin/bash

# Usage: ./run_queue.sh <directories_file> <model> [default_prompt]
# The directories_file can contain "directory" or "directory,prompt" per line.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <directories_file> <model> [prompt]"
    exit 1
fi

DIR_FILE=$1
MODEL=$2
PROMPT=${3:-"std2x2-2Among5"}

LOG_FILE="queue_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting queue for $MODEL using directories in $DIR_FILE"
echo "Logging to $LOG_FILE"
echo "You can disconnect now. The process will continue running."

nohup python3.13 queueFireworks.py -f "$DIR_FILE" -m "$MODEL" -p "$PROMPT" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process ID: $PID"

#!/bin/bash

# Usage: ./process_queue.sh <directories_file> <model>

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <directories_file> <model>"
    exit 1
fi

DIR_FILE=$1
MODEL=$2

echo "Starting result processing for $MODEL using directories in $DIR_FILE"

python3.13 processQueueResults.py -f "$DIR_FILE" -m "$MODEL"

#!/bin/bash
#SBATCH -A BURDEN-MECHINT-SL2-GPU

#SBATCH --job-name=llama_activations
#SBATCH --output=logs/llama_%j.out
#SBATCH --error=logs/llama_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm2833@cam.ac.uk

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (following John Burden project pattern)
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

###########
module load python/3.8 cuda/11.8 cudnn/8.9_cuda-11.8 
source /rds/user/mm2833/hpc-work/venv_py38/bin/activate

###########

export HF_HOME="/rds-d6/user/mm2833/hpc-work/.cache/huggingface"
export TRANSFORMERS_CACHE="/rds-d6/user/mm2833/hpc-work/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/rds-d6/user/mm2833/hpc-work/.cache/huggingface/datasets"
export HF_TOKEN="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"

mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
export HF_TOKEN="$HF_TOKEN"

# Print environment info
echo "🐍 Python version: $(python --version)"
echo "🔧 CUDA version: $(nvcc --version | grep release)"
echo "💾 HF cache location: $HF_HOME"
echo "🎯 Model: $MODEL"
echo "📁 Directory: $DIRECTORY"
echo "📄 Batch file: $REQUESTS"
echo "📤 Output file: $OUTPUT"


# Run the LLaMA script
python runLlama.py -d "$DIRECTORY" -m "$MODEL" -b "$REQUESTS" -o "$OUTPUT"

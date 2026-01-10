#!/bin/bash
#SBATCH --job-name=hyenadna-extract-embeddings-container
#SBATCH --account=project_2010726
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

# HyenaDNA Embedding Extraction with Container
# Extracts embeddings from DNA sequences using HyenaDNA model

echo "================================================================"
echo "🧬 HYNEDNA EMBEDDING EXTRACTION (CONTAINER)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Set working directory
SCRATCH_DIR="/scratch/project_2010726/hyenadna_analysis"
cd $SCRATCH_DIR

# Check if container exists
CONTAINER_PATH="/projappl/project_2010726/HyenaDNA/HyenaDNA.sif"
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "❌ Container $CONTAINER_PATH not found!"
    echo "Please build the container first"
    exit 1
fi

# Check if data is prepared
if [ ! -f "data/processed/data_train.pkl" ]; then
    echo "❌ Processed data not found!"
    echo "Please run: sbatch slurm_scripts/run-prepare-data-container.sh first"
    exit 1
fi

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Create necessary directories
mkdir -p embeddings
mkdir -p logs

# Set environment variables for HuggingFace cache
export TRANSFORMERS_CACHE=/tmp/huggingface_cache
mkdir -p $TRANSFORMERS_CACHE

echo "Extracting HyenaDNA embeddings using container..."
echo ""

# Run embedding extraction in container with GPU support
apptainer exec --nv $CONTAINER_PATH python3 scripts/extract_hyenadna_embeddings.py \
    --model_name "hyenadna-medium-160k-seqlen" \
    --data_dir "data/processed" \
    --output_dir "embeddings" \
    --batch_size 2 \
    --pooling_strategy "mean" \
    --device "cuda"

echo ""
echo "================================================================"
echo "✅ EMBEDDING EXTRACTION COMPLETE"
echo "================================================================"
echo "Time: $(date)"
echo ""

# Show results
echo "📊 RESULTS SUMMARY:"
echo "Embedding files:"
ls -la embeddings/

echo ""
echo "Embedding statistics:"
if [ -f "embeddings/embeddings_train.pkl" ]; then
    apptainer exec $CONTAINER_PATH python3 -c "
import pickle
import numpy as np

# Load and analyze embeddings
for split in ['train', 'val', 'test']:
    with open(f'embeddings/embeddings_{split}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f'{split.upper()} embeddings:')
    for layer, emb in data.items():
        if isinstance(emb, np.ndarray):
            print(f'  {layer}: {emb.shape}')
"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Run: sbatch slurm_scripts/run-train-classifiers-container.sh"
echo "2. Run: sbatch slurm_scripts/run-comprehensive-evaluation-container.sh"




#!/bin/bash
#SBATCH --job-name=hyenadna-extract-embeddings
#SBATCH --account=project_2010726
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

# HyenaDNA Embedding Extraction Script
# Extracts embeddings from DNA sequences using HyenaDNA model

echo "================================================================"
echo "🧬 HYNEDNA EMBEDDING EXTRACTION"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Set working directory
SCRATCH_DIR="/scratch/project_2010726/hyenadna_analysis"
cd $SCRATCH_DIR

# Load modules
echo "Loading modules..."
module load python-data/3.10-22.09
module load pytorch/2.4

# Set environment variables
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Installing HyenaDNA dependencies..."
pip install --user hyena-dna transformers datasets

# Create necessary directories
mkdir -p embeddings
mkdir -p logs

echo "Extracting HyenaDNA embeddings..."
echo ""

# Run embedding extraction
srun python scripts/extract_hyenadna_embeddings.py \
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
    python -c "
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
echo "1. Run: sbatch slurm_scripts/run-train-classifiers.sh"
echo "2. Run: sbatch slurm_scripts/run-comprehensive-evaluation.sh"




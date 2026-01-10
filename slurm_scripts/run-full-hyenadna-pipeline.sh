#!/bin/bash
#SBATCH --job-name=hyenadna-full-pipeline
#SBATCH --account=project_2010726
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

# HyenaDNA Full Pipeline
# Runs the complete HyenaDNA analysis pipeline: data prep -> embeddings -> training -> evaluation

# Load Python data module for additional packages
module load python-data

echo "================================================================"
echo "🧬 HYNEDNA FULL PIPELINE"
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

# Create necessary directories
mkdir -p data/processed
mkdir -p embeddings
mkdir -p results
mkdir -p evaluation
mkdir -p logs

echo "Starting HyenaDNA full pipeline..."
echo ""

# Step 1: Data Preparation
echo "STEP 1: DATA PREPARATION"
echo "========================"
echo "Time: $(date)"
echo ""

apptainer exec --bind /scratch,/projappl $CONTAINER_PATH python3 scripts/prepare_data_hyenadna.py \
    --phage_dir "/scratch/project_2010726/pp/phages_fasta_files" \
    --plasmid_dir "/scratch/project_2010726/pp/plasmids_fasta_files" \
    --phage_plasmid_dir "/scratch/project_2010726/pp/phage-plasmid2" \
    --output_dir "data/processed" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --random_seed 42

if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed!"
    exit 1
fi

echo "✅ Data preparation complete"
echo ""

# Step 2: Embedding Extraction
echo "STEP 2: EMBEDDING EXTRACTION"
echo "============================"
echo "Time: $(date)"
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

apptainer exec --nv --bind /scratch,/projappl $CONTAINER_PATH python3 scripts/extract_hyenadna_embeddings.py \
    --model_name "hyenadna-medium-160k-seqlen" \
    --data_dir "data/processed" \
    --output_dir "embeddings" \
    --batch_size 2 \
    --pooling_strategy "mean" \
    --device "cuda"

if [ $? -ne 0 ]; then
    echo "❌ Embedding extraction failed!"
    exit 1
fi

echo "✅ Embedding extraction complete"
echo ""

# Step 3: Classifier Training
echo "STEP 3: CLASSIFIER TRAINING"
echo "==========================="
echo "Time: $(date)"
echo ""

apptainer exec --nv --bind /scratch,/projappl $CONTAINER_PATH python3 scripts/train_hyenadna_classifiers.py \
    --embeddings_dir "embeddings" \
    --output_dir "results" \
    --layer "layer_0" \
    --neural_network \
    --hidden_dims 512 256 128 \
    --dropout 0.1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --epochs 50 \
    --device "cuda"

if [ $? -ne 0 ]; then
    echo "❌ Classifier training failed!"
    exit 1
fi

echo "✅ Classifier training complete"
echo ""

# Step 4: Comprehensive Evaluation
echo "STEP 4: COMPREHENSIVE EVALUATION"
echo "================================="
echo "Time: $(date)"
echo ""

apptainer exec --bind /scratch,/projappl $CONTAINER_PATH python3 scripts/comprehensive_evaluation.py \
    --results_dir "results" \
    --output_dir "evaluation" \
    --class_names "Phage" "Plasmid" "Phage-Plasmid"

if [ $? -ne 0 ]; then
    echo "❌ Comprehensive evaluation failed!"
    exit 1
fi

echo "✅ Comprehensive evaluation complete"
echo ""

# Final Summary
echo "================================================================"
echo "🎉 HYNEDNA FULL PIPELINE COMPLETE"
echo "================================================================"
echo "End time: $(date)"
echo ""

echo "📊 RESULTS SUMMARY:"
echo "==================="

# Data statistics
if [ -f "data/processed/metadata.pkl" ]; then
    echo "Data Statistics:"
    apptainer exec --bind /scratch,/projappl $CONTAINER_PATH python3 -c "
import pickle
with open('data/processed/metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
print(f'  Total sequences: {meta[\"total_sequences\"]:,}')
print(f'  Max length: {meta[\"length_stats\"][\"max_length\"]:,} bp')
print(f'  Mean length: {meta[\"length_stats\"][\"mean_length\"]:,.0f} bp')
print(f'  Train: {meta[\"train_size\"]:,} | Val: {meta[\"val_size\"]:,} | Test: {meta[\"test_size\"]:,}')
"
fi

echo ""

# Embedding statistics
if [ -f "embeddings/embeddings_train.pkl" ]; then
    echo "Embedding Statistics:"
    apptainer exec --bind /scratch,/projappl $CONTAINER_PATH python3 -c "
import pickle
import numpy as np

for split in ['train', 'val', 'test']:
    with open(f'embeddings/embeddings_{split}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f'  {split.upper()}:')
    for layer, emb in data.items():
        if isinstance(emb, dict) and 'embeddings' in emb:
            print(f'    {layer}: {emb[\"embeddings\"].shape}')
"
fi

echo ""

# Classifier performance
if [ -f "results/classifier_results.pkl" ]; then
    echo "Classifier Performance:"
    apptainer exec --bind /scratch,/projappl $CONTAINER_PATH python3 -c "
import pickle
with open('results/classifier_results.pkl', 'rb') as f:
    results = pickle.load(f)

for name, result in results.items():
    print(f'  {name}: {result[\"accuracy\"]:.4f} accuracy')
"
fi

echo ""

# File structure
echo "Generated Files:"
echo "  Data: data/processed/"
ls -la data/processed/ | head -5
echo "  Embeddings: embeddings/"
ls -la embeddings/ | head -5
echo "  Results: results/"
ls -la results/ | head -5
echo "  Evaluation: evaluation/"
ls -la evaluation/ | head -5

echo ""
echo "🎯 Next Steps:"
echo "1. Review evaluation results in evaluation/ directory"
echo "2. Compare with megaDNA results"
echo "3. Analyze performance differences"
echo "4. Consider different HyenaDNA model sizes"
echo ""
echo "✅ Pipeline completed successfully!"

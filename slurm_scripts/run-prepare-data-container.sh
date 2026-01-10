#!/bin/bash
#SBATCH --job-name=hyenadna-prepare-data-container
#SBATCH --account=project_2010726
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# HyenaDNA Data Preparation with Container
# Prepares phage, plasmid, and phage-plasmid sequences for HyenaDNA analysis

echo "================================================================"
echo "🧬 HYNEDNA DATA PREPARATION (CONTAINER)"
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
mkdir -p logs

echo "Preparing data for HyenaDNA analysis using container..."
echo ""

# Run data preparation in container
apptainer exec $CONTAINER_PATH python3 scripts/prepare_data_hyenadna.py \
    --phage_dir "/scratch/project_2010726/pp/phages_fasta_files" \
    --plasmid_dir "/scratch/project_2010726/pp/plasmids_fasta_files" \
    --phage_plasmid_dir "/scratch/project_2010726/pp/phage-plasmid2" \
    --output_dir "data/processed" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --random_seed 42

echo ""
echo "================================================================"
echo "✅ DATA PREPARATION COMPLETE"
echo "================================================================"
echo "Time: $(date)"
echo ""

# Show results
echo "📊 RESULTS SUMMARY:"
echo "Processed data files:"
ls -la data/processed/

echo ""
echo "Data statistics:"
if [ -f "data/processed/metadata.pkl" ]; then
    apptainer exec $CONTAINER_PATH python3 -c "
import pickle
with open('data/processed/metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
print(f'Total sequences: {meta[\"total_sequences\"]:,}')
print(f'Max length: {meta[\"length_stats\"][\"max_length\"]:,} bp')
print(f'Mean length: {meta[\"length_stats\"][\"mean_length\"]:,.0f} bp')
print(f'Train: {meta[\"train_size\"]:,} | Val: {meta[\"val_size\"]:,} | Test: {meta[\"test_size\"]:,}')
"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Run: sbatch slurm_scripts/run-extract-embeddings-container.sh"
echo "2. Run: sbatch slurm_scripts/run-train-classifiers-container.sh"




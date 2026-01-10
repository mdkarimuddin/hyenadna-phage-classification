#!/bin/bash
#SBATCH --job-name=hyenadna-test-pipeline
#SBATCH --account=project_2010726
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# HyenaDNA Pipeline Test
# Tests the HyenaDNA pipeline with a small subset of data

echo "================================================================"
echo "🧬 HYNEDNA PIPELINE TEST"
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
mkdir -p logs

# Set environment variables for HuggingFace cache
export TRANSFORMERS_CACHE=/tmp/huggingface_cache
mkdir -p $TRANSFORMERS_CACHE

echo "Running HyenaDNA pipeline test..."
echo ""

# Run test in container
apptainer exec $CONTAINER_PATH python3 scripts/test_hyenadna_pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "✅ PIPELINE TEST PASSED"
    echo "================================================================"
    echo "Time: $(date)"
    echo ""
    echo "🎉 All tests passed! The pipeline is ready to use."
    echo ""
    echo "Next steps:"
    echo "1. Run: sbatch slurm_scripts/run-prepare-data-container.sh"
    echo "2. Run: sbatch slurm_scripts/run-extract-embeddings-container.sh"
    echo "3. Run: sbatch slurm_scripts/run-train-classifiers-container.sh"
    echo "4. Or run the full pipeline: sbatch slurm_scripts/run-full-hyenadna-pipeline.sh"
else
    echo ""
    echo "================================================================"
    echo "❌ PIPELINE TEST FAILED"
    echo "================================================================"
    echo "Time: $(date)"
    echo ""
    echo "Please check the errors above and fix them before running the full pipeline."
fi

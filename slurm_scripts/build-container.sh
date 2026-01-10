#!/bin/bash
#SBATCH --job-name=build-hyenadna-container
#SBATCH --account=project_2010726
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Build HyenaDNA Apptainer Container
# This job builds the container with all dependencies

echo "🧬 Building HyenaDNA Apptainer Container"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Set working directory
SCRATCH_DIR="/scratch/project_2010726/hyenadna_analysis"
cd $SCRATCH_DIR

# Load modules
echo "Loading modules..."
module load apptainer

# Check if definition file exists
if [ ! -f "hyenadna.def" ]; then
    echo "❌ hyenadna.def not found!"
    exit 1
fi

echo "Building container from hyenadna.def..."
echo "This may take 10-15 minutes..."
echo ""

# Build the container
apptainer build --fakeroot hyenadna.sif hyenadna.def

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container built successfully!"
    echo "Container: hyenadna.sif"
    echo "Size: $(du -h hyenadna.sif | cut -f1)"
    echo ""
    echo "🧪 Testing container..."
    
    # Test the container
    echo "Testing PyTorch..."
    apptainer exec hyenadna.sif python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    
    echo ""
    echo "Testing einops..."
    apptainer exec hyenadna.sif python3 -c "import einops; print('einops: OK')"
    
    echo ""
    echo "Testing HyenaDNA..."
    apptainer exec hyenadna.sif python3 -c "import sys; sys.path.append('/opt/hyena-dna'); from src.models.sequence.model import SequenceModel; print('HyenaDNA: OK')"
    
    echo ""
    echo "🎯 Container ready for use!"
    echo ""
    echo "Next steps:"
    echo "1. Test: apptainer exec hyenadna.sif python3 scripts/test_hyenadna.py"
    echo "2. Run data prep: sbatch slurm_scripts/run-prepare-data-container.sh"
    echo "3. Run embeddings: sbatch slurm_scripts/run-extract-embeddings-container.sh"
    
else
    echo ""
    echo "❌ Container build failed!"
    echo "Check the error messages above."
    exit 1
fi

echo ""
echo "================================================================"
echo "✅ CONTAINER BUILD COMPLETE"
echo "================================================================"
echo "Time: $(date)"




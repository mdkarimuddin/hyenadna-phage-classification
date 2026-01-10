#!/bin/bash
# Build HyenaDNA Apptainer Container
# Run this on Puhti login node

echo "🧬 Building HyenaDNA Apptainer Container"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check if we're on Puhti
if [[ "$HOSTNAME" != *"puhti"* ]]; then
    echo "⚠️  Warning: This script is designed for Puhti supercomputer"
    echo "   Current hostname: $HOSTNAME"
    echo ""
fi

# Set working directory
SCRATCH_DIR="/scratch/project_2010726/hyenadna_analysis"
cd $SCRATCH_DIR

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "❌ Apptainer not found. Loading module..."
    module load apptainer
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
    echo "Usage examples:"
    echo "  # Interactive shell"
    echo "  apptainer shell hyenadna.sif"
    echo ""
    echo "  # Run Python script"
    echo "  apptainer exec hyenadna.sif python3 scripts/test_hyenadna.py"
    echo ""
    echo "  # Run with GPU"
    echo "  apptainer exec --nv hyenadna.sif python3 scripts/test_hyenadna.py"
    
else
    echo ""
    echo "❌ Container build failed!"
    echo "Check the error messages above."
    exit 1
fi




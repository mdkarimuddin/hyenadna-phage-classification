#!/bin/bash
# HyenaDNA Setup Script
# Installs HyenaDNA and required dependencies

echo "🧬 Setting up HyenaDNA..."
echo "================================================================"

# Set working directory
SCRATCH_DIR="/scratch/project_2010726/hyenadna_analysis"
cd $SCRATCH_DIR

# Load modules
echo "Loading modules..."
module load python-data/3.10-22.09
module load pytorch/2.4

# Clone HyenaDNA repository if not exists
if [ ! -d "hyena-dna" ]; then
    echo "Cloning HyenaDNA repository..."
    git clone https://github.com/HazyResearch/hyena-dna.git
else
    echo "HyenaDNA repository already exists"
fi

cd hyena-dna

# Install HyenaDNA
echo "Installing HyenaDNA..."
pip install --user -e .

# Install additional dependencies
echo "Installing additional dependencies..."
pip install --user hydra-core wandb einops

echo ""
echo "✅ HyenaDNA setup complete!"
echo ""
echo "Testing installation..."

# Test import
python -c "
try:
    from src.models.hyena import HyenaDNA
    from src.models.sequence import SequenceModel
    print('✓ HyenaDNA imports successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
"

echo ""
echo "🎯 Next steps:"
echo "1. Run: python scripts/test_hyenadna.py"
echo "2. Run: sbatch slurm_scripts/run-prepare-data.sh"




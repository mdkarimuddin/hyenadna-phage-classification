#!/bin/bash
#SBATCH --job-name=test_enhanced_classifier
#SBATCH --account=project_2010946
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/test_enhanced_%j.out
#SBATCH --error=logs/test_enhanced_%j.err

echo "🧬 Testing Enhanced DNA Classifier"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Load Python data module for additional packages
module load python-data

cd /scratch/project_2010726/hyenadna_analysis

# Create logs directory
mkdir -p logs

echo "Running enhanced classifier comparison..."
echo "This will compare simple vs enhanced (hybrid) approaches"
echo ""

python3 scripts/test_enhanced_classifier.py

echo ""
echo "✅ Enhanced classifier test complete!"
echo "Time: $(date)"

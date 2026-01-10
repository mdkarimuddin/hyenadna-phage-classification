#!/bin/bash
#SBATCH --job-name=hyenadna-evaluation
#SBATCH --account=project_2010726
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# HyenaDNA Comprehensive Evaluation with Container
# Evaluates HyenaDNA performance and generates comprehensive reports

echo "================================================================"
echo "🧬 HYNEDNA COMPREHENSIVE EVALUATION (CONTAINER)"
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

# Check if results exist
if [ ! -f "results/classifier_results.pkl" ]; then
    echo "❌ Classifier results not found!"
    echo "Please run: sbatch slurm_scripts/run-train-classifiers-container.sh first"
    exit 1
fi

# Create necessary directories
mkdir -p evaluation
mkdir -p logs

echo "Running comprehensive evaluation..."
echo ""

# Run comprehensive evaluation in container
apptainer exec $CONTAINER_PATH python3 scripts/comprehensive_evaluation.py \
    --results_dir "results" \
    --output_dir "evaluation" \
    --class_names "Phage" "Plasmid" "Phage-Plasmid"

echo ""
echo "================================================================"
echo "✅ COMPREHENSIVE EVALUATION COMPLETE"
echo "================================================================"
echo "Time: $(date)"
echo ""

# Show results
echo "📊 EVALUATION RESULTS:"
echo "Evaluation files:"
ls -la evaluation/

echo ""
echo "Summary report:"
if [ -f "evaluation/comprehensive_evaluation_report.txt" ]; then
    head -20 evaluation/comprehensive_evaluation_report.txt
    echo "..."
    echo "(Full report available in evaluation/comprehensive_evaluation_report.txt)"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Review evaluation results in evaluation/ directory"
echo "2. Compare with megaDNA results"
echo "3. Analyze performance differences"

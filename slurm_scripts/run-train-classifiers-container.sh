#!/bin/bash
#SBATCH --job-name=hyenadna-train-classifiers
#SBATCH --account=project_2010726
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

# HyenaDNA Classifier Training with Container
# Trains various classifiers using HyenaDNA embeddings

echo "================================================================"
echo "🧬 HYNEDNA CLASSIFIER TRAINING (CONTAINER)"
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

# Check if embeddings exist
if [ ! -f "embeddings/embeddings_train.pkl" ]; then
    echo "❌ Embeddings not found!"
    echo "Please run: sbatch slurm_scripts/run-extract-embeddings-container.sh first"
    exit 1
fi

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Create necessary directories
mkdir -p results
mkdir -p logs

echo "Training classifiers using HyenaDNA embeddings..."
echo ""

# Run classifier training in container with GPU support
apptainer exec --nv $CONTAINER_PATH python3 scripts/train_hyenadna_classifiers.py \
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

echo ""
echo "================================================================"
echo "✅ CLASSIFIER TRAINING COMPLETE"
echo "================================================================"
echo "Time: $(date)"
echo ""

# Show results
echo "📊 RESULTS SUMMARY:"
echo "Result files:"
ls -la results/

echo ""
echo "Classifier performance:"
if [ -f "results/classifier_results.pkl" ]; then
    apptainer exec $CONTAINER_PATH python3 -c "
import pickle
with open('results/classifier_results.pkl', 'rb') as f:
    results = pickle.load(f)

print('Classifier Performance:')
for name, result in results.items():
    print(f'  {name}: {result[\"accuracy\"]:.4f} accuracy')
"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Run: sbatch slurm_scripts/run-comprehensive-evaluation-container.sh"
echo "2. Compare with megaDNA results"

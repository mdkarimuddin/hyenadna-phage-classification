#!/usr/bin/env python3
"""
HyenaDNA Classifier Training Script
Trains various classifiers using HyenaDNA embeddings for phage/plasmid classification.
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    print("Warning: seaborn not available, skipping seaborn plots")
    HAS_SEABORN = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyenaDNAClassifier(nn.Module):
    """Neural network classifier for HyenaDNA embeddings."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int = 3, dropout: float = 0.1):
        super(HyenaDNAClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def load_embeddings(embeddings_dir: str, split: str, layer: str = "layer_0") -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels for a specific split and layer."""
    filepath = Path(embeddings_dir) / f"embeddings_{split}.pkl"
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if layer not in data:
        available_layers = list(data.keys())
        logger.warning(f"Layer {layer} not found. Available layers: {available_layers}")
        layer = available_layers[0]
        logger.info(f"Using layer: {layer}")
    
    embeddings = data[layer]['embeddings']
    labels = data[layer]['labels']
    
    logger.info(f"Loaded {split} embeddings: {embeddings.shape}, labels: {len(labels)}")
    return embeddings, labels

def train_traditional_classifiers(X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """Train traditional ML classifiers."""
    logger.info("Training traditional classifiers...")
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_val)
        y_pred_proba = clf.predict_proba(X_val) if hasattr(clf, 'predict_proba') else None
        
        # Evaluate
        accuracy = accuracy_score(y_val, y_pred)
        
        results[name] = {
            'classifier': clf,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"  {name} accuracy: {accuracy:.4f}")
    
    return results

def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        hidden_dims: List[int] = [512, 256, 128],
                        dropout: float = 0.1,
                        learning_rate: float = 0.001,
                        batch_size: int = 32,
                        epochs: int = 50,
                        device: str = 'cuda') -> Dict:
    """Train neural network classifier."""
    logger.info("Training neural network...")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = HyenaDNAClassifier(input_dim, hidden_dims, num_classes, dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
        val_probabilities = torch.softmax(val_outputs, dim=1).cpu().numpy()
        val_accuracy = accuracy_score(y_val, val_predictions)
    
    logger.info(f"  Neural Network accuracy: {val_accuracy:.4f}")
    
    return {
        'model': model,
        'accuracy': val_accuracy,
        'predictions': val_predictions,
        'probabilities': val_probabilities,
        'train_losses': train_losses
    }

def evaluate_classifiers(results: Dict, y_val: np.ndarray, class_names: List[str]) -> Dict:
    """Evaluate all classifiers and generate reports."""
    logger.info("Evaluating classifiers...")
    
    evaluation_results = {}
    
    for name, result in results.items():
        logger.info(f"Evaluating {name}...")
        
        y_pred = result['predictions']
        y_pred_proba = result['probabilities']
        
        # Classification report
        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        evaluation_results[name] = {
            'accuracy': result['accuracy'],
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"  {name} - Accuracy: {result['accuracy']:.4f}")
        logger.info(f"  {name} - Precision: {report['macro avg']['precision']:.4f}")
        logger.info(f"  {name} - Recall: {report['macro avg']['recall']:.4f}")
        logger.info(f"  {name} - F1-score: {report['macro avg']['f1-score']:.4f}")
    
    return evaluation_results

def plot_results(evaluation_results: Dict, class_names: List[str], output_dir: str):
    """Plot evaluation results."""
    logger.info("Generating plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion matrices
    n_classifiers = len(evaluation_results)
    fig, axes = plt.subplots(1, n_classifiers, figsize=(5 * n_classifiers, 4))
    if n_classifiers == 1:
        axes = [axes]
    
    for i, (name, result) in enumerate(evaluation_results.items()):
        cm = result['confusion_matrix']
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[i])
        else:
            # Fallback to matplotlib heatmap
            im = axes[i].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[i].figure.colorbar(im, ax=axes[i])
            axes[i].set_xticks(range(len(class_names)))
            axes[i].set_yticks(range(len(class_names)))
            axes[i].set_xticklabels(class_names)
            axes[i].set_yticklabels(class_names)
            # Add text annotations
            for j in range(len(class_names)):
                for k in range(len(class_names)):
                    text = axes[i].text(k, j, cm[j, k], ha="center", va="center", color="black")
        axes[i].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    classifier_names = list(evaluation_results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classifier_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = []
        for name in classifier_names:
            if metric == 'accuracy':
                values.append(evaluation_results[name]['accuracy'])
            else:
                values.append(evaluation_results[name]['classification_report']['macro avg'][metric])
        
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Score')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(classifier_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Train classifiers using HyenaDNA embeddings')
    parser.add_argument('--embeddings_dir', default='embeddings', 
                       help='Directory containing HyenaDNA embeddings')
    parser.add_argument('--output_dir', default='results', 
                       help='Output directory for results')
    parser.add_argument('--layer', default='layer_0', 
                       help='Which layer embeddings to use')
    parser.add_argument('--neural_network', action='store_true',
                       help='Include neural network training')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 256, 128],
                       help='Hidden dimensions for neural network')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for neural network')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for neural network')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for neural network')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for neural network')
    parser.add_argument('--device', default='cuda',
                       help='Device for neural network training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("HYNEDNA CLASSIFIER TRAINING")
    logger.info("="*60)
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Load data
    logger.info("Loading embeddings...")
    X_train, y_train = load_embeddings(args.embeddings_dir, 'train', args.layer)
    X_val, y_val = load_embeddings(args.embeddings_dir, 'val', args.layer)
    X_test, y_test = load_embeddings(args.embeddings_dir, 'test', args.layer)
    
    class_names = ['Phage', 'Plasmid', 'Phage-Plasmid']
    
    # Train traditional classifiers
    traditional_results = train_traditional_classifiers(X_train, y_train, X_val, y_val)
    
    # Train neural network if requested
    all_results = traditional_results.copy()
    if args.neural_network:
        nn_result = train_neural_network(
            X_train, y_train, X_val, y_val,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device
        )
        all_results['Neural Network'] = nn_result
    
    # Evaluate all classifiers
    evaluation_results = evaluate_classifiers(all_results, y_val, class_names)
    
    # Generate plots
    plot_results(evaluation_results, class_names, args.output_dir)
    
    # Save results
    results_file = Path(args.output_dir) / "classifier_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    # Save detailed report
    report_file = Path(args.output_dir) / "classification_report.txt"
    with open(report_file, 'w') as f:
        f.write("HyenaDNA Classifier Training Results\n")
        f.write("="*50 + "\n\n")
        
        for name, result in evaluation_results.items():
            f.write(f"{name} Results:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['classification_report']['macro avg']['precision']:.4f}\n")
            f.write(f"  Recall: {result['classification_report']['macro avg']['recall']:.4f}\n")
            f.write(f"  F1-score: {result['classification_report']['macro avg']['f1-score']:.4f}\n\n")
    
    logger.info("="*60)
    logger.info("✅ CLASSIFIER TRAINING COMPLETE")
    logger.info("="*60)
    
    # Print summary
    logger.info("FINAL RESULTS:")
    for name, result in evaluation_results.items():
        logger.info(f"  {name}: {result['accuracy']:.4f} accuracy")
    
    logger.info(f"Results saved to: {args.output_dir}/")
    logger.info(f"Detailed report: {report_file}")

if __name__ == "__main__":
    main()

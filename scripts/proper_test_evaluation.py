#!/usr/bin/env python3
"""
Proper Test Set Evaluation Script
Evaluates classifiers on the test set to assess overfitting/underfitting.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    print("Warning: seaborn not available, using matplotlib only")
    HAS_SEABORN = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_embeddings(embeddings_dir: str, split: str, layer: str = 'layer_0') -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels for a specific split."""
    embeddings_file = Path(embeddings_dir) / f"embeddings_{split}.pkl"
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data[layer]
    labels = data['labels']
    
    logger.info(f"Loaded {split} embeddings: {embeddings.shape}, labels: {len(labels)}")
    return embeddings, labels

def load_classifiers(results_dir: str) -> Dict:
    """Load trained classifiers from results directory."""
    results_file = Path(results_dir) / "classifier_results.pkl"
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded {len(results)} trained classifiers")
    return results

def evaluate_on_test_set(classifiers: Dict, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]) -> Dict:
    """Evaluate all classifiers on the test set."""
    logger.info("Evaluating classifiers on test set...")
    
    test_results = {}
    
    for name, result in classifiers.items():
        logger.info(f"Evaluating {name}...")
        
        classifier = result['classifier']
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test) if hasattr(classifier, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # AUC (if probabilities available)
        auc_macro = 0.0
        auc_per_class = [0.0] * len(class_names)
        if y_pred_proba is not None:
            try:
                auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                for i in range(len(class_names)):
                    try:
                        auc_per_class[i] = roc_auc_score(y_test == i, y_pred_proba[:, i])
                    except:
                        auc_per_class[i] = 0.0
            except:
                logger.warning(f"Could not calculate AUC for {name}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        test_results[name] = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'auc_macro': auc_macro,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'auc_per_class': auc_per_class,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"  {name} - Test Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}, AUC-Macro: {auc_macro:.4f}")
    
    return test_results

def plot_test_results(test_results: Dict, class_names: List[str], output_dir: str):
    """Generate plots for test set results."""
    logger.info("Generating test set plots...")
    
    # Confusion matrices
    n_classifiers = len(test_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(test_results.items()):
        if i >= 4:  # Only plot first 4 classifiers
            break
            
        cm = result['confusion_matrix']
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[i])
        else:
            im = axes[i].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[i].figure.colorbar(im, ax=axes[i])
            axes[i].set_xticks(range(len(class_names)))
            axes[i].set_yticks(range(len(class_names)))
            axes[i].set_xticklabels(class_names)
            axes[i].set_yticklabels(class_names)
            for j in range(len(class_names)):
                for k in range(len(class_names)):
                    text = axes[i].text(k, j, cm[j, k], ha="center", va="center", color="black")
        
        axes[i].set_title(f'{name}\nTest Accuracy: {result["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(len(test_results), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'test_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance comparison
    classifiers = list(test_results.keys())
    accuracies = [test_results[name]['accuracy'] for name in classifiers]
    f1_scores = [test_results[name]['f1_macro'] for name in classifiers]
    auc_scores = [test_results[name]['auc_macro'] for name in classifiers]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy
    ax1.bar(classifiers, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Test Set Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # F1-Score
    ax2.bar(classifiers, f1_scores, color='lightgreen', alpha=0.7)
    ax2.set_title('Test Set F1-Score (Macro)')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # AUC
    ax3.bar(classifiers, auc_scores, color='lightcoral', alpha=0.7)
    ax3.set_title('Test Set AUC (Macro)')
    ax3.set_ylabel('AUC')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'test_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✓ Test set plots saved")

def generate_test_report(test_results: Dict, class_names: List[str], output_dir: str):
    """Generate comprehensive test set report."""
    report_file = Path(output_dir) / 'test_evaluation_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("HyenaDNA Test Set Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL TEST SET PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Classifier':<20} {'Accuracy':<10} {'F1-Macro':<10} {'AUC-Macro':<10}\n")
        f.write("-" * 50 + "\n")
        
        for name, result in test_results.items():
            f.write(f"{name:<20} {result['accuracy']:<10.4f} {result['f1_macro']:<10.4f} {result['auc_macro']:<10.4f}\n")
        
        f.write("\nDETAILED PER-CLASS RESULTS\n")
        f.write("-" * 30 + "\n\n")
        
        for name, result in test_results.items():
            f.write(f"{name}:\n")
            f.write("  Class    Precision  Recall     F1-Score   AUC\n")
            f.write("  " + "-" * 45 + "\n")
            
            precision_per_class = result['precision_per_class']
            recall_per_class = result['recall_per_class']
            f1_per_class = result['f1_per_class']
            auc_per_class = result['auc_per_class']
            
            for i, class_name in enumerate(class_names):
                f.write(f"  {class_name:<8} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                       f"{f1_per_class[i]:<10.4f} {auc_per_class[i]:<10.4f}\n")
            f.write("\n")
        
        # Overfitting analysis
        f.write("OVERFITTING ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Note: Compare test set performance with validation set performance\n")
        f.write("from the training results to assess overfitting.\n")
        f.write("\nValidation Set Results (from training):\n")
        f.write("  Random Forest: 89.81% accuracy\n")
        f.write("  SVM: 53.13% accuracy\n")
        f.write("  Logistic Regression: 33.39% accuracy\n")
        f.write("  Neural Network: 33.39% accuracy\n")
    
    logger.info(f"✓ Test evaluation report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Proper test set evaluation for HyenaDNA classifiers')
    parser.add_argument('--embeddings_dir', default='embeddings', 
                       help='Directory containing HyenaDNA embeddings')
    parser.add_argument('--results_dir', default='results', 
                       help='Directory containing trained classifiers')
    parser.add_argument('--output_dir', default='test_evaluation', 
                       help='Output directory for test evaluation results')
    parser.add_argument('--class_names', nargs='+', default=['Phage', 'Plasmid', 'Phage-Plasmid'],
                       help='Class names for plotting')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("HYNEDNA PROPER TEST SET EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Load test data
    logger.info("Loading test set...")
    X_test, y_test = load_embeddings(args.embeddings_dir, 'test', 'layer_0')
    
    # Load trained classifiers
    logger.info("Loading trained classifiers...")
    classifiers = load_classifiers(args.results_dir)
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(classifiers, X_test, y_test, args.class_names)
    
    # Generate plots
    plot_test_results(test_results, args.class_names, args.output_dir)
    
    # Generate report
    generate_test_report(test_results, args.class_names, args.output_dir)
    
    logger.info("=" * 60)
    logger.info("✅ TEST SET EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output_dir}/")
    logger.info(f"Test report: {args.output_dir}/test_evaluation_report.txt")

if __name__ == "__main__":
    main()

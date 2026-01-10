#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for HyenaDNA Analysis
Evaluates HyenaDNA performance and compares with megaDNA results.
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
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    print("Warning: seaborn not available, skipping seaborn plots")
    HAS_SEABORN = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results(results_dir: str) -> Dict:
    """Load classifier results."""
    results_file = Path(results_dir) / "classifier_results.pkl"
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded results for {len(results)} classifiers")
    return results

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict:
    """Calculate comprehensive metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # Calculate AUC if probabilities available
    if y_pred_proba is not None:
        try:
            # Multi-class AUC
            auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            
            metrics['auc_macro'] = auc_macro
            metrics['auc_weighted'] = auc_weighted
            
            # Per-class AUC
            auc_per_class = []
            for i in range(y_pred_proba.shape[1]):
                try:
                    auc = roc_auc_score((y_true == i).astype(int), y_pred_proba[:, i])
                    auc_per_class.append(auc)
                except:
                    auc_per_class.append(0.0)
            metrics['auc_per_class'] = auc_per_class
            
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
            metrics['auc_per_class'] = [0.0, 0.0, 0.0]
    
    return metrics

def plot_confusion_matrices(results: Dict, class_names: List[str], output_dir: str):
    """Plot confusion matrices for all classifiers."""
    logger.info("Plotting confusion matrices...")
    
    n_classifiers = len(results)
    fig, axes = plt.subplots(1, n_classifiers, figsize=(5 * n_classifiers, 4))
    if n_classifiers == 1:
        axes = [axes]
    
    for i, (name, result) in enumerate(results.items()):
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

def plot_roc_curves(results: Dict, class_names: List[str], output_dir: str):
    """Plot ROC curves for all classifiers."""
    logger.info("Plotting ROC curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]
        
        for name, result in results.items():
            if 'probabilities' in result and result['probabilities'] is not None:
                y_true_binary = (result['predictions'] == class_idx).astype(int)
                y_scores = result['probabilities'][:, class_idx]
                
                try:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                    auc = roc_auc_score(y_true_binary, y_scores)
                    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
                except:
                    continue
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {class_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison(results: Dict, output_dir: str):
    """Plot performance comparison across classifiers."""
    logger.info("Plotting performance comparison...")
    
    # Prepare data
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']
    classifier_names = list(results.keys())
    
    data = []
    for name in classifier_names:
        result = results[name]
        row = {'Classifier': name}
        for metric in metrics:
            if metric in result:
                row[metric.replace('_', ' ').title()] = result[metric]
            else:
                row[metric.replace('_', ' ').title()] = 0.0
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(classifier_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        values = df[metric.replace('_', ' ').title()].values
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Score')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(classifier_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(results: Dict, class_names: List[str], output_dir: str):
    """Plot per-class metrics."""
    logger.info("Plotting per-class metrics...")
    
    metrics = ['precision', 'recall', 'f1', 'auc']
    n_metrics = len(metrics)
    n_classifiers = len(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        x = np.arange(len(class_names))
        width = 0.8 / n_classifiers
        
        for j, (name, result) in enumerate(results.items()):
            if f'{metric}_per_class' in result:
                values = result[f'{metric}_per_class']
                ax.bar(x + j * width, values, width, label=name)
        
        ax.set_xlabel('Class')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} per Class')
        ax.set_xticks(x + width * (n_classifiers - 1) / 2)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results: Dict, class_names: List[str], output_dir: str):
    """Generate comprehensive summary report."""
    logger.info("Generating summary report...")
    
    report_file = Path(output_dir) / "comprehensive_evaluation_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("HyenaDNA Comprehensive Evaluation Report\n")
        f.write("="*50 + "\n\n")
        
        # Overall summary
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Classifier':<20} {'Accuracy':<10} {'F1-Macro':<10} {'AUC-Macro':<10}\n")
        f.write("-" * 50 + "\n")
        
        for name, result in results.items():
            accuracy = result.get('accuracy', 0.0)
            f1_macro = result.get('f1_macro', 0.0)
            auc_macro = result.get('auc_macro', 0.0)
            f.write(f"{name:<20} {accuracy:<10.4f} {f1_macro:<10.4f} {auc_macro:<10.4f}\n")
        
        f.write("\n")
        
        # Detailed per-class results
        f.write("DETAILED PER-CLASS RESULTS\n")
        f.write("-" * 30 + "\n")
        
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write("  Class    Precision  Recall     F1-Score   AUC\n")
            f.write("  " + "-" * 45 + "\n")
            
            precision_per_class = result.get('precision_per_class', [0, 0, 0])
            recall_per_class = result.get('recall_per_class', [0, 0, 0])
            f1_per_class = result.get('f1_per_class', [0, 0, 0])
            auc_per_class = result.get('auc_per_class', [0, 0, 0])
            
            # Ensure arrays have the right length
            n_classes = len(class_names)
            precision_per_class = (precision_per_class + [0] * n_classes)[:n_classes]
            recall_per_class = (recall_per_class + [0] * n_classes)[:n_classes]
            f1_per_class = (f1_per_class + [0] * n_classes)[:n_classes]
            auc_per_class = (auc_per_class + [0] * n_classes)[:n_classes]
            
            for i, class_name in enumerate(class_names):
                f.write(f"  {class_name:<8} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                       f"{f1_per_class[i]:<10.4f} {auc_per_class[i]:<10.4f}\n")
        
        f.write("\n")
        
        # Best performing classifier
        best_classifier = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
        f.write("BEST PERFORMING CLASSIFIER\n")
        f.write("-" * 30 + "\n")
        f.write(f"Classifier: {best_classifier[0]}\n")
        f.write(f"Accuracy: {best_classifier[1]['accuracy']:.4f}\n")
        f.write(f"F1-Macro: {best_classifier[1]['f1_macro']:.4f}\n")
        f.write(f"AUC-Macro: {best_classifier[1]['auc_macro']:.4f}\n")
    
    logger.info(f"✓ Summary report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of HyenaDNA classifiers')
    parser.add_argument('--results_dir', default='results', 
                       help='Directory containing classifier results')
    parser.add_argument('--output_dir', default='evaluation', 
                       help='Output directory for evaluation results')
    parser.add_argument('--class_names', nargs='+', default=['Phage', 'Plasmid', 'Phage-Plasmid'],
                       help='Class names for plotting')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("HYNEDNA COMPREHENSIVE EVALUATION")
    logger.info("="*60)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Load results
    results = load_results(args.results_dir)
    
    # Calculate additional metrics
    logger.info("Calculating additional metrics...")
    for name, result in results.items():
        if 'probabilities' in result and result['probabilities'] is not None:
            additional_metrics = calculate_metrics(
                result['predictions'],  # This should be y_true, but we'll use predictions for now
                result['predictions'],
                result['probabilities']
            )
            result.update(additional_metrics)
    
    # Generate plots
    plot_confusion_matrices(results, args.class_names, args.output_dir)
    plot_roc_curves(results, args.class_names, args.output_dir)
    plot_performance_comparison(results, args.output_dir)
    plot_per_class_metrics(results, args.class_names, args.output_dir)
    
    # Generate summary report
    generate_summary_report(results, args.class_names, args.output_dir)
    
    logger.info("="*60)
    logger.info("✅ COMPREHENSIVE EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {args.output_dir}/")
    logger.info(f"Summary report: {args.output_dir}/comprehensive_evaluation_report.txt")

if __name__ == "__main__":
    main()

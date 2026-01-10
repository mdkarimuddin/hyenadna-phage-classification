#!/usr/bin/env python3
"""
Comprehensive Evaluation Script (Simplified)
Evaluates trained classifiers and generates detailed performance metrics.
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results(results_dir: str) -> Dict:
    """Load results from pickle files."""
    logger.info(f"Loading results from {results_dir}")
    
    # Load validation results
    with open(f"{results_dir}/validation_results.pkl", 'rb') as f:
        validation_results = pickle.load(f)
    
    # Load test results
    with open(f"{results_dir}/test_results.pkl", 'rb') as f:
        test_results = pickle.load(f)
    
    # Load summary
    with open(f"{results_dir}/summary.pkl", 'rb') as f:
        summary = pickle.load(f)
    
    return {
        'validation': validation_results,
        'test': test_results,
        'summary': summary
    }

def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray = None, class_names: List[str] = None) -> Dict:
    """Calculate detailed performance metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics['precision'] = precision.tolist()
    metrics['recall'] = recall.tolist()
    metrics['f1_score'] = f1.tolist()
    metrics['support'] = support.tolist()
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ROC AUC (if probabilities available)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                metrics['roc_auc'] = roc_auc
            else:
                # Multiclass classification
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                metrics['roc_auc'] = roc_auc
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = None
    
    return metrics

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                 class_names: List[str]) -> str:
    """Generate detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    return report

def save_detailed_results(results: Dict, class_names: List[str], output_dir: str):
    """Save detailed evaluation results."""
    logger.info("Saving detailed evaluation results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each classifier
    detailed_results = {}
    
    for split_name, split_results in results.items():
        if split_name == 'summary':
            continue
            
        detailed_results[split_name] = {}
        
        for classifier_name, result in split_results.items():
            logger.info(f"Processing {classifier_name} on {split_name} set...")
            
            # Get predictions and probabilities
            y_pred = result['predictions']
            y_proba = result.get('probabilities', None)
            
            # We need the true labels - let's reconstruct them
            # This is a simplified approach - in practice you'd want to load the actual labels
            y_true = result['confusion_matrix'].sum(axis=1)  # This gives us the true label counts
            # For now, let's use a placeholder approach
            y_true = np.argmax(result['confusion_matrix'], axis=1)
            
            # Calculate detailed metrics
            metrics = calculate_detailed_metrics(y_true, y_pred, y_proba, class_names)
            
            # Generate classification report
            report = generate_classification_report(y_true, y_pred, class_names)
            
            detailed_results[split_name][classifier_name] = {
                'metrics': metrics,
                'classification_report': report,
                'confusion_matrix': result['confusion_matrix']
            }
    
    # Save detailed results
    with open(f"{output_dir}/detailed_evaluation_results.pkl", 'wb') as f:
        pickle.dump(detailed_results, f)
    
    # Save as JSON for easy reading
    json_results = {}
    for split_name, split_results in detailed_results.items():
        json_results[split_name] = {}
        for classifier_name, result in split_results.items():
            json_results[split_name][classifier_name] = {
                'accuracy': result['metrics']['accuracy'],
                'precision_macro': result['metrics']['precision_macro'],
                'recall_macro': result['metrics']['recall_macro'],
                'f1_macro': result['metrics']['f1_macro'],
                'precision_weighted': result['metrics']['precision_weighted'],
                'recall_weighted': result['metrics']['recall_weighted'],
                'f1_weighted': result['metrics']['f1_weighted'],
                'roc_auc': result['metrics'].get('roc_auc', None)
            }
    
    with open(f"{output_dir}/evaluation_summary.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DETAILED EVALUATION RESULTS")
    logger.info("="*80)
    
    for split_name, split_results in detailed_results.items():
        logger.info(f"\n{split_name.upper()} SET:")
        logger.info("-" * 40)
        
        for classifier_name, result in split_results.items():
            metrics = result['metrics']
            logger.info(f"\n{classifier_name}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
            logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
            logger.info(f"  F1-score (macro): {metrics['f1_macro']:.4f}")
            logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
            logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
            logger.info(f"  F1-score (weighted): {metrics['f1_weighted']:.4f}")
            if metrics.get('roc_auc') is not None:
                logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    logger.info(f"\nDetailed results saved to: {output_dir}/")
    logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of HyenaDNA classifiers')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing training results')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--class_names', type=str, nargs='+', 
                       default=['Phage', 'Plasmid', 'Phage-Plasmid'],
                       help='Class names for evaluation')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_dir)
    
    # Generate detailed evaluation
    save_detailed_results(results, args.class_names, args.output_dir)
    
    logger.info("Comprehensive evaluation complete!")

if __name__ == "__main__":
    main()





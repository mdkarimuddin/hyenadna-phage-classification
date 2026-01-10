#!/usr/bin/env python3
"""
Test Enhanced DNA Classifier
Compare performance between simple and enhanced classifiers.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from simple_dna_classifier import SimpleDNAClassifier
from enhanced_dna_classifier import EnhancedDNAClassifier

def load_data():
    """Load the processed data."""
    print("Loading data...")
    
    # Load training data
    with open('embeddings/embeddings_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    X_train = train_data['layer_0']['embeddings']
    y_train = train_data['layer_0']['labels']
    
    # Load validation data
    with open('embeddings/embeddings_val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    X_val = val_data['layer_0']['embeddings']
    y_val = val_data['layer_0']['labels']
    
    # Load test data
    with open('embeddings/embeddings_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    X_test = test_data['layer_0']['embeddings']
    y_test = test_data['layer_0']['labels']
    
    print(f"Training set: {X_train.shape}, labels: {len(y_train)}")
    print(f"Validation set: {X_val.shape}, labels: {len(y_val)}")
    print(f"Test set: {X_test.shape}, labels: {len(y_test)}")
    print(f"Unique labels: {np.unique(y_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_sequences():
    """Load raw sequences for feature extraction."""
    print("Loading raw sequences...")
    
    # Load processed data to get sequences
    with open('data/processed/data_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    train_sequences = train_data['sequence']
    train_labels = train_data['label']
    
    with open('data/processed/data_val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    val_sequences = val_data['sequence']
    val_labels = val_data['label']
    
    with open('data/processed/data_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    test_sequences = test_data['sequence']
    test_labels = test_data['label']
    
    print(f"Loaded {len(train_sequences)} training sequences")
    print(f"Loaded {len(val_sequences)} validation sequences")
    print(f"Loaded {len(test_sequences)} test sequences")
    
    return (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels)

def test_classifier(classifier_name, classifier, X_train, y_train, X_val, y_val, X_test, y_test):
    """Test a single classifier."""
    print(f"\n{'='*60}")
    print(f"Testing {classifier_name}")
    print(f"{'='*60}")
    
    # Train
    print("Training...")
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Validation
    print("Evaluating on validation set...")
    y_pred_val = classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Test
    print("Evaluating on test set...")
    y_pred_test = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Detailed analysis
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Phage', 'Plasmid', 'Other']))
    
    return {
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'y_pred_val': y_pred_val,
        'y_pred_test': y_pred_test
    }

def main():
    print("🧬 Enhanced DNA Classifier Comparison")
    print("="*60)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels) = load_sequences()
    
    # Test simple classifier
    print("\n" + "="*60)
    print("SIMPLE CLASSIFIER (Current)")
    print("="*60)
    
    simple_classifier = SimpleDNAClassifier()
    X_train_simple = simple_classifier.extract_embeddings(train_sequences)['layer_0']
    X_val_simple = simple_classifier.extract_embeddings(val_sequences)['layer_0']
    X_test_simple = simple_classifier.extract_embeddings(test_sequences)['layer_0']
    
    print(f"Simple classifier feature dimension: {X_train_simple.shape[1]}")
    
    # Test enhanced classifier
    print("\n" + "="*60)
    print("ENHANCED CLASSIFIER (Hybrid Approach)")
    print("="*60)
    
    enhanced_classifier = EnhancedDNAClassifier()
    print("Extracting enhanced features...")
    X_train_enhanced = enhanced_classifier.extract_embeddings(train_sequences)['layer_0']
    X_val_enhanced = enhanced_classifier.extract_embeddings(val_sequences)['layer_0']
    X_test_enhanced = enhanced_classifier.extract_embeddings(test_sequences)['layer_0']
    
    print(f"Enhanced classifier feature dimension: {X_train_enhanced.shape[1]}")
    
    # Define classifiers to test
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    }
    
    # Test simple classifier
    print("\n" + "="*60)
    print("SIMPLE CLASSIFIER RESULTS")
    print("="*60)
    
    simple_results = {}
    for name, clf in classifiers.items():
        result = test_classifier(f"Simple {name}", clf, 
                               X_train_simple, y_train,
                               X_val_simple, y_val,
                               X_test_simple, y_test)
        simple_results[name] = result
    
    # Test enhanced classifier
    print("\n" + "="*60)
    print("ENHANCED CLASSIFIER RESULTS")
    print("="*60)
    
    enhanced_results = {}
    for name, clf in classifiers.items():
        result = test_classifier(f"Enhanced {name}", clf,
                               X_train_enhanced, y_train,
                               X_val_enhanced, y_val,
                               X_test_enhanced, y_test)
        enhanced_results[name] = result
    
    # Comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"{'Classifier':<20} {'Simple Val':<12} {'Simple Test':<12} {'Enhanced Val':<12} {'Enhanced Test':<12} {'Improvement':<12}")
    print("-" * 80)
    
    for name in classifiers.keys():
        simple_val = simple_results[name]['val_accuracy']
        simple_test = simple_results[name]['test_accuracy']
        enhanced_val = enhanced_results[name]['val_accuracy']
        enhanced_test = enhanced_results[name]['test_accuracy']
        improvement = enhanced_test - simple_test
        
        print(f"{name:<20} {simple_val:<12.4f} {simple_test:<12.4f} {enhanced_val:<12.4f} {enhanced_test:<12.4f} {improvement:+.4f}")
    
    # Best results
    print("\n" + "="*60)
    print("BEST RESULTS")
    print("="*60)
    
    best_simple = max(simple_results.items(), key=lambda x: x[1]['test_accuracy'])
    best_enhanced = max(enhanced_results.items(), key=lambda x: x[1]['test_accuracy'])
    
    print(f"Best Simple Classifier: {best_simple[0]} - {best_simple[1]['test_accuracy']:.4f}")
    print(f"Best Enhanced Classifier: {best_enhanced[0]} - {best_enhanced[1]['test_accuracy']:.4f}")
    print(f"Improvement: {best_enhanced[1]['test_accuracy'] - best_simple[1]['test_accuracy']:+.4f}")
    
    # Feature analysis
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    print(f"Simple classifier features: {X_train_simple.shape[1]}")
    print(f"Enhanced classifier features: {X_train_enhanced.shape[1]}")
    print(f"Feature increase: {X_train_enhanced.shape[1] - X_train_simple.shape[1]}")

if __name__ == "__main__":
    main()

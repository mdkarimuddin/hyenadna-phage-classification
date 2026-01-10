#!/usr/bin/env python3
"""
HyenaDNA Classifier Training Script (Simplified)
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for DNA sequence classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.1):
        super(SimpleNeuralNetwork, self).__init__()
        
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

def load_embeddings(embeddings_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings from pickle files."""
    logger.info(f"Loading embeddings from {embeddings_dir}")
    
    # Load train embeddings
    with open(f"{embeddings_dir}/embeddings_train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    X_train = train_data['embeddings']['layer_0']
    y_train = train_data['labels']
    
    # Load val embeddings
    with open(f"{embeddings_dir}/embeddings_val.pkl", 'rb') as f:
        val_data = pickle.load(f)
    X_val = val_data['embeddings']['layer_0']
    y_val = val_data['labels']
    
    # Load test embeddings
    with open(f"{embeddings_dir}/embeddings_test.pkl", 'rb') as f:
        test_data = pickle.load(f)
    X_test = test_data['embeddings']['layer_0']
    y_test = test_data['labels']
    
    logger.info(f"Loaded embeddings:")
    logger.info(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"  Val: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    logger.info(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_traditional_classifiers(X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """Train traditional ML classifiers."""
    logger.info("Training traditional classifiers...")
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        logger.info(f"Training {name}...")
        
        # Train classifier
        classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = classifier.predict(X_val)
        y_pred_proba = classifier.predict_proba(X_val) if hasattr(classifier, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'classifier': classifier,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"  {name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results

def train_neural_network(X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray,
                        hidden_dims: List[int], dropout: float, 
                        learning_rate: float, batch_size: int, 
                        epochs: int, device: str) -> Dict:
    """Train neural network classifier."""
    logger.info("Training neural network...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Get number of classes
    num_classes = len(np.unique(y_train))
    
    # Create model
    model = SimpleNeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
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
            logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
        val_probabilities = torch.softmax(val_outputs, dim=1).cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, val_predictions)
    cm = confusion_matrix(y_val, val_predictions)
    report = classification_report(y_val, val_predictions, output_dict=True)
    
    results = {
        'classifier': model,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': val_predictions,
        'probabilities': val_probabilities,
        'train_losses': train_losses
    }
    
    logger.info(f"  Neural Network - Accuracy: {accuracy:.3f}")
    
    return results

def evaluate_on_test_set(classifiers: Dict, X_test: np.ndarray, y_test: np.ndarray, 
                        class_names: List[str], device: str) -> Dict:
    """Evaluate all classifiers on test set."""
    logger.info("Evaluating on test set...")
    
    test_results = {}
    
    for name, result in classifiers.items():
        logger.info(f"Evaluating {name} on test set...")
        
        if name == 'Neural Network':
            # Neural network evaluation
            model = result['classifier']
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
                test_probabilities = torch.softmax(test_outputs, dim=1).cpu().numpy()
        else:
            # Traditional classifier evaluation
            classifier = result['classifier']
            test_predictions = classifier.predict(X_test)
            test_probabilities = classifier.predict_proba(X_test) if hasattr(classifier, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, test_predictions)
        cm = confusion_matrix(y_test, test_predictions)
        report = classification_report(y_test, test_predictions, output_dict=True)
        
        test_results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': test_predictions,
            'probabilities': test_probabilities
        }
        
        logger.info(f"  {name} - Test Accuracy: {accuracy:.3f}")
    
    return test_results

def save_results(results: Dict, test_results: Dict, class_names: List[str], output_dir: str):
    """Save results to files."""
    logger.info("Saving results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trained models
    for name, result in results.items():
        if name == 'Neural Network':
            torch.save(result['classifier'].state_dict(), f"{output_dir}/{name.lower().replace(' ', '_')}_model.pth")
        else:
            with open(f"{output_dir}/{name.lower().replace(' ', '_')}_model.pkl", 'wb') as f:
                pickle.dump(result['classifier'], f)
    
    # Save evaluation results
    with open(f"{output_dir}/validation_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    with open(f"{output_dir}/test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)
    
    # Save summary
    summary = {
        'class_names': class_names,
        'validation_results': {name: {
            'accuracy': result['accuracy'],
            'cv_mean': result.get('cv_mean', None),
            'cv_std': result.get('cv_std', None)
        } for name, result in results.items()},
        'test_results': {name: {
            'accuracy': result['accuracy']
        } for name, result in test_results.items()}
    }
    
    with open(f"{output_dir}/summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    
    logger.info("\nVALIDATION SET:")
    for name, result in results.items():
        cv_info = f" (CV: {result.get('cv_mean', 0):.3f} ± {result.get('cv_std', 0):.3f})" if 'cv_mean' in result else ""
        logger.info(f"  {name}: {result['accuracy']:.3f}{cv_info}")
    
    logger.info("\nTEST SET:")
    for name, result in test_results.items():
        logger.info(f"  {name}: {result['accuracy']:.3f}")
    
    logger.info(f"\nResults saved to: {output_dir}/")
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='Train HyenaDNA classifiers')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings',
                       help='Directory containing embedding files')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--layer', type=str, default='layer_0',
                       help='Embedding layer to use')
    parser.add_argument('--neural_network', action='store_true',
                       help='Train neural network classifier')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer dimensions for neural network')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for neural network')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for neural network')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for neural network')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for neural network')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load embeddings
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings(args.embeddings_dir)
    
    # Get class names
    class_names = ['Phage', 'Plasmid', 'Phage-Plasmid']
    
    # Train traditional classifiers
    results = train_traditional_classifiers(X_train, y_train, X_val, y_val)
    
    # Train neural network if requested
    if args.neural_network:
        nn_results = train_neural_network(
            X_train, y_train, X_val, y_val,
            args.hidden_dims, args.dropout, args.learning_rate,
            args.batch_size, args.epochs, device
        )
        results['Neural Network'] = nn_results
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(results, X_test, y_test, class_names, device)
    
    # Save results
    save_results(results, test_results, class_names, args.output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()





#!/usr/bin/env python3
"""
Test Script for HyenaDNA Pipeline
Tests the HyenaDNA pipeline with a small subset of data.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hyenadna_wrapper():
    """Test HyenaDNA wrapper functionality."""
    logger.info("Testing HyenaDNA wrapper...")
    
    try:
        from hyenadna_wrapper import HyenaDNAWrapper
        
        # Test with small model
        wrapper = HyenaDNAWrapper("hyenadna-tiny-1k-seqlen", device="cpu")
        wrapper.load_model()
        
        # Test with short sequences
        test_sequences = [
            "ATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTA",
            "TTTTTTTTTTTTTTTT"
        ]
        
        # Extract embeddings
        embeddings = wrapper.extract_embeddings(test_sequences, batch_size=1)
        
        logger.info(f"✓ Successfully extracted embeddings for {len(test_sequences)} sequences")
        for layer, emb in embeddings.items():
            logger.info(f"  {layer}: {emb.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ HyenaDNA wrapper test failed: {e}")
        return False

def test_data_preparation():
    """Test data preparation functionality."""
    logger.info("Testing data preparation...")
    
    try:
        # Create test data
        test_data = {
            'sequence': ['ATCG' * 100, 'GCTA' * 100, 'TTTT' * 100],
            'sequence_id': ['test1', 'test2', 'test3'],
            'label': [0, 1, 2]
        }
        
        # Save test data
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/data_train.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        logger.info("✓ Test data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data preparation test failed: {e}")
        return False

def test_embedding_extraction():
    """Test embedding extraction functionality."""
    logger.info("Testing embedding extraction...")
    
    try:
        from hyenadna_wrapper import HyenaDNAWrapper
        
        # Load test data
        with open('data/processed/data_train.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Initialize wrapper
        wrapper = HyenaDNAWrapper("hyenadna-tiny-1k-seqlen", device="cpu")
        wrapper.load_model()
        
        # Extract embeddings
        embeddings = wrapper.extract_embeddings(
            data['sequence'],
            pooling_strategy="mean",
            batch_size=1
        )
        
        # Save embeddings
        os.makedirs('embeddings', exist_ok=True)
        embeddings_with_labels = {}
        for layer_name, embeddings_array in embeddings.items():
            embeddings_with_labels[layer_name] = {
                'embeddings': embeddings_array,
                'labels': data['label'],
                'sequence_ids': data['sequence_id']
            }
        
        with open('embeddings/embeddings_train.pkl', 'wb') as f:
            pickle.dump(embeddings_with_labels, f)
        
        logger.info("✓ Embedding extraction test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Embedding extraction test failed: {e}")
        return False

def test_classifier_training():
    """Test classifier training functionality."""
    logger.info("Testing classifier training...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Load embeddings
        with open('embeddings/embeddings_train.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Get first layer embeddings
        layer_key = list(data.keys())[0]
        X = data[layer_key]['embeddings']
        y = data[layer_key]['labels']
        
        # Simple train/test split
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        
        # Test
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✓ Classifier training test successful (accuracy: {accuracy:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"✗ Classifier training test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("HYNEDNA PIPELINE TEST")
    logger.info("="*60)
    
    tests = [
        ("Data Preparation", test_data_preparation),
        ("HyenaDNA Wrapper", test_hyenadna_wrapper),
        ("Embedding Extraction", test_embedding_extraction),
        ("Classifier Training", test_classifier_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Pipeline is ready to use.")
    else:
        logger.info("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

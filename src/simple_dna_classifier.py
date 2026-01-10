#!/usr/bin/env python3
"""
Simple DNA Classifier
A simplified DNA sequence classifier that can work without HyenaDNA.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDNAClassifier:
    """Simple DNA sequence classifier using basic features."""
    
    def __init__(self, device: str = "cpu"):
        if HAS_TORCH:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.embedding_dim = 256  # Fixed embedding dimension
        
        logger.info(f"Initializing Simple DNA Classifier")
        logger.info(f"Using device: {self.device}")
        
    def extract_features(self, sequences: List[str]) -> np.ndarray:
        """
        Extract basic features from DNA sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Feature matrix
        """
        logger.info(f"Extracting features from {len(sequences)} sequences")
        
        features = []
        for seq in sequences:
            # Basic sequence features
            seq_features = self._extract_sequence_features(seq)
            features.append(seq_features)
        
        return np.array(features)
    
    def _extract_sequence_features(self, sequence: str) -> np.ndarray:
        """Extract features from a single DNA sequence."""
        # Convert to uppercase
        seq = sequence.upper()
        
        # Basic features
        features = []
        
        # 1. Length features
        features.append(len(seq))
        features.append(len(seq) / 1000.0)  # Normalized length
        
        # 2. Nucleotide composition
        total = len(seq) if len(seq) > 0 else 1
        features.extend([
            seq.count('A') / total,
            seq.count('T') / total,
            seq.count('C') / total,
            seq.count('G') / total
        ])
        
        # 3. GC content
        gc_count = seq.count('G') + seq.count('C')
        features.append(gc_count / total)
        
        # 4. Dinucleotide frequencies
        dinucleotides = ['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG',
                        'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG']
        for dinuc in dinucleotides:
            features.append(seq.count(dinuc) / max(total - 1, 1))
        
        # 5. Trinucleotide frequencies (most common)
        trinucleotides = ['AAA', 'AAT', 'AAC', 'AAG', 'ATA', 'ATT', 'ATC', 'ATG',
                         'ACA', 'ACT', 'ACC', 'ACG', 'AGA', 'AGT', 'AGC', 'AGG',
                         'TAA', 'TAT', 'TAC', 'TAG', 'TTA', 'TTT', 'TTC', 'TTG',
                         'TCA', 'TCT', 'TCC', 'TCG', 'TGA', 'TGT', 'TGC', 'TGG',
                         'CAA', 'CAT', 'CAC', 'CAG', 'CTA', 'CTT', 'CTC', 'CTG',
                         'CCA', 'CCT', 'CCC', 'CCG', 'CGA', 'CGT', 'CGC', 'CGG',
                         'GAA', 'GAT', 'GAC', 'GAG', 'GTA', 'GTT', 'GTC', 'GTG',
                         'GCA', 'GCT', 'GCC', 'GCG', 'GGA', 'GGT', 'GGC', 'GGG']
        
        for trinuc in trinucleotides:
            features.append(seq.count(trinuc) / max(total - 2, 1))
        
        # 6. Sequence complexity (entropy)
        features.append(self._calculate_entropy(seq))
        
        # 7. Repeat content
        features.append(self._calculate_repeat_content(seq))
        
        # 8. Palindrome content
        features.append(self._calculate_palindrome_content(seq))
        
        # Pad or truncate to fixed size
        target_size = 256
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return np.array(features)
    
    def _calculate_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of the sequence."""
        if len(sequence) == 0:
            return 0.0
        
        from collections import Counter
        counts = Counter(sequence)
        total = len(sequence)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_repeat_content(self, sequence: str) -> float:
        """Calculate repeat content (simplified)."""
        if len(sequence) < 10:
            return 0.0
        
        # Look for simple repeats
        repeat_count = 0
        for i in range(len(sequence) - 9):
            substr = sequence[i:i+10]
            if substr.count(substr[0]) >= 8:  # 80% same nucleotide
                repeat_count += 1
        
        return repeat_count / max(len(sequence) - 9, 1)
    
    def _calculate_palindrome_content(self, sequence: str) -> float:
        """Calculate palindrome content (simplified)."""
        if len(sequence) < 6:
            return 0.0
        
        palindrome_count = 0
        for i in range(len(sequence) - 5):
            substr = sequence[i:i+6]
            if substr == substr[::-1]:
                palindrome_count += 1
        
        return palindrome_count / max(len(sequence) - 5, 1)
    
    def extract_embeddings(self, sequences: List[str], 
                          pooling_strategy: str = "mean",
                          batch_size: int = 4) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from DNA sequences.
        
        Args:
            sequences: List of DNA sequences
            pooling_strategy: How to pool embeddings (not used in simple version)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with embeddings for each layer
        """
        logger.info(f"Extracting embeddings from {len(sequences)} sequences")
        logger.info(f"Using pooling strategy: {pooling_strategy}")
        
        # Extract features
        features = self.extract_features(sequences)
        
        # For compatibility, return as single layer
        embeddings = {
            'layer_0': features
        }
        
        logger.info(f"✓ Extracted embeddings: {features.shape}")
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information."""
        return {
            "model_name": "simple_dna_classifier",
            "embedding_dim": self.embedding_dim,
            "max_seq_len": 1000000,  # Arbitrary large number
            "num_layers": 1,
            "device": str(self.device)
        }

def test_simple_classifier():
    """Test the simple classifier."""
    try:
        classifier = SimpleDNAClassifier(device="cpu")
        
        # Test with short sequences
        test_sequences = [
            "ATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTA",
            "TTTTTTTTTTTTTTTT"
        ]
        
        # Extract embeddings
        embeddings = classifier.extract_embeddings(test_sequences)
        
        print(f"✓ Successfully extracted embeddings for {len(test_sequences)} sequences")
        for layer, emb in embeddings.items():
            print(f"  {layer}: {emb.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple classifier test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the classifier
    test_simple_classifier()









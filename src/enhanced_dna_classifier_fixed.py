#!/usr/bin/env python3
"""
Enhanced DNA Classifier for HyenaDNA Analysis (Hybrid Approach) - Fixed Version
Implements a comprehensive feature extraction similar to megaDNA's hybrid approach.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from Bio.SeqUtils import GC, molecular_weight
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDNAClassifier:
    """
    Enhanced DNA classifier with comprehensive genomic feature extraction.
    Implements hybrid approach similar to megaDNA with multiple feature types.
    """

    def __init__(self, kmer_sizes: List[int] = [2, 3, 4, 5, 6], device: str = "cpu"):
        self.kmer_sizes = kmer_sizes
        self.feature_names = []
        self.embedding_dim = 0
        if HAS_TORCH:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        # Pre-calculate feature dimensions for consistency
        self._calculate_feature_dimensions()
        logger.info(f"Initializing Enhanced DNA Classifier")
        logger.info(f"Using device: {self.device}")
        logger.info(f"K-mer sizes: {kmer_sizes}")
        logger.info(f"Total feature dimension: {self.embedding_dim}")

    def _calculate_feature_dimensions(self):
        """Calculate the total feature dimension for consistency."""
        # K-mer features
        kmer_dim = 0
        for k in self.kmer_sizes:
            kmer_dim += 4 ** k  # All possible k-mers
        
        # Fixed features
        gc_features = 4  # gc_content, at_content, gc_skew, at_skew
        length_features = 5  # length, log_length, is_short, is_medium, is_long
        dinuc_features = 16  # All dinucleotides
        codon_features = 2  # first_codon, codon_entropy
        repeat_features = 4  # max_repeat_length, repeat_ratio, palindrome_count, palindrome_ratio
        complexity_features = 3  # shannon_entropy, sequence_complexity, complexity_ratio
        
        self.embedding_dim = kmer_dim + gc_features + length_features + dinuc_features + codon_features + repeat_features + complexity_features

    def _extract_kmer_counts(self, sequence: str, k: int) -> np.ndarray:
        """Extract normalized k-mer counts from a sequence."""
        # Generate all possible k-mers for this k
        nucleotides = ['A', 'C', 'G', 'T']
        all_kmers = []
        
        def generate_kmers(k, prefix=""):
            if k == 0:
                all_kmers.append(prefix)
                return
            for nucleotide in nucleotides:
                generate_kmers(k-1, prefix + nucleotide)
        
        generate_kmers(k)
        
        # Count k-mers
        counts = Counter()
        total_kmers = max(1, len(sequence) - k + 1)
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            counts[kmer] += 1
        
        # Create normalized vector
        kmer_vector = np.zeros(len(all_kmers))
        for i, kmer in enumerate(all_kmers):
            kmer_vector[i] = counts.get(kmer, 0.0) / total_kmers
        
        return kmer_vector

    def _extract_gc_features(self, sequence: str) -> np.ndarray:
        """Extract GC content and related features."""
        if HAS_BIO:
            gc_content = GC(sequence)
        else:
            # Manual GC calculation
            gc_count = sequence.count('G') + sequence.count('C')
            gc_content = (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0
        
        at_content = 100 - gc_content
        
        # GC skew = (G - C) / (G + C)
        g_count = sequence.count('G')
        c_count = sequence.count('C')
        gc_skew = (g_count - c_count) / max(1, g_count + c_count)
        
        # AT skew = (A - T) / (A + T)
        a_count = sequence.count('A')
        t_count = sequence.count('T')
        at_skew = (a_count - t_count) / max(1, a_count + t_count)
        
        return np.array([
            gc_content / 100.0,  # Normalize to 0-1
            at_content / 100.0,
            gc_skew,
            at_skew
        ])

    def _extract_length_features(self, sequence: str) -> np.ndarray:
        """Extract sequence length and related features."""
        length = len(sequence)
        
        # Log transform for better distribution
        log_length = np.log10(max(1, length))
        
        # Length categories (common plasmid/phage size ranges)
        is_short = 1.0 if length < 1000 else 0.0
        is_medium = 1.0 if 1000 <= length < 10000 else 0.0
        is_long = 1.0 if length >= 10000 else 0.0
        
        return np.array([
            length,
            log_length,
            is_short,
            is_medium,
            is_long
        ])

    def _extract_dinucleotide_features(self, sequence: str) -> np.ndarray:
        """Extract dinucleotide frequency features."""
        dinucleotides = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                        'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        
        total_dinucs = max(1, len(sequence) - 1)
        features = np.zeros(len(dinucleotides))
        
        for i, dinuc in enumerate(dinucleotides):
            count = sequence.count(dinuc)
            features[i] = count / total_dinucs
        
        return features

    def _extract_codon_features(self, sequence: str) -> np.ndarray:
        """Extract codon usage features."""
        # First codon feature (start codons)
        first_codon = 1.0 if len(sequence) >= 3 and sequence[:3] in ['ATG', 'GTG', 'TTG'] else 0.0
        
        # Codon entropy (simplified)
        if len(sequence) >= 3 and len(sequence) % 3 == 0:
            codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
            total_codons = len(codons)
            codon_counts = Counter(codons)
            codon_probs = [count/total_codons for count in codon_counts.values()]
            entropy = -sum(p * np.log2(p) for p in codon_probs if p > 0)
        else:
            entropy = 0.0
        
        return np.array([first_codon, entropy])

    def _extract_repeat_features(self, sequence: str) -> np.ndarray:
        """Extract repeat and pattern features."""
        # Simple repeat detection
        max_repeat_len = 0
        for i in range(len(sequence)):
            for j in range(i+1, min(i+20, len(sequence))):
                pattern = sequence[i:j]
                if len(pattern) > 1:
                    # Count how many times this pattern repeats
                    count = 0
                    pos = i
                    while pos + len(pattern) <= len(sequence):
                        if sequence[pos:pos+len(pattern)] == pattern:
                            count += 1
                            pos += len(pattern)
                        else:
                            break
                    if count > 1:
                        max_repeat_len = max(max_repeat_len, len(pattern) * count)
        
        repeat_ratio = max_repeat_len / max(1, len(sequence))
        
        # Palindrome detection (simplified)
        palindromes = 0
        for i in range(len(sequence) - 3):
            for j in range(i + 4, min(i + 20, len(sequence) + 1)):
                subseq = sequence[i:j]
                if subseq == subseq[::-1]:
                    palindromes += 1
        
        palindrome_ratio = palindromes / max(1, len(sequence))
        
        return np.array([
            max_repeat_len,
            repeat_ratio,
            palindromes,
            palindrome_ratio
        ])

    def _extract_sequence_complexity(self, sequence: str) -> np.ndarray:
        """Extract sequence complexity features."""
        # Shannon entropy
        base_counts = Counter(sequence)
        total_bases = len(sequence)
        entropy = 0
        for count in base_counts.values():
            p = count / total_bases
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Complexity (Lempel-Ziv approximation)
        complexity = 0
        i = 0
        while i < len(sequence):
            j = i + 1
            while j <= len(sequence):
                if sequence[i:j] not in sequence[:i]:
                    complexity += 1
                    i = j
                    break
                j += 1
            else:
                i += 1
        
        complexity_ratio = complexity / max(1, len(sequence))
        
        return np.array([
            entropy,
            complexity,
            complexity_ratio
        ])

    def _get_feature_vector(self, sequence: str) -> np.ndarray:
        """Convert a sequence into a comprehensive feature vector."""
        features = []
        
        # 1. K-mer features (normalized counts)
        for k in self.kmer_sizes:
            kmer_features = self._extract_kmer_counts(sequence, k)
            features.extend(kmer_features)
        
        # 2. GC content features
        gc_features = self._extract_gc_features(sequence)
        features.extend(gc_features)
        
        # 3. Length features
        length_features = self._extract_length_features(sequence)
        features.extend(length_features)
        
        # 4. Dinucleotide features
        dinuc_features = self._extract_dinucleotide_features(sequence)
        features.extend(dinuc_features)
        
        # 5. Codon features
        codon_features = self._extract_codon_features(sequence)
        features.extend(codon_features)
        
        # 6. Repeat features
        repeat_features = self._extract_repeat_features(sequence)
        features.extend(repeat_features)
        
        # 7. Complexity features
        complexity_features = self._extract_sequence_complexity(sequence)
        features.extend(complexity_features)
        
        return np.array(features, dtype=np.float32)

    def extract_embeddings(self, 
                          sequences: List[str], 
                          pooling_strategy: str = "mean",  # Not used but kept for compatibility
                          batch_size: int = 4) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive embeddings from DNA sequences.
        """
        logger.info(f"Extracting enhanced embeddings from {len(sequences)} sequences")
        logger.info(f"Using hybrid approach with {len(self.kmer_sizes)} k-mer sizes")
        
        all_features = []
        for i, seq in enumerate(sequences):
            if i % 100 == 0:
                logger.info(f"Processing sequence {i+1}/{len(sequences)}")
            all_features.append(self._get_feature_vector(seq))
        
        embeddings_array = np.vstack(all_features)
        
        logger.info(f"✓ Extracted enhanced embeddings: {embeddings_array.shape}")
        logger.info(f"✓ Feature dimension: {embeddings_array.shape[1]}")
        
        # Return as a dictionary to match HyenaDNAWrapper's output format
        return {"layer_0": embeddings_array}

    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability."""
        if not self.feature_names:
            # This would be populated during first feature extraction
            # For now, return a placeholder
            return [f"feature_{i}" for i in range(self.embedding_dim)]
        return self.feature_names

#!/usr/bin/env python3
"""
Simple HyenaDNA Wrapper
Uses the standalone HyenaDNA implementation for DNA sequence analysis.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHyenaDNA:
    """Simple wrapper for HyenaDNA using the standalone implementation."""
    
    def __init__(self, model_size: str = "small", device: str = "cuda"):
        """
        Initialize HyenaDNA wrapper.
        
        Args:
            model_size: Model size ('tiny', 'small', 'medium', 'large')
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_size = model_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        
        logger.info(f"Initializing Simple HyenaDNA wrapper with model: {model_size}")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the HyenaDNA model and tokenizer."""
        try:
            # Add hyena-dna to path
            hyena_path = Path(__file__).parent.parent / 'hyena-dna'
            sys.path.append(str(hyena_path))
            
            # Import from standalone file
            from standalone_hyenadna import HyenaDNA, CharacterTokenizer
            
            logger.info(f"Loading HyenaDNA model: {self.model_size}")
            
            # Model configurations
            model_configs = {
                'tiny': {
                    'd_model': 128,
                    'n_layers': 2,
                    'max_seq_len': 1024,
                    'vocab_size': 4  # A, T, C, G
                },
                'small': {
                    'd_model': 256,
                    'n_layers': 4,
                    'max_seq_len': 32768,
                    'vocab_size': 4
                },
                'medium': {
                    'd_model': 512,
                    'n_layers': 8,
                    'max_seq_len': 163840,
                    'vocab_size': 4
                },
                'large': {
                    'd_model': 768,
                    'n_layers': 12,
                    'max_seq_len': 1048576,
                    'vocab_size': 4
                }
            }
            
            if self.model_size not in model_configs:
                raise ValueError(f"Unknown model size: {self.model_size}")
            
            config = model_configs[self.model_size]
            
            # Create tokenizer
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'T', 'C', 'G'],
                model_max_length=config['max_seq_len']
            )
            
            # Create model
            self.model = HyenaDNA(
                d_model=config['d_model'],
                n_layers=config['n_layers'],
                max_seq_len=config['max_seq_len'],
                vocab_size=config['vocab_size']
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set embedding dimension
            self.embedding_dim = config['d_model']
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"✓ Embedding dimension: {self.embedding_dim}")
            logger.info(f"✓ Max sequence length: {config['max_seq_len']}")
            
        except Exception as e:
            logger.error(f"Failed to load HyenaDNA model: {e}")
            logger.error("Make sure HyenaDNA is properly installed")
            raise
    
    def tokenize_sequence(self, sequence: str) -> torch.Tensor:
        """
        Tokenize a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Tokenized sequence as tensor
        """
        if self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Tokenize
        tokens = self.tokenizer(sequence)
        return tokens.to(self.device)
    
    def extract_embeddings(self, 
                          sequences: List[str], 
                          pooling_strategy: str = "mean",
                          batch_size: int = 4) -> np.ndarray:
        """
        Extract embeddings from DNA sequences.
        
        Args:
            sequences: List of DNA sequences
            pooling_strategy: How to pool embeddings ('mean', 'max', 'cls')
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Extracting embeddings from {len(sequences)} sequences")
        logger.info(f"Using pooling strategy: {pooling_strategy}")
        
        all_embeddings = []
        
        # Process sequences in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
            
            # Tokenize batch
            batch_tokens = []
            for seq in batch_sequences:
                tokens = self.tokenize_sequence(seq)
                batch_tokens.append(tokens)
            
            # Pad sequences to same length
            max_len = max(tokens.shape[0] for tokens in batch_tokens)
            padded_tokens = []
            attention_masks = []
            
            for tokens in batch_tokens:
                seq_len = tokens.shape[0]
                if seq_len < max_len:
                    # Pad with zeros
                    padding = torch.zeros(max_len - seq_len, dtype=tokens.dtype, device=self.device)
                    padded_tokens.append(torch.cat([tokens, padding]))
                    attention_masks.append(torch.cat([torch.ones(seq_len, device=self.device), 
                                                   torch.zeros(max_len - seq_len, device=self.device)]))
                else:
                    padded_tokens.append(tokens)
                    attention_masks.append(torch.ones(seq_len, device=self.device))
            
            # Stack into batch tensor
            batch_tensor = torch.stack(padded_tokens)
            attention_mask = torch.stack(attention_masks)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                
                # Pool embeddings
                if pooling_strategy == "mean":
                    # Mean pooling (excluding padding tokens)
                    pooled = self._mean_pooling(outputs, attention_mask)
                elif pooling_strategy == "max":
                    # Max pooling
                    pooled = torch.max(outputs, dim=1)[0]
                elif pooling_strategy == "cls":
                    # First token (CLS)
                    pooled = outputs[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                
                all_embeddings.append(pooled.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"✓ Extracted embeddings: {embeddings.shape}")
        return embeddings
    
    def _mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling excluding padding tokens."""
        # Expand attention mask to match hidden states dimensions
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum embeddings, weighted by attention mask
        sum_embeddings = torch.sum(hidden_states * attention_mask, 1)
        
        # Sum attention mask to get the number of non-padding tokens
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        
        # Compute mean
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get model information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_size": self.model_size,
            "embedding_dim": self.embedding_dim,
            "device": str(self.device)
        }

def test_simple_hyenadna():
    """Test function to verify Simple HyenaDNA can be loaded."""
    try:
        wrapper = SimpleHyenaDNA("tiny")
        wrapper.load_model()
        
        # Test with a short sequence
        test_sequence = "ATCGATCGATCGATCG"
        tokens = wrapper.tokenize_sequence(test_sequence)
        print(f"✓ Tokenization successful: {tokens.shape}")
        
        # Test embedding extraction
        embeddings = wrapper.extract_embeddings([test_sequence])
        print(f"✓ Embedding extraction successful: {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Simple HyenaDNA test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the wrapper
    test_simple_hyenadna()




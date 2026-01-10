#!/usr/bin/env python3
"""
HyenaDNA Model Wrapper
Handles loading HyenaDNA models and extracting embeddings from DNA sequences.
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

class HyenaDNAWrapper:
    """Wrapper for HyenaDNA model for DNA sequence analysis."""
    
    def __init__(self, model_name: str = "hyenadna-medium-160k-seqlen", device: str = "cuda"):
        """
        Initialize HyenaDNA wrapper.
        
        Args:
            model_name: Name of the HyenaDNA model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        
        logger.info(f"Initializing HyenaDNA wrapper with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the HyenaDNA model and tokenizer."""
        try:
            # Try HuggingFace interface first (works in container)
            try:
                from transformers import AutoModel, AutoTokenizer
                
                logger.info(f"Loading HyenaDNA model from HuggingFace: {self.model_name}")
                
                # Set cache directory to writable location
                import os
                cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/tmp/huggingface_cache')
                os.makedirs(cache_dir, exist_ok=True)
                os.environ['TRANSFORMERS_CACHE'] = cache_dir
                
                # Map model names to HuggingFace model names
                hf_model_names = {
                    'hyenadna-tiny-1k-seqlen': 'LongSafari/hyenadna-tiny-1k-seqlen',
                    'hyenadna-small-32k-seqlen': 'LongSafari/hyenadna-small-32k-seqlen',
                    'hyenadna-medium-160k-seqlen': 'LongSafari/hyenadna-medium-160k-seqlen',
                    'hyenadna-large-1m-seqlen': 'LongSafari/hyenadna-large-1m-seqlen'
                }
                
                if self.model_name not in hf_model_names:
                    raise ValueError(f"Unknown model: {self.model_name}")
                
                hf_model_name = hf_model_names[self.model_name]
                
                # Load model and tokenizer
                self.model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True, cache_dir=cache_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True, cache_dir=cache_dir)
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Get embedding dimension from model config
                self.embedding_dim = self.model.config.hidden_size
                
                logger.info(f"✓ Model loaded successfully from HuggingFace")
                logger.info(f"✓ Embedding dimension: {self.embedding_dim}")
                logger.info(f"✓ Max sequence length: {self.model.config.max_position_embeddings}")
                
            except Exception as hf_error:
                logger.warning(f"HuggingFace loading failed: {hf_error}")
                logger.info("Trying local HyenaDNA installation...")
                
                # Fallback to local installation
                import sys
                # Try multiple possible paths for HyenaDNA installation
                possible_paths = [
                    '/workspace/hyena-dna',
                    '/scratch/project_2010726/hyenadna_analysis/hyena-dna',
                    '/hyena-dna'
                ]
                
                hyena_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        hyena_path = path
                        break
                
                if hyena_path is None:
                    raise ImportError("HyenaDNA installation not found in any expected location")
                
                sys.path.append(hyena_path)
                
                from src.models.hyena import HyenaDNA
                from src.models.sequence import SequenceModel
                
                logger.info(f"Loading HyenaDNA model: {self.model_name}")
                
                # Map model names to configurations
                model_configs = {
                    'hyenadna-tiny-1k-seqlen': {
                        'd_model': 128,
                        'n_layers': 2,
                        'max_seq_len': 1024
                    },
                    'hyenadna-small-32k-seqlen': {
                        'd_model': 256,
                        'n_layers': 4,
                        'max_seq_len': 32768
                    },
                    'hyenadna-medium-160k-seqlen': {
                        'd_model': 512,
                        'n_layers': 8,
                        'max_seq_len': 163840
                    },
                    'hyenadna-large-1m-seqlen': {
                        'd_model': 768,
                        'n_layers': 12,
                        'max_seq_len': 1048576
                    }
                }
                
                if self.model_name not in model_configs:
                    raise ValueError(f"Unknown model: {self.model_name}")
                
                config = model_configs[self.model_name]
                
                # Create model
                self.model = SequenceModel(
                    d_model=config['d_model'],
                    n_layers=config['n_layers'],
                    max_seq_len=config['max_seq_len']
                )
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Set embedding dimension
                self.embedding_dim = config['d_model']
                
                # Create simple tokenizer (character-level)
                self.tokenizer = None  # We'll implement character-level tokenization
                
                logger.info(f"✓ Model loaded successfully from local installation")
                logger.info(f"✓ Embedding dimension: {self.embedding_dim}")
                logger.info(f"✓ Max sequence length: {config['max_seq_len']}")
            
        except Exception as e:
            logger.error(f"Failed to load HyenaDNA model: {e}")
            logger.warning("Falling back to simple DNA classifier...")
            
        # Fallback to enhanced classifier
        from enhanced_dna_classifier_fixed import EnhancedDNAClassifier
        self.simple_classifier = EnhancedDNAClassifier(device=str(self.device))
        self.model = None
        self.tokenizer = None
        self.embedding_dim = self.simple_classifier.embedding_dim
        
        logger.info("✓ Using enhanced DNA classifier as fallback")
    
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
        
        # HyenaDNA uses character-level tokenization
        # Convert to uppercase and tokenize
        sequence = sequence.upper()
        
        # Tokenize the sequence
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings
        )
        
        return tokens.to(self.device)
    
    def extract_embeddings(self, 
                          sequences: List[str], 
                          pooling_strategy: str = "mean",
                          layer_indices: Optional[List[int]] = None,
                          batch_size: int = 4) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from DNA sequences.
        
        Args:
            sequences: List of DNA sequences
            pooling_strategy: How to pool embeddings ('mean', 'max', 'cls', 'last')
            layer_indices: Which layers to extract from (None for all)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with embeddings for each layer
        """
        if self.model is None and not hasattr(self, 'simple_classifier'):
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use simple classifier if HyenaDNA is not available
        if hasattr(self, 'simple_classifier'):
            logger.info("Using simple DNA classifier for embedding extraction")
            return self.simple_classifier.extract_embeddings(sequences, pooling_strategy, batch_size)
        
        logger.info(f"Extracting embeddings from {len(sequences)} sequences")
        logger.info(f"Using pooling strategy: {pooling_strategy}")
        
        all_embeddings = {}
        
        # Process sequences in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
            
            # Tokenize batch
            batch_tokens = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**batch_tokens, output_hidden_states=True)
                
                # Extract embeddings from specified layers
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # HuggingFace model with hidden states
                    if layer_indices is None:
                        # Use all layers
                        hidden_states = outputs.hidden_states
                    else:
                        # Use specified layers
                        hidden_states = [outputs.hidden_states[i] for i in layer_indices]
                else:
                    # Fallback: use last_hidden_state as single layer
                    hidden_states = [outputs.last_hidden_state]
                
                # Process each layer
                for layer_idx, hidden_state in enumerate(hidden_states):
                    layer_key = f"layer_{layer_idx}"
                    
                    if layer_key not in all_embeddings:
                        all_embeddings[layer_key] = []
                    
                    # Pool embeddings
                    if pooling_strategy == "mean":
                        # Mean pooling (excluding padding tokens)
                        attention_mask = batch_tokens['attention_mask']
                        pooled = self._mean_pooling(hidden_state, attention_mask)
                    elif pooling_strategy == "max":
                        # Max pooling
                        pooled = torch.max(hidden_state, dim=1)[0]
                    elif pooling_strategy == "cls":
                        # CLS token (first token)
                        pooled = hidden_state[:, 0, :]
                    elif pooling_strategy == "last":
                        # Last token
                        pooled = hidden_state[:, -1, :]
                    else:
                        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                    
                    all_embeddings[layer_key].append(pooled.cpu().numpy())
        
        # Concatenate all batches for each layer
        for layer_key in all_embeddings:
            all_embeddings[layer_key] = np.vstack(all_embeddings[layer_key])
        
        logger.info(f"✓ Extracted embeddings for {len(all_embeddings)} layers")
        for layer_key, embeddings in all_embeddings.items():
            logger.info(f"  {layer_key}: {embeddings.shape}")
        
        return all_embeddings
    
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
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_seq_len": self.model.config.max_position_embeddings,
            "num_layers": self.model.config.num_hidden_layers,
            "device": str(self.device)
        }

def test_hyenadna_loading():
    """Test function to verify HyenaDNA can be loaded."""
    try:
        wrapper = HyenaDNAWrapper("hyenadna-small-32k-seqlen")
        wrapper.load_model()
        
        # Test with a short sequence
        test_sequence = "ATCGATCGATCGATCG"
        tokens = wrapper.tokenize_sequence(test_sequence)
        print(f"✓ Tokenization successful: {tokens['input_ids'].shape}")
        
        # Test embedding extraction
        embeddings = wrapper.extract_embeddings([test_sequence])
        print(f"✓ Embedding extraction successful: {len(embeddings)} layers")
        
        return True
    except Exception as e:
        print(f"✗ HyenaDNA test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the wrapper
    test_hyenadna_loading()

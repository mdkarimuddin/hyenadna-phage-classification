#!/usr/bin/env python3
"""
HyenaDNA Embedding Extraction Script
Extracts embeddings from DNA sequences using HyenaDNA models.
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from hyenadna_wrapper import HyenaDNAWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_dir: str, split: str) -> Dict:
    """Load processed data for a specific split."""
    filepath = Path(data_dir) / f"data_{split}.pkl"
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded {split} data: {len(data['sequence'])} sequences")
    return data

def extract_embeddings_for_split(wrapper: HyenaDNAWrapper, 
                                data: Dict, 
                                split: str,
                                output_dir: str,
                                batch_size: int = 4,
                                pooling_strategy: str = "mean") -> Dict:
    """
    Extract embeddings for a specific data split.
    
    Args:
        wrapper: HyenaDNA wrapper instance
        data: Data dictionary containing sequences and labels
        split: Split name (train/val/test)
        output_dir: Output directory for embeddings
        batch_size: Batch size for processing
        pooling_strategy: Pooling strategy for embeddings
        
    Returns:
        Dictionary containing embeddings for all layers
    """
    logger.info(f"Extracting embeddings for {split} set...")
    
    sequences = data['sequence']
    labels = data['label']
    
    # Extract embeddings
    all_embeddings = wrapper.extract_embeddings(
        sequences=sequences,
        pooling_strategy=pooling_strategy,
        batch_size=batch_size
    )
    
    # Add labels to each layer's embeddings
    embeddings_with_labels = {}
    for layer_name, embeddings in all_embeddings.items():
        embeddings_with_labels[layer_name] = {
            'embeddings': embeddings,
            'labels': labels,
            'sequence_ids': data['sequence_id']
        }
    
    # Save embeddings
    output_file = Path(output_dir) / f"embeddings_{split}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_with_labels, f)
    
    logger.info(f"✓ Saved {split} embeddings to {output_file}")
    
    # Log embedding statistics
    for layer_name, layer_data in embeddings_with_labels.items():
        emb_shape = layer_data['embeddings'].shape
        logger.info(f"  {layer_name}: {emb_shape[0]} sequences × {emb_shape[1]} features")
    
    return embeddings_with_labels

def analyze_sequence_lengths(data: Dict) -> Dict:
    """Analyze sequence lengths in the data."""
    lengths = [len(seq) for seq in data['sequence']]
    
    stats = {
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths)
    }
    
    logger.info(f"Sequence length statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:,.0f}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Extract HyenaDNA embeddings from DNA sequences')
    parser.add_argument('--model_name', default='hyenadna-medium-160k-seqlen', 
                       help='HyenaDNA model name to use')
    parser.add_argument('--data_dir', default='data/processed', 
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', default='embeddings', 
                       help='Output directory for embeddings')
    parser.add_argument('--batch_size', type=int, default=2, 
                       help='Batch size for processing')
    parser.add_argument('--pooling_strategy', default='mean', 
                       choices=['mean', 'max', 'cls', 'last'],
                       help='Pooling strategy for embeddings')
    parser.add_argument('--device', default='cuda', 
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Data splits to process')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("HYNEDNA EMBEDDING EXTRACTION")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Pooling strategy: {args.pooling_strategy}")
    logger.info("")
    
    # Initialize HyenaDNA wrapper
    logger.info("Initializing HyenaDNA wrapper...")
    wrapper = HyenaDNAWrapper(model_name=args.model_name, device=args.device)
    wrapper.load_model()
    
    # Get model info
    model_info = wrapper.get_model_info()
    logger.info(f"Model info: {model_info}")
    logger.info("")
    
    # Process each split
    all_results = {}
    
    for split in args.splits:
        logger.info(f"Processing {split} split...")
        
        # Load data
        data = load_data(args.data_dir, split)
        
        # Analyze sequence lengths
        length_stats = analyze_sequence_lengths(data)
        
        # Extract embeddings
        embeddings = extract_embeddings_for_split(
            wrapper=wrapper,
            data=data,
            split=split,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            pooling_strategy=args.pooling_strategy
        )
        
        all_results[split] = {
            'embeddings': embeddings,
            'length_stats': length_stats
        }
        
        logger.info(f"✓ Completed {split} split")
        logger.info("")
    
    # Save summary
    summary = {
        'model_name': args.model_name,
        'model_info': model_info,
        'pooling_strategy': args.pooling_strategy,
        'batch_size': args.batch_size,
        'results': all_results
    }
    
    summary_file = Path(args.output_dir) / "extraction_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    logger.info("="*60)
    logger.info("✅ EMBEDDING EXTRACTION COMPLETE")
    logger.info("="*60)
    
    # Print summary
    for split, result in all_results.items():
        logger.info(f"{split.upper()} SET:")
        for layer_name, layer_data in result['embeddings'].items():
            emb_shape = layer_data['embeddings'].shape
            logger.info(f"  {layer_name}: {emb_shape[0]} sequences × {emb_shape[1]} features")
        
        stats = result['length_stats']
        logger.info(f"  Sequence lengths: {stats['min_length']:,} - {stats['max_length']:,} bp (mean: {stats['mean_length']:,.0f})")
        logger.info("")
    
    logger.info(f"All embeddings saved to: {args.output_dir}/")
    logger.info(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()




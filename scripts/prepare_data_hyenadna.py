#!/usr/bin/env python3
"""
Data Preparation for HyenaDNA Analysis
Prepares phage, plasmid, and phage-plasmid sequences for HyenaDNA processing.
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fasta_files(directory: str) -> List[Tuple[str, str]]:
    """
    Load all FASTA files from a directory.
    
    Args:
        directory: Path to directory containing FASTA files
        
    Returns:
        List of (sequence_id, sequence) tuples
    """
    sequences = []
    fasta_files = list(Path(directory).glob("*.fasta"))
    
    logger.info(f"Loading FASTA files from {directory}")
    logger.info(f"Found {len(fasta_files)} FASTA files")
    
    for fasta_file in fasta_files:
        try:
            with open(fasta_file, 'r') as f:
                current_seq = ""
                current_id = ""
                
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq and current_id:
                            sequences.append((current_id, current_seq))
                        current_id = line[1:]  # Remove '>'
                        current_seq = ""
                    else:
                        current_seq += line
                
                # Add the last sequence
                if current_seq and current_id:
                    sequences.append((current_id, current_seq))
                    
        except Exception as e:
            logger.warning(f"Error reading {fasta_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(sequences)} sequences from {directory}")
    return sequences

def create_labels(phage_seqs: List[Tuple[str, str]], 
                 plasmid_seqs: List[Tuple[str, str]], 
                 phage_plasmid_seqs: List[Tuple[str, str]]) -> Tuple[List[str], List[str], List[int]]:
    """
    Create sequences and labels for all classes.
    
    Args:
        phage_seqs: List of (id, sequence) for phages
        plasmid_seqs: List of (id, sequence) for plasmids  
        phage_plasmid_seqs: List of (id, sequence) for phage-plasmids
        
    Returns:
        Tuple of (all_sequences, all_ids, all_labels)
    """
    all_sequences = []
    all_ids = []
    all_labels = []
    
    # Add phage sequences (label 0)
    for seq_id, sequence in phage_seqs:
        all_sequences.append(sequence)
        all_ids.append(seq_id)
        all_labels.append(0)
    
    # Add plasmid sequences (label 1)
    for seq_id, sequence in plasmid_seqs:
        all_sequences.append(sequence)
        all_ids.append(seq_id)
        all_labels.append(1)
    
    # Add phage-plasmid sequences (label 2)
    for seq_id, sequence in phage_plasmid_seqs:
        all_sequences.append(sequence)
        all_ids.append(seq_id)
        all_labels.append(2)
    
    logger.info(f"Total sequences: {len(all_sequences)}")
    logger.info(f"  Phages: {sum(1 for l in all_labels if l == 0)}")
    logger.info(f"  Plasmids: {sum(1 for l in all_labels if l == 1)}")
    logger.info(f"  Phage-Plasmids: {sum(1 for l in all_labels if l == 2)}")
    
    return all_sequences, all_ids, all_labels

def analyze_sequence_lengths(sequences: List[str]) -> Dict[str, float]:
    """Analyze sequence length statistics."""
    lengths = [len(seq) for seq in sequences]
    
    stats = {
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths)
    }
    
    logger.info("Sequence length statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.2f}")
    
    return stats

def split_data(sequences: List[str], 
               ids: List[str], 
               labels: List[int],
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        sequences: List of DNA sequences
        ids: List of sequence IDs
        labels: List of labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    logger.info(f"Splitting data: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        sequences, labels, ids,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp,
        test_size=val_size,
        random_state=random_seed,
        stratify=y_temp
    )
    
    # Create data dictionaries
    train_data = {
        'sequence': X_train,
        'sequence_id': ids_train,
        'label': y_train
    }
    
    val_data = {
        'sequence': X_val,
        'sequence_id': ids_val,
        'label': y_val
    }
    
    test_data = {
        'sequence': X_test,
        'sequence_id': ids_test,
        'label': y_test
    }
    
    # Log split statistics
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        labels = split_data['label']
        logger.info(f"{split_name} set: {len(labels)} sequences")
        logger.info(f"  Phages: {sum(1 for l in labels if l == 0)}")
        logger.info(f"  Plasmids: {sum(1 for l in labels if l == 1)}")
        logger.info(f"  Phage-Plasmids: {sum(1 for l in labels if l == 2)}")
    
    return train_data, val_data, test_data

def save_data(data: Dict, filepath: str):
    """Save data dictionary to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved data to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Prepare data for HyenaDNA analysis')
    parser.add_argument('--phage_dir', required=True, help='Directory containing phage FASTA files')
    parser.add_argument('--plasmid_dir', required=True, help='Directory containing plasmid FASTA files')
    parser.add_argument('--phage_plasmid_dir', required=True, help='Directory containing phage-plasmid FASTA files')
    parser.add_argument('--output_dir', default='data/processed', help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("PREPARING DATA FOR HYNEDNA ANALYSIS")
    logger.info("="*60)
    
    # Load sequences from all directories
    logger.info("Loading sequences...")
    phage_seqs = load_fasta_files(args.phage_dir)
    plasmid_seqs = load_fasta_files(args.plasmid_dir)
    phage_plasmid_seqs = load_fasta_files(args.phage_plasmid_dir)
    
    # Create combined dataset
    logger.info("Creating combined dataset...")
    all_sequences, all_ids, all_labels = create_labels(phage_seqs, plasmid_seqs, phage_plasmid_seqs)
    
    # Analyze sequence lengths
    logger.info("Analyzing sequence lengths...")
    length_stats = analyze_sequence_lengths(all_sequences)
    
    # Split data
    logger.info("Splitting data...")
    train_data, val_data, test_data = split_data(
        all_sequences, all_ids, all_labels,
        args.train_ratio, args.val_ratio, args.test_ratio,
        args.random_seed
    )
    
    # Save processed data
    logger.info("Saving processed data...")
    save_data(train_data, f"{args.output_dir}/data_train.pkl")
    save_data(val_data, f"{args.output_dir}/data_val.pkl")
    save_data(test_data, f"{args.output_dir}/data_test.pkl")
    
    # Save metadata
    metadata = {
        'length_stats': length_stats,
        'total_sequences': len(all_sequences),
        'train_size': len(train_data['sequence']),
        'val_size': len(val_data['sequence']),
        'test_size': len(test_data['sequence']),
        'class_distribution': {
            'phages': sum(1 for l in all_labels if l == 0),
            'plasmids': sum(1 for l in all_labels if l == 1),
            'phage_plasmids': sum(1 for l in all_labels if l == 2)
        }
    }
    
    with open(f"{args.output_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info("="*60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total sequences: {len(all_sequences)}")
    logger.info(f"Max sequence length: {length_stats['max_length']:,} bp")
    logger.info(f"Mean sequence length: {length_stats['mean_length']:,.0f} bp")
    logger.info(f"Data saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()




"""Unit tests for HyenaDNA wrapper."""

import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hyenadna_wrapper import HyenaDNAWrapper


class TestHyenaDNAWrapper(unittest.TestCase):
    """Test cases for HyenaDNAWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.wrapper = HyenaDNAWrapper(
            model_name="hyenadna-small-32k-seqlen",
            device="cpu"  # Use CPU for testing
        )
    
    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertIsNotNone(self.wrapper)
        self.assertEqual(self.wrapper.model_name, "hyenadna-small-32k-seqlen")
    
    def test_validate_sequence(self):
        """Test sequence validation."""
        valid_seq = "ATCGATCGATCG"
        invalid_seq = "ATCGXYZATCG"
        
        # Check if validate_sequence method exists
        if hasattr(self.wrapper, 'validate_sequence'):
            self.assertTrue(self.wrapper.validate_sequence(valid_seq))
            self.assertFalse(self.wrapper.validate_sequence(invalid_seq))
    
    def test_tokenize_sequence(self):
        """Test sequence tokenization."""
        seq = "ATCGATCG"
        # Check if tokenize method exists
        if hasattr(self.wrapper, 'tokenize'):
            tokens = self.wrapper.tokenize(seq)
            self.assertIsNotNone(tokens)


if __name__ == '__main__':
    unittest.main()

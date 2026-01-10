#!/usr/bin/env python3
"""
Test HyenaDNA Model Loading
Simple test to verify HyenaDNA can be loaded and basic functionality works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from hyenadna_simple import SimpleHyenaDNA, test_simple_hyenadna

def main():
    print("🧬 Testing HyenaDNA Model Loading")
    print("="*50)
    
    # Test with smallest model first
    print("Testing with tiny model...")
    success = test_simple_hyenadna()
    
    if success:
        print("\n✅ HyenaDNA test PASSED!")
        print("Ready to run the full pipeline.")
    else:
        print("\n❌ HyenaDNA test FAILED!")
        print("Check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()

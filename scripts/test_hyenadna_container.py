#!/usr/bin/env python3
"""
Test HyenaDNA Container
Simple test to verify the container works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_container_environment():
    """Test if the container environment is working correctly."""
    print("🧬 Testing HyenaDNA Container Environment")
    print("="*60)
    
    # Test 1: Basic imports
    print("1. Testing basic imports...")
    try:
        import torch
        print(f"   ✓ PyTorch: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"   ✗ PyTorch import failed: {e}")
        return False
    
    # Test 2: Required packages
    print("\n2. Testing required packages...")
    packages = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'einops', 'biopython']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"   ✓ {pkg}")
        except Exception as e:
            print(f"   ✗ {pkg}: {e}")
            return False
    
    # Test 3: HyenaDNA imports
    print("\n3. Testing HyenaDNA imports...")
    try:
        # Add hyena-dna to path
        hyena_path = Path('/opt/hyena-dna')
        sys.path.append(str(hyena_path))
        
        from src.models.sequence.model import SequenceModel
        print("   ✓ SequenceModel import successful")
        
        from src.models.sequence.hyena import HyenaDNA
        print("   ✓ HyenaDNA import successful")
        
    except Exception as e:
        print(f"   ✗ HyenaDNA import failed: {e}")
        return False
    
    # Test 4: Simple model creation
    print("\n4. Testing model creation...")
    try:
        # Create a small test model
        model = SequenceModel(
            d_model=64,
            n_layers=2,
            layer={'name': 'hyena', 'd_model': 64, 'd_conv': 4, 'd_filter': 8, 'd_ff': 128, 'dropout': 0.1},
            residual={'name': 'hyena'},
            norm={'name': 'layer'},
            dropout=0.1
        )
        print("   ✓ Model creation successful")
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        x = torch.randn(batch_size, seq_len, 64)
        with torch.no_grad():
            output = model(x)
        print(f"   ✓ Forward pass successful: {x.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"   ✗ Model test failed: {e}")
        return False
    
    # Test 5: GPU test (if available)
    print("\n5. Testing GPU functionality...")
    if torch.cuda.is_available():
        try:
            model_gpu = model.cuda()
            x_gpu = x.cuda()
            with torch.no_grad():
                output_gpu = model_gpu(x_gpu)
            print(f"   ✓ GPU test successful: {x_gpu.shape} -> {output_gpu.shape}")
        except Exception as e:
            print(f"   ✗ GPU test failed: {e}")
            return False
    else:
        print("   ⚠ GPU not available, skipping GPU test")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("Container is ready for HyenaDNA analysis")
    print("="*60)
    
    return True

def main():
    success = test_container_environment()
    
    if success:
        print("\n🎯 Container is ready!")
        print("Next steps:")
        print("1. Run data preparation: sbatch slurm_scripts/run-prepare-data-container.sh")
        print("2. Run embedding extraction: sbatch slurm_scripts/run-extract-embeddings-container.sh")
    else:
        print("\n❌ Container test failed!")
        print("Check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()




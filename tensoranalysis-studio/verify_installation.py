#!/usr/bin/env python3
"""
Verification script to test Tensor Analysis Studio installation.

Run this script after installation to verify everything works:
    python verify_installation.py
"""

import sys
import numpy as np


def test_import():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from tas import Tensor, Index
        from tas.core.metrics import Metric, euclidean_metric, minkowski_metric
        from tas.core.einsum_parser import einsum_eval, contract, outer
        from tas.core.algebra import symmetrize, antisymmetrize, trace
        from tas.core.connections import Connection
        from tas.core.coords import CartesianFrame, SphericalFrame, CylindricalFrame
        print("✓ All core imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_tensor():
    """Test basic tensor creation and operations."""
    print("\nTesting basic tensor operations...")
    try:
        from tas import Tensor
        
        # Create tensor
        A = Tensor(np.array([[1, 2], [3, 4]]), indices=("^i", "_j"))
        
        # Check properties
        assert A.shape == (2, 2)
        assert A.ndim == 2
        assert len(A.indices) == 2
        
        # Transpose
        B = A.transpose((1, 0))
        assert B.shape == (2, 2)
        
        print("✓ Basic tensor operations work")
        return True
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False


def test_einstein_notation():
    """Test Einstein notation parser."""
    print("\nTesting Einstein notation...")
    try:
        from tas import Tensor
        from tas.core.einsum_parser import einsum_eval
        
        # Matrix multiplication
        A = Tensor(np.array([[1, 2], [3, 4]]), indices=("^i", "_j"))
        B = Tensor(np.array([[5, 6], [7, 8]]), indices=("^j", "_k"))
        
        C = einsum_eval("A^i_j B^j_k", A=A, B=B)
        
        # Verify result
        expected = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
        assert np.allclose(C.data, expected)
        
        print("✓ Einstein notation works")
        return True
    except Exception as e:
        print(f"✗ Einstein notation failed: {e}")
        return False


def test_metrics():
    """Test metric tensors and raise/lower."""
    print("\nTesting metrics and index operations...")
    try:
        from tas import Tensor
        from tas.core.metrics import euclidean_metric
        
        g = euclidean_metric(3)
        
        # Create vector
        V = Tensor(np.array([1, 2, 3]), indices=("_i",))
        
        # Raise index
        V_up = g.raise_index(V, axis=0)
        assert V_up.indices[0].variance == "up"
        
        # Lower it back
        V_back = g.lower_index(V_up, axis=0)
        assert np.allclose(V_back.data, V.data)
        
        print("✓ Metrics and index operations work")
        return True
    except Exception as e:
        print(f"✗ Metrics failed: {e}")
        return False


def test_coordinates():
    """Test coordinate systems."""
    print("\nTesting coordinate systems...")
    try:
        from tas.core.coords import SphericalFrame
        
        spherical = SphericalFrame()
        
        # Convert point
        cart_point = np.array([1.0, 1.0, 1.0])
        sph_point = spherical.from_cartesian(cart_point)
        
        # Convert back
        cart_back = spherical.to_cartesian(sph_point)
        assert np.allclose(cart_back, cart_point, atol=1e-10)
        
        print("✓ Coordinate systems work")
        return True
    except Exception as e:
        print(f"✗ Coordinate systems failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Tensor Analysis Studio - Installation Verification")
    print("=" * 60)
    
    tests = [
        test_import,
        test_basic_tensor,
        test_einstein_notation,
        test_metrics,
        test_coordinates,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    if all(results):
        print("SUCCESS: All tests passed! ✓")
        print("Tensor Analysis Studio is ready to use.")
        return 0
    else:
        print("FAILURE: Some tests failed. ✗")
        print("Please check your installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

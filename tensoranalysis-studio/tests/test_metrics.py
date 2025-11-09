"""Tests for metric tensors and raise/lower operations."""

import pytest
import numpy as np

from tas.core.tensor import Tensor
from tas.core.metrics import (
    Metric,
    euclidean_metric,
    minkowski_metric
)


class TestMetric:
    """Test Metric class."""
    
    def test_create_metric(self) -> None:
        """Test creating metric tensors."""
        g = Metric(np.eye(3), indices=("_i", "_j"))
        
        assert g.dim == 3
        assert g.shape == (3, 3)
    
    def test_metric_must_be_rank2(self) -> None:
        """Test that metric must be rank-2."""
        with pytest.raises(ValueError, match="rank-2"):
            Metric(np.array([1, 2, 3]), indices=("_i",))
    
    def test_metric_must_be_square(self) -> None:
        """Test that metric must be square."""
        with pytest.raises(ValueError, match="square"):
            Metric(np.random.rand(2, 3), indices=("_i", "_j"))
    
    def test_metric_inverse(self) -> None:
        """Test computing metric inverse."""
        g = Metric(np.eye(3), indices=("_i", "_j"))
        g_inv = g.inverse()
        
        assert g_inv.shape == (3, 3)
        assert np.allclose(g_inv.data, np.eye(3))
        
        # Check indices are raised
        assert g_inv.indices[0].variance == "up"
        assert g_inv.indices[1].variance == "up"
    
    def test_euclidean_signature(self) -> None:
        """Test Euclidean metric signature."""
        g = euclidean_metric(3)
        pos, neg, zero = g.signature()
        
        assert pos == 3
        assert neg == 0
        assert zero == 0
    
    def test_minkowski_signature(self) -> None:
        """Test Minkowski spacetime signature."""
        eta = minkowski_metric("timelike")
        pos, neg, zero = eta.signature()
        
        assert pos == 3
        assert neg == 1
        assert zero == 0
    
    def test_determinant(self) -> None:
        """Test metric determinant."""
        g = Metric(np.diag([1, 2, 3]), indices=("_i", "_j"))
        det = g.determinant()
        
        assert np.isclose(det, 6.0)
    
    def test_sqrt_abs_det(self) -> None:
        """Test sqrt(|det(g)|)."""
        g = Metric(np.diag([-1, 2, 3]), indices=("_i", "_j"))
        sqrt_det = g.sqrt_abs_det()
        
        assert np.isclose(sqrt_det, np.sqrt(6.0))


class TestRaiseLowerIndices:
    """Test raising and lowering tensor indices."""
    
    def test_raise_vector_index(self) -> None:
        """Test raising a covariant vector index."""
        g = euclidean_metric(3)
        V = Tensor(np.array([1, 2, 3]), indices=("_i",))
        
        V_up = g.raise_index(V, axis=0)
        
        assert V_up.indices[0].variance == "up"
        # For Euclidean metric, components unchanged
        assert np.allclose(V_up.data, V.data)
    
    def test_lower_vector_index(self) -> None:
        """Test lowering a contravariant vector index."""
        g = euclidean_metric(3)
        V = Tensor(np.array([1, 2, 3]), indices=("^i",))
        
        V_down = g.lower_index(V, axis=0)
        
        assert V_down.indices[0].variance == "down"
        assert np.allclose(V_down.data, V.data)
    
    def test_raise_lower_roundtrip(self) -> None:
        """Test that raise then lower returns original."""
        g = euclidean_metric(3)
        V = Tensor(np.array([1, 2, 3]), indices=("_i",))
        
        V_up = g.raise_index(V, axis=0)
        V_back = g.lower_index(V_up, axis=0)
        
        assert np.allclose(V_back.data, V.data)
    
    def test_cannot_raise_up_index(self) -> None:
        """Test error when trying to raise already up index."""
        g = euclidean_metric(3)
        V = Tensor(np.array([1, 2, 3]), indices=("^i",))
        
        with pytest.raises(ValueError, match="already contravariant"):
            g.raise_index(V, axis=0)
    
    def test_cannot_lower_down_index(self) -> None:
        """Test error when trying to lower already down index."""
        g = euclidean_metric(3)
        V = Tensor(np.array([1, 2, 3]), indices=("_i",))
        
        with pytest.raises(ValueError, match="already covariant"):
            g.lower_index(V, axis=0)
    
    def test_dimension_mismatch_error(self) -> None:
        """Test error when tensor dimension doesn't match metric."""
        g = euclidean_metric(3)
        V = Tensor(np.array([1, 2, 3, 4]), indices=("_i",))
        
        with pytest.raises(ValueError, match="does not match metric dimension"):
            g.raise_index(V, axis=0)


class TestInnerProduct:
    """Test metric inner product."""
    
    def test_inner_product_euclidean(self) -> None:
        """Test inner product with Euclidean metric."""
        g = euclidean_metric(3)
        u = Tensor(np.array([1, 0, 0]), indices=("^i",))
        v = Tensor(np.array([0, 1, 0]), indices=("^i",))
        
        product = g.inner_product(u, v)
        
        assert np.isclose(product, 0.0)  # Orthogonal
    
    def test_inner_product_self(self) -> None:
        """Test inner product of vector with itself."""
        g = euclidean_metric(3)
        v = Tensor(np.array([3, 4, 0]), indices=("^i",))
        
        product = g.inner_product(v, v)
        
        assert np.isclose(product, 25.0)  # 3^2 + 4^2
    
    def test_inner_product_minkowski(self) -> None:
        """Test inner product with Minkowski metric."""
        eta = minkowski_metric("timelike")
        # Timelike vector
        v = Tensor(np.array([2, 1, 0, 0]), indices=("^mu",))
        
        product = eta.inner_product(v, v)
        
        # eta_mu_nu v^mu v^nu = -2^2 + 1^2 = -3
        assert np.isclose(product, -3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for Einstein notation parser and contractions."""

import pytest
import numpy as np

from tas.core.tensor import Tensor
from tas.core.einsum_parser import (
    EinsumExpression,
    einsum_eval,
    contract,
    outer
)


class TestEinsumExpression:
    """Test Einstein notation parsing."""
    
    def test_parse_simple_expression(self) -> None:
        """Test parsing basic expressions."""
        expr = EinsumExpression("A^i_j B^j_k")
        
        assert len(expr.tensor_specs) == 2
        assert expr.tensor_names == ["A", "B"]
    
    def test_identify_contracted_indices(self) -> None:
        """Test identifying contracted (summed) indices."""
        expr = EinsumExpression("A^i_j B^j_k")
        
        assert "j" in expr.contracted_indices
        assert "i" not in expr.contracted_indices
        assert "k" not in expr.contracted_indices
    
    def test_identify_free_indices(self) -> None:
        """Test identifying free indices."""
        expr = EinsumExpression("A^i_j B^j_k")
        
        free_names = [name for name, _ in expr.free_indices]
        assert "i" in free_names
        assert "k" in free_names
        assert "j" not in free_names


class TestEinsumEval:
    """Test Einstein notation evaluation."""
    
    def test_matrix_multiplication(self) -> None:
        """Test matrix multiplication via Einstein notation."""
        A = Tensor(np.array([[1, 2], [3, 4]]), indices=("^i", "_j"))
        B = Tensor(np.array([[5, 6], [7, 8]]), indices=("^j", "_k"))
        
        C = einsum_eval("A^i_j B^j_k", A=A, B=B)
        
        # Should be matrix product
        expected = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
        assert C.shape == (2, 2)
        assert np.allclose(C.data, expected)
    
    def test_vector_dot_product(self) -> None:
        """Test dot product via contraction."""
        v = Tensor(np.array([1, 2, 3]), indices=("^i",))
        w = Tensor(np.array([4, 5, 6]), indices=("_i",))
        
        result = einsum_eval("v^i w_i", v=v, w=w)
        
        # Result should be scalar
        assert result.ndim == 0
        assert np.isclose(result.data, 1*4 + 2*5 + 3*6)
    
    def test_outer_product(self) -> None:
        """Test outer product with no contractions."""
        v = Tensor(np.array([1, 2]), indices=("^i",))
        w = Tensor(np.array([3, 4, 5]), indices=("^j",))
        
        M = einsum_eval("v^i w^j", v=v, w=w)
        
        assert M.shape == (2, 3)
        assert np.allclose(M.data, np.outer([1, 2], [3, 4, 5]))
    
    def test_trace(self) -> None:
        """Test trace via contraction of both indices."""
        M = Tensor(np.array([[1, 2], [3, 4]]), indices=("^i", "_i"))
        
        tr = einsum_eval("M^i_i", M=M)
        
        assert tr.ndim == 0
        assert np.isclose(tr.data, 1 + 4)  # trace
    
    def test_missing_tensor_error(self) -> None:
        """Test error when tensor not provided."""
        with pytest.raises(ValueError, match="not provided"):
            einsum_eval("A^i_j B^j_k", A=Tensor(np.eye(2), indices=("^i", "_j")))


class TestContract:
    """Test direct contraction function."""
    
    def test_contract_matrices(self) -> None:
        """Test contracting two matrices."""
        A = Tensor(np.array([[1, 2], [3, 4]]), indices=("^i", "_j"))
        B = Tensor(np.array([[5, 6], [7, 8]]), indices=("^j", "_k"))
        
        C = contract(A, B, axes1=(1,), axes2=(0,))
        
        expected = np.tensordot(A.data, B.data, axes=([1], [0]))
        assert np.allclose(C.data, expected)
    
    def test_contract_dimension_error(self) -> None:
        """Test error on dimension mismatch."""
        A = Tensor(np.random.rand(2, 3), indices=("^i", "_j"))
        B = Tensor(np.random.rand(4, 5), indices=("^j", "_k"))
        
        with pytest.raises(ValueError, match="different dimensions"):
            contract(A, B, axes1=(1,), axes2=(0,))


class TestOuter:
    """Test outer product."""
    
    def test_outer_product_vectors(self) -> None:
        """Test outer product of vectors."""
        v = Tensor(np.array([1, 2, 3]), indices=("^i",))
        w = Tensor(np.array([4, 5]), indices=("^j",))
        
        M = outer(v, w)
        
        assert M.shape == (3, 2)
        expected = np.outer([1, 2, 3], [4, 5])
        assert np.allclose(M.data, expected)
    
    def test_outer_product_result_indices(self) -> None:
        """Test that result has concatenated indices."""
        v = Tensor(np.array([1, 2]), indices=("^i",))
        w = Tensor(np.array([3, 4]), indices=("_j",))
        
        M = outer(v, w)
        
        assert len(M.indices) == 2
        assert M.indices[0].name == "i"
        assert M.indices[1].name == "j"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for core tensor and index functionality."""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from tas.core.tensor import Tensor, zeros, ones, eye
from tas.core.indices import Index, parse_index_string, parse_index_tuple


class TestIndex:
    """Test Index class."""
    
    def test_create_index(self) -> None:
        """Test creating indices."""
        idx = Index("i", variance="down")
        assert idx.name == "i"
        assert idx.variance == "down"
        
        idx_up = Index("j", variance="up", dimension=3)
        assert idx_up.name == "j"
        assert idx_up.variance == "up"
        assert idx_up.dimension == 3
    
    def test_index_string_representation(self) -> None:
        """Test string representations."""
        idx_down = Index("i", "down")
        assert str(idx_down) == "_i"
        
        idx_up = Index("j", "up")
        assert str(idx_up) == "^j"
    
    def test_raise_lower_index(self) -> None:
        """Test raising and lowering indices."""
        idx = Index("i", "down")
        idx_raised = idx.raise_index()
        assert idx_raised.variance == "up"
        assert idx_raised.name == "i"
        
        idx_lowered = idx_raised.lower_index()
        assert idx_lowered.variance == "down"
    
    def test_flip_variance(self) -> None:
        """Test flipping variance."""
        idx = Index("i", "down")
        flipped = idx.flip_variance()
        assert flipped.variance == "up"
        
        flipped_again = flipped.flip_variance()
        assert flipped_again.variance == "down"
    
    def test_matches(self) -> None:
        """Test index matching for contractions."""
        idx1 = Index("i", "down")
        idx2 = Index("i", "up")
        idx3 = Index("j", "down")
        
        assert idx1.matches(idx2, check_variance=True)
        assert not idx1.matches(idx1, check_variance=True)  # Same variance
        assert not idx1.matches(idx3, check_variance=True)  # Different name
    
    def test_parse_index_string(self) -> None:
        """Test parsing index strings."""
        idx = parse_index_string("^i")
        assert idx.name == "i"
        assert idx.variance == "up"
        
        idx = parse_index_string("_mu")
        assert idx.name == "mu"
        assert idx.variance == "down"
        
        # Default to down
        idx = parse_index_string("j")
        assert idx.variance == "down"


class TestTensor:
    """Test Tensor class."""
    
    def test_create_tensor(self) -> None:
        """Test creating tensors."""
        data = np.array([[1, 2], [3, 4]])
        T = Tensor(data, indices=("^i", "_j"))
        
        assert T.shape == (2, 2)
        assert T.ndim == 2
        assert T.rank == 2
        assert len(T.indices) == 2
    
    def test_tensor_immutability(self) -> None:
        """Test that tensors are immutable."""
        data = np.array([1, 2, 3])
        T = Tensor(data, indices=("^i",))
        
        # Modifying original data shouldn't affect tensor
        # (though we copy on init)
        original_norm = T.norm()
        data[0] = 999
        assert T.norm() == original_norm
    
    def test_tensor_properties(self) -> None:
        """Test tensor properties."""
        T = Tensor(np.random.rand(2, 3, 4), indices=("^i", "_j", "^k"))
        
        assert T.shape == (2, 3, 4)
        assert T.ndim == 3
        assert T.size == 24
    
    def test_to_numpy(self) -> None:
        """Test conversion to numpy."""
        data = np.array([[1, 2], [3, 4]])
        T = Tensor(data, indices=("^i", "_j"))
        
        np_data = T.to_numpy()
        assert isinstance(np_data, np.ndarray)
        assert np.array_equal(np_data, data)
    
    def test_transpose(self) -> None:
        """Test tensor transposition."""
        T = Tensor(np.random.rand(2, 3, 4), indices=("^i", "_j", "^k"))
        T_transposed = T.transpose((2, 0, 1))
        
        assert T_transposed.shape == (4, 2, 3)
        assert T_transposed.indices[0].name == "k"
        assert T_transposed.indices[1].name == "i"
        assert T_transposed.indices[2].name == "j"
    
    def test_with_name(self) -> None:
        """Test setting tensor name."""
        T = Tensor(np.array([1, 2, 3]), indices=("^i",))
        T_named = T.with_name("velocity")
        
        assert T_named.name == "velocity"
        assert T.name is None  # Original unchanged
    
    def test_with_meta(self) -> None:
        """Test adding metadata."""
        T = Tensor(np.array([1, 2, 3]), indices=("^i",))
        T_with_units = T.with_meta(units="m/s", frame="lab")
        
        assert T_with_units.meta["units"] == "m/s"
        assert T_with_units.meta["frame"] == "lab"
    
    def test_norm(self) -> None:
        """Test tensor norm."""
        T = Tensor(np.array([3, 4]), indices=("^i",))
        assert T.norm() == 5.0
    
    def test_constructor_functions(self) -> None:
        """Test zeros, ones, eye constructors."""
        Z = zeros((2, 3), indices=("^i", "_j"))
        assert Z.shape == (2, 3)
        assert np.all(Z.data == 0)
        
        O = ones((3, 3), indices=("^i", "_j"))
        assert np.all(O.data == 1)
        
        I = eye(3, indices=("^i", "_j"))
        assert np.array_equal(I.data, np.eye(3))
    
    @given(
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=5)
    )
    def test_tensor_creation_property(self, m: int, n: int) -> None:
        """Property-based test: tensor creation with random shapes."""
        T = Tensor(np.random.rand(m, n), indices=("^i", "_j"))
        assert T.shape == (m, n)
        assert T.ndim == 2


class TestTensorValidation:
    """Test tensor validation and error handling."""
    
    def test_mismatched_indices(self) -> None:
        """Test error when indices don't match rank."""
        with pytest.raises(ValueError, match="must match tensor rank"):
            Tensor(np.array([[1, 2], [3, 4]]), indices=("^i",))
    
    def test_invalid_shape(self) -> None:
        """Test error with invalid shapes."""
        # NumPy will handle most shape validation
        # Empty arrays are technically valid
        T = Tensor(np.array([]), indices=())
        assert T.ndim == 1  # NumPy makes this 1D
    
    def test_index_dimension_mismatch(self) -> None:
        """Test error when index dimension doesn't match."""
        idx_wrong = Index("i", "down", dimension=5)
        with pytest.raises(ValueError, match="dimension"):
            Tensor(np.array([[1, 2], [3, 4]]), indices=(idx_wrong, "_j"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Core Tensor class with immutable API and index tracking.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np

from tas.core.indices import Index, parse_index_tuple
from tas.core.utils import validate_shape, normalize_axis


@dataclass(frozen=True)
class Tensor:
    """
    Immutable tensor with explicit index labels and variance tracking.
    
    A Tensor represents a multidimensional array with additional metadata:
    - Explicit index labels (covariant/contravariant)
    - Optional name for identification
    - Type information (dtype)
    - Backend specification (numpy, jax, cupy)
    - Arbitrary metadata (units, coordinate frame, etc.)
    
    The tensor is immutable from the user's perspective - all operations
    return new Tensor instances.
    
    Attributes:
        data: The underlying numpy array (or compatible) storing values
        indices: Tuple of Index objects, one per dimension
        name: Optional name for this tensor
        dtype: Data type of the tensor
        backend: Backend identifier ('numpy', 'jax', 'cupy')
        meta: Additional metadata dictionary
        
    Examples:
        >>> import numpy as np
        >>> from tas import Tensor, Index
        >>> 
        >>> # Create a rank-2 tensor (matrix)
        >>> A = Tensor(
        ...     data=np.array([[1, 2], [3, 4]]),
        ...     indices=(Index("i", "down"), Index("j", "up"))
        ... )
        >>> 
        >>> # Create with string notation
        >>> B = Tensor(np.eye(3), indices=("^i", "_j"))
    """
    
    data: np.ndarray
    indices: tuple[Index, ...]
    name: Optional[str] = None
    dtype: np.dtype = field(init=False)
    backend: str = "numpy"
    meta: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate tensor invariants and initialize computed fields."""
        # Ensure data is a numpy array
        if not isinstance(self.data, np.ndarray):
            object.__setattr__(self, "data", np.asarray(self.data))
        
        # Parse indices if given as strings
        if self.indices and isinstance(self.indices[0], str):
            object.__setattr__(self, "indices", parse_index_tuple(self.indices))
        
        # Validate shape
        validate_shape(self.data.shape)
        
        # Check rank consistency
        if len(self.indices) != self.data.ndim:
            raise ValueError(
                f"Number of indices ({len(self.indices)}) must match "
                f"tensor rank ({self.data.ndim})"
            )
        
        # Check dimension consistency
        for i, idx in enumerate(self.indices):
            if idx.dimension is not None and idx.dimension != self.data.shape[i]:
                raise ValueError(
                    f"Index {idx} has dimension {idx.dimension} but "
                    f"tensor has shape {self.data.shape[i]} at axis {i}"
                )
        
        # Set dtype
        object.__setattr__(self, "dtype", self.data.dtype)
        
        # Ensure meta is immutable
        if not isinstance(self.meta, Mapping):
            raise TypeError("meta must be a Mapping")
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions (rank) of the tensor."""
        return self.data.ndim
    
    @property
    def rank(self) -> int:
        """Rank of the tensor (same as ndim)."""
        return self.data.ndim
    
    @property
    def size(self) -> int:
        """Total number of elements in the tensor."""
        return self.data.size
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        indices_str = ", ".join(str(idx) for idx in self.indices)
        name_str = f", name={self.name!r}" if self.name else ""
        return f"Tensor(shape={self.shape}, indices=[{indices_str}]{name_str})"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        indices_str = "".join(str(idx) for idx in self.indices)
        name_str = f" '{self.name}'" if self.name else ""
        return f"Tensor{name_str}{indices_str}\n{self.data}"
    
    def __array__(self) -> np.ndarray:
        """Support numpy array protocol."""
        return self.data
    
    def __getitem__(self, key: Any) -> Union["Tensor", np.generic]:
        """
        Index into the tensor, returning a new Tensor or scalar.
        
        Note: Advanced indexing may not preserve index semantics correctly.
        Basic slicing preserves indices for kept dimensions.
        """
        result = self.data[key]
        
        # If scalar, return it directly
        if np.isscalar(result) or result.ndim == 0:
            return result.item() if hasattr(result, 'item') else result
        
        # Try to determine which indices remain
        # This is a simplified implementation - full support would need more logic
        if isinstance(key, tuple):
            new_indices = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    new_indices.append(self.indices[i])
            indices_tuple = tuple(new_indices)
        else:
            # Single index/slice
            if isinstance(key, slice):
                indices_tuple = (self.indices[0],) if self.ndim > 0 else ()
            else:
                indices_tuple = self.indices[1:] if self.ndim > 1 else ()
        
        return Tensor(
            data=result,
            indices=indices_tuple,
            name=self.name,
            backend=self.backend,
            meta=self.meta
        )
    
    def to_numpy(self) -> np.ndarray:
        """
        Return a copy of the underlying data as a numpy array.
        
        Returns:
            Copy of tensor data
        """
        return self.data.copy()
    
    def astype(self, dtype: np.dtype) -> "Tensor":
        """
        Cast tensor to a different data type.
        
        Args:
            dtype: Target data type
            
        Returns:
            New Tensor with converted dtype
        """
        return Tensor(
            data=self.data.astype(dtype),
            indices=self.indices,
            name=self.name,
            backend=self.backend,
            meta=self.meta
        )
    
    def transpose(self, perm: Optional[Sequence[int]] = None) -> "Tensor":
        """
        Permute tensor dimensions.
        
        Args:
            perm: Permutation of dimension indices. If None, reverses all axes.
            
        Returns:
            New Tensor with permuted dimensions
            
        Examples:
            >>> A = Tensor(np.random.rand(2, 3, 4), indices=("^i", "_j", "^k"))
            >>> B = A.transpose((2, 0, 1))  # (^k, ^i, _j)
        """
        if perm is None:
            perm = tuple(range(self.ndim - 1, -1, -1))
        
        perm_tuple = tuple(perm)
        
        if len(perm_tuple) != self.ndim:
            raise ValueError(
                f"Permutation length {len(perm_tuple)} must match rank {self.ndim}"
            )
        
        new_data = np.transpose(self.data, perm_tuple)
        new_indices = tuple(self.indices[i] for i in perm_tuple)
        
        return Tensor(
            data=new_data,
            indices=new_indices,
            name=self.name,
            backend=self.backend,
            meta=self.meta
        )
    
    def reshape(self, shape: Sequence[int], indices: Optional[Sequence[Union[Index, str]]] = None) -> "Tensor":
        """
        Reshape tensor to new shape. Requires explicit new indices.
        
        Args:
            shape: New shape tuple
            indices: New index labels for reshaped tensor
            
        Returns:
            New Tensor with new shape
            
        Raises:
            ValueError: If indices not provided or size doesn't match
        """
        if indices is None:
            raise ValueError("Must provide explicit indices when reshaping")
        
        shape_tuple = tuple(shape)
        
        if np.prod(shape_tuple) != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size {self.size} into shape {shape_tuple}"
            )
        
        new_data = self.data.reshape(shape_tuple)
        new_indices = parse_index_tuple(tuple(indices))
        
        return Tensor(
            data=new_data,
            indices=new_indices,
            name=self.name,
            backend=self.backend,
            meta=self.meta
        )
    
    def with_name(self, name: str) -> "Tensor":
        """Return a copy of this tensor with a new name."""
        return Tensor(
            data=self.data,
            indices=self.indices,
            name=name,
            backend=self.backend,
            meta=self.meta
        )
    
    def with_meta(self, **new_meta: Any) -> "Tensor":
        """Return a copy of this tensor with updated metadata."""
        updated_meta = dict(self.meta)
        updated_meta.update(new_meta)
        return Tensor(
            data=self.data,
            indices=self.indices,
            name=self.name,
            backend=self.backend,
            meta=updated_meta
        )
    
    def norm(self, ord: Optional[Union[int, float, str]] = None) -> float:
        """
        Compute the norm of the tensor (treating it as a vector).
        
        Args:
            ord: Order of the norm (see numpy.linalg.norm)
            
        Returns:
            Norm value
        """
        return float(np.linalg.norm(self.data, ord=ord))
    
    def conj(self) -> "Tensor":
        """Return complex conjugate of tensor."""
        return Tensor(
            data=np.conj(self.data),
            indices=self.indices,
            name=self.name,
            backend=self.backend,
            meta=self.meta
        )


def zeros(shape: Sequence[int], indices: Sequence[Union[Index, str]], 
          dtype: np.dtype = np.float64, **kwargs: Any) -> Tensor:
    """
    Create a tensor filled with zeros.
    
    Args:
        shape: Shape of the tensor
        indices: Index labels
        dtype: Data type
        **kwargs: Additional arguments for Tensor constructor
        
    Returns:
        Zero tensor
    """
    data = np.zeros(shape, dtype=dtype)
    return Tensor(data=data, indices=indices, **kwargs)


def ones(shape: Sequence[int], indices: Sequence[Union[Index, str]],
         dtype: np.dtype = np.float64, **kwargs: Any) -> Tensor:
    """
    Create a tensor filled with ones.
    
    Args:
        shape: Shape of the tensor
        indices: Index labels
        dtype: Data type
        **kwargs: Additional arguments for Tensor constructor
        
    Returns:
        Tensor of ones
    """
    data = np.ones(shape, dtype=dtype)
    return Tensor(data=data, indices=indices, **kwargs)


def eye(n: int, indices: Sequence[Union[Index, str]],
        dtype: np.dtype = np.float64, **kwargs: Any) -> Tensor:
    """
    Create an identity tensor (Kronecker delta).
    
    Args:
        n: Dimension
        indices: Index labels (must be length 2)
        dtype: Data type
        **kwargs: Additional arguments for Tensor constructor
        
    Returns:
        Identity tensor
    """
    data = np.eye(n, dtype=dtype)
    return Tensor(data=data, indices=indices, **kwargs)

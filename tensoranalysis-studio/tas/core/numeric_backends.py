"""
Backend abstraction layer for different numeric computation libraries.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
import numpy as np


class Backend(ABC):
    """
    Abstract interface for numeric computation backends.
    
    Defines a minimal set of operations needed for tensor computations.
    Implementations: NumPyBackend, JAXBackend, CuPyBackend.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass
    
    @abstractmethod
    def asarray(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """Convert data to backend array."""
        pass
    
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create array of zeros."""
        pass
    
    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create array of ones."""
        pass
    
    @abstractmethod
    def eye(self, n: int, dtype: Any = None) -> Any:
        """Create identity matrix."""
        pass
    
    @abstractmethod
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """Perform Einstein summation."""
        pass
    
    @abstractmethod
    def tensordot(self, a: Any, b: Any, axes: Union[int, Tuple] = 2) -> Any:
        """Tensor dot product."""
        pass
    
    @abstractmethod
    def transpose(self, a: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transpose array dimensions."""
        pass
    
    @abstractmethod
    def reshape(self, a: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape array."""
        pass
    
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        pass
    
    @abstractmethod
    def inv(self, a: Any) -> Any:
        """Matrix inverse."""
        pass
    
    @abstractmethod
    def solve(self, a: Any, b: Any) -> Any:
        """Solve linear system."""
        pass
    
    @abstractmethod
    def eigvalsh(self, a: Any) -> Any:
        """Eigenvalues of Hermitian matrix."""
        pass
    
    @abstractmethod
    def det(self, a: Any) -> float:
        """Matrix determinant."""
        pass
    
    @abstractmethod
    def trace(self, a: Any, axis1: int = 0, axis2: int = 1) -> Any:
        """Trace of array."""
        pass
    
    @abstractmethod
    def sum(self, a: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Sum array elements."""
        pass
    
    @abstractmethod
    def to_numpy(self, a: Any) -> np.ndarray:
        """Convert backend array to numpy."""
        pass


class NumPyBackend(Backend):
    """NumPy backend implementation (default)."""
    
    @property
    def name(self) -> str:
        return "numpy"
    
    def asarray(self, data: Any, dtype: Optional[Any] = None) -> np.ndarray:
        return np.asarray(data, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or np.float64)
    
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype or np.float64)
    
    def eye(self, n: int, dtype: Any = None) -> np.ndarray:
        return np.eye(n, dtype=dtype or np.float64)
    
    def einsum(self, subscripts: str, *operands: Any) -> np.ndarray:
        return np.einsum(subscripts, *operands)
    
    def tensordot(self, a: Any, b: Any, axes: Union[int, Tuple] = 2) -> np.ndarray:
        return np.tensordot(a, b, axes=axes)
    
    def transpose(self, a: Any, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        return np.transpose(a, axes=axes)
    
    def reshape(self, a: Any, shape: Tuple[int, ...]) -> np.ndarray:
        return np.reshape(a, shape)
    
    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.matmul(a, b)
    
    def inv(self, a: Any) -> np.ndarray:
        return np.linalg.inv(a)
    
    def solve(self, a: Any, b: Any) -> np.ndarray:
        return np.linalg.solve(a, b)
    
    def eigvalsh(self, a: Any) -> np.ndarray:
        return np.linalg.eigvalsh(a)
    
    def det(self, a: Any) -> float:
        return float(np.linalg.det(a))
    
    def trace(self, a: Any, axis1: int = 0, axis2: int = 1) -> np.ndarray:
        return np.trace(a, axis1=axis1, axis2=axis2)
    
    def sum(self, a: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        return np.sum(a, axis=axis)
    
    def to_numpy(self, a: Any) -> np.ndarray:
        return np.asarray(a)


class JAXBackend(Backend):
    """JAX backend for automatic differentiation and JIT compilation."""
    
    def __init__(self) -> None:
        try:
            import jax.numpy as jnp
            import jax
            self.jnp = jnp
            self.jax = jax
        except ImportError:
            raise ImportError(
                "JAX backend requires jax package. Install with: pip install jax jaxlib"
            )
    
    @property
    def name(self) -> str:
        return "jax"
    
    def asarray(self, data: Any, dtype: Optional[Any] = None) -> Any:
        return self.jnp.asarray(data, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        return self.jnp.zeros(shape, dtype=dtype or self.jnp.float64)
    
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        return self.jnp.ones(shape, dtype=dtype or self.jnp.float64)
    
    def eye(self, n: int, dtype: Any = None) -> Any:
        return self.jnp.eye(n, dtype=dtype or self.jnp.float64)
    
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return self.jnp.einsum(subscripts, *operands)
    
    def tensordot(self, a: Any, b: Any, axes: Union[int, Tuple] = 2) -> Any:
        return self.jnp.tensordot(a, b, axes=axes)
    
    def transpose(self, a: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        return self.jnp.transpose(a, axes=axes)
    
    def reshape(self, a: Any, shape: Tuple[int, ...]) -> Any:
        return self.jnp.reshape(a, shape)
    
    def matmul(self, a: Any, b: Any) -> Any:
        return self.jnp.matmul(a, b)
    
    def inv(self, a: Any) -> Any:
        return self.jnp.linalg.inv(a)
    
    def solve(self, a: Any, b: Any) -> Any:
        return self.jnp.linalg.solve(a, b)
    
    def eigvalsh(self, a: Any) -> Any:
        return self.jnp.linalg.eigvalsh(a)
    
    def det(self, a: Any) -> float:
        return float(self.jnp.linalg.det(a))
    
    def trace(self, a: Any, axis1: int = 0, axis2: int = 1) -> Any:
        return self.jnp.trace(a, axis1=axis1, axis2=axis2)
    
    def sum(self, a: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        return self.jnp.sum(a, axis=axis)
    
    def to_numpy(self, a: Any) -> np.ndarray:
        return np.asarray(a)


class CuPyBackend(Backend):
    """CuPy backend for GPU acceleration."""
    
    def __init__(self) -> None:
        try:
            import cupy as cp
            self.cp = cp
        except ImportError:
            raise ImportError(
                "CuPy backend requires cupy package. Install with: pip install cupy"
            )
    
    @property
    def name(self) -> str:
        return "cupy"
    
    def asarray(self, data: Any, dtype: Optional[Any] = None) -> Any:
        return self.cp.asarray(data, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        return self.cp.zeros(shape, dtype=dtype or self.cp.float64)
    
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        return self.cp.ones(shape, dtype=dtype or self.cp.float64)
    
    def eye(self, n: int, dtype: Any = None) -> Any:
        return self.cp.eye(n, dtype=dtype or self.cp.float64)
    
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return self.cp.einsum(subscripts, *operands)
    
    def tensordot(self, a: Any, b: Any, axes: Union[int, Tuple] = 2) -> Any:
        return self.cp.tensordot(a, b, axes=axes)
    
    def transpose(self, a: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        return self.cp.transpose(a, axes=axes)
    
    def reshape(self, a: Any, shape: Tuple[int, ...]) -> Any:
        return self.cp.reshape(a, shape)
    
    def matmul(self, a: Any, b: Any) -> Any:
        return self.cp.matmul(a, b)
    
    def inv(self, a: Any) -> Any:
        return self.cp.linalg.inv(a)
    
    def solve(self, a: Any, b: Any) -> Any:
        return self.cp.linalg.solve(a, b)
    
    def eigvalsh(self, a: Any) -> Any:
        return self.cp.linalg.eigvalsh(a)
    
    def det(self, a: Any) -> float:
        return float(self.cp.linalg.det(a).get())
    
    def trace(self, a: Any, axis1: int = 0, axis2: int = 1) -> Any:
        return self.cp.trace(a, axis1=axis1, axis2=axis2)
    
    def sum(self, a: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        return self.cp.sum(a, axis=axis)
    
    def to_numpy(self, a: Any) -> np.ndarray:
        return self.cp.asnumpy(a)


# Global backend registry
_BACKENDS = {
    "numpy": NumPyBackend,
    "jax": JAXBackend,
    "cupy": CuPyBackend,
}

_default_backend: Backend = NumPyBackend()


def get_backend(name: str = "numpy") -> Backend:
    """
    Get a backend by name.
    
    Args:
        name: Backend name ('numpy', 'jax', 'cupy')
        
    Returns:
        Backend instance
        
    Raises:
        ValueError: If backend not found
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )
    
    backend_class = _BACKENDS[name]
    return backend_class()


def set_default_backend(name: str) -> None:
    """
    Set the default backend globally.
    
    Args:
        name: Backend name
    """
    global _default_backend
    _default_backend = get_backend(name)


def get_default_backend() -> Backend:
    """Get the current default backend."""
    return _default_backend

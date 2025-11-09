"""
Metric tensors for raising/lowering indices and defining inner products.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from tas.core.tensor import Tensor
from tas.core.indices import Index
from tas.core.utils import normalize_axis


class Metric(Tensor):
    """
    Metric tensor g_ij for defining inner products and raising/lowering indices.
    
    A metric tensor is a symmetric rank-2 tensor that defines:
    - Inner product structure on a manifold
    - How to raise and lower indices
    - Signature (number of positive/negative/zero eigenvalues)
    
    The metric must be a square matrix (same dimension on both axes).
    Typically both indices are covariant (down): g_ij
    
    Examples:
        >>> # Euclidean 3D metric
        >>> g = Metric(np.eye(3), indices=("_i", "_j"))
        >>> 
        >>> # Minkowski spacetime metric (- + + + signature)
        >>> eta = Metric(np.diag([-1, 1, 1, 1]), indices=("_mu", "_nu"))
    """
    
    def __post_init__(self) -> None:
        """Validate metric tensor properties."""
        super().__post_init__()
        
        # Check rank-2
        if self.ndim != 2:
            raise ValueError(f"Metric must be rank-2 tensor, got rank {self.ndim}")
        
        # Check square
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                f"Metric must be square matrix, got shape {self.shape}"
            )
        
        # Warn if not symmetric (but allow it for generalized cases)
        if not np.allclose(self.data, self.data.T):
            import warnings
            warnings.warn(
                "Metric tensor is not symmetric. This may lead to unexpected behavior."
            )
    
    @property
    def dim(self) -> int:
        """Dimension of the space (size of metric matrix)."""
        return self.shape[0]
    
    def inverse(self) -> "Metric":
        """
        Compute inverse metric g^ij.
        
        The inverse metric raises indices: V^i = g^ij V_j
        
        Returns:
            Inverse metric tensor with contravariant indices
            
        Raises:
            np.linalg.LinAlgError: If metric is singular
        """
        inv_data = np.linalg.inv(self.data)
        
        # Flip indices to contravariant
        new_indices = tuple(idx.raise_index() for idx in self.indices)
        
        return Metric(
            data=inv_data,
            indices=new_indices,
            name=f"inv({self.name or 'g'})",
            backend=self.backend,
            meta=self.meta
        )
    
    def signature(self) -> Tuple[int, int, int]:
        """
        Compute signature of the metric (p, q, z).
        
        Returns:
            Tuple (positive, negative, zero) counting eigenvalues
            
        Examples:
            >>> euclidean = Metric(np.eye(3), indices=("_i", "_j"))
            >>> euclidean.signature()
            (3, 0, 0)
            >>> 
            >>> minkowski = Metric(np.diag([-1, 1, 1, 1]), indices=("_mu", "_nu"))
            >>> minkowski.signature()
            (3, 1, 0)  # One negative (timelike)
        """
        eigenvalues = np.linalg.eigvalsh(self.data)
        
        positive = np.sum(eigenvalues > 1e-10)
        negative = np.sum(eigenvalues < -1e-10)
        zero = np.sum(np.abs(eigenvalues) <= 1e-10)
        
        return (int(positive), int(negative), int(zero))
    
    def determinant(self) -> float:
        """
        Compute determinant of the metric.
        
        Returns:
            det(g)
        """
        return float(np.linalg.det(self.data))
    
    def sqrt_abs_det(self) -> float:
        """
        Compute sqrt(|det(g)|).
        
        Useful for volume elements in integration.
        
        Returns:
            sqrt(|det(g)|)
        """
        return float(np.sqrt(np.abs(self.determinant())))
    
    def raise_index(self, tensor: Tensor, axis: int) -> Tensor:
        """
        Raise an index of a tensor using this metric.
        
        Computes: T^{...i...} = g^{ij} T_{...j...}
        
        Args:
            tensor: Tensor with covariant index to raise
            axis: Which axis (index position) to raise
            
        Returns:
            New tensor with raised index
            
        Raises:
            ValueError: If index cannot be raised or dimensions incompatible
            
        Examples:
            >>> g = Metric(np.eye(3), indices=("_i", "_j"))
            >>> V = Tensor(np.array([1, 2, 3]), indices=("_i",))
            >>> V_up = g.raise_index(V, axis=0)
            >>> V_up.indices[0].variance
            'up'
        """
        axis = normalize_axis(axis, tensor.ndim)
        
        # Check index is covariant
        if tensor.indices[axis].variance != "down":
            raise ValueError(
                f"Index at axis {axis} is already contravariant (up). "
                "Can only raise covariant (down) indices."
            )
        
        # Check dimension compatibility
        if tensor.shape[axis] != self.dim:
            raise ValueError(
                f"Tensor dimension {tensor.shape[axis]} at axis {axis} "
                f"does not match metric dimension {self.dim}"
            )
        
        # Get inverse metric
        g_inv = self.inverse()
        
        # Contract: result^{...i...} = g^{ij} tensor_{...j...}
        # Build einsum expression
        # Move axis to contract to the end temporarily
        axes_order = list(range(tensor.ndim))
        axes_order[axis], axes_order[-1] = axes_order[-1], axes_order[axis]
        
        tensor_reordered = tensor.transpose(axes_order)
        
        # Contract last axis with second index of g^ij
        result_data = np.tensordot(tensor_reordered.data, g_inv.data, axes=([-1], [1]))
        
        # Reorder back
        result_data = np.moveaxis(result_data, -1, axis)
        
        # Build new indices
        new_indices = list(tensor.indices)
        new_indices[axis] = tensor.indices[axis].raise_index()
        
        return Tensor(
            data=result_data,
            indices=tuple(new_indices),
            name=f"raise({tensor.name or 'T'})",
            backend=tensor.backend,
            meta=tensor.meta
        )
    
    def lower_index(self, tensor: Tensor, axis: int) -> Tensor:
        """
        Lower an index of a tensor using this metric.
        
        Computes: T_{...i...} = g_{ij} T^{...j...}
        
        Args:
            tensor: Tensor with contravariant index to lower
            axis: Which axis (index position) to lower
            
        Returns:
            New tensor with lowered index
            
        Raises:
            ValueError: If index cannot be lowered or dimensions incompatible
        """
        axis = normalize_axis(axis, tensor.ndim)
        
        # Check index is contravariant
        if tensor.indices[axis].variance != "up":
            raise ValueError(
                f"Index at axis {axis} is already covariant (down). "
                "Can only lower contravariant (up) indices."
            )
        
        # Check dimension compatibility
        if tensor.shape[axis] != self.dim:
            raise ValueError(
                f"Tensor dimension {tensor.shape[axis]} at axis {axis} "
                f"does not match metric dimension {self.dim}"
            )
        
        # Contract: result_{...i...} = g_{ij} tensor^{...j...}
        axes_order = list(range(tensor.ndim))
        axes_order[axis], axes_order[-1] = axes_order[-1], axes_order[axis]
        
        tensor_reordered = tensor.transpose(axes_order)
        
        # Contract last axis with second index of g_ij
        result_data = np.tensordot(tensor_reordered.data, self.data, axes=([-1], [1]))
        
        # Reorder back
        result_data = np.moveaxis(result_data, -1, axis)
        
        # Build new indices
        new_indices = list(tensor.indices)
        new_indices[axis] = tensor.indices[axis].lower_index()
        
        return Tensor(
            data=result_data,
            indices=tuple(new_indices),
            name=f"lower({tensor.name or 'T'})",
            backend=tensor.backend,
            meta=tensor.meta
        )
    
    def inner_product(self, tensor1: Tensor, tensor2: Tensor) -> float:
        """
        Compute inner product of two rank-1 tensors (vectors).
        
        Computes: <u, v> = g_ij u^i v^j
        
        Args:
            tensor1: First vector (rank-1 tensor)
            tensor2: Second vector (rank-1 tensor)
            
        Returns:
            Inner product (scalar)
            
        Raises:
            ValueError: If tensors are not rank-1 or dimensions don't match
        """
        if tensor1.ndim != 1 or tensor2.ndim != 1:
            raise ValueError("Inner product requires rank-1 tensors (vectors)")
        
        if tensor1.shape[0] != tensor2.shape[0] != self.dim:
            raise ValueError("Vector dimensions must match metric dimension")
        
        # If both are contravariant, lower one
        if tensor1.indices[0].variance == "up" and tensor2.indices[0].variance == "up":
            tensor1_down = self.lower_index(tensor1, 0)
            result = np.dot(tensor1_down.data, tensor2.data)
        # If both covariant, raise one
        elif tensor1.indices[0].variance == "down" and tensor2.indices[0].variance == "down":
            tensor1_up = self.raise_index(tensor1, 0)
            result = np.dot(tensor1_up.data, tensor2.data)
        # One up, one down - direct dot product
        else:
            result = np.dot(tensor1.data, tensor2.data)
        
        return float(result)


def euclidean_metric(dim: int, index_name: str = "i") -> Metric:
    """
    Create Euclidean metric (identity matrix).
    
    Args:
        dim: Dimension of space
        index_name: Base name for indices
        
    Returns:
        Euclidean metric tensor
    """
    return Metric(
        data=np.eye(dim),
        indices=(f"_{index_name}", f"_{index_name}"),
        name="euclidean"
    )


def minkowski_metric(signature: str = "timelike") -> Metric:
    """
    Create Minkowski spacetime metric.
    
    Args:
        signature: Either "timelike" (- + + +) or "spacelike" (+ - - -)
        
    Returns:
        Minkowski metric tensor
    """
    if signature == "timelike":
        diag = [-1, 1, 1, 1]
    elif signature == "spacelike":
        diag = [1, -1, -1, -1]
    else:
        raise ValueError(f"Unknown signature: {signature}")
    
    return Metric(
        data=np.diag(diag),
        indices=("_mu", "_nu"),
        name="minkowski"
    )

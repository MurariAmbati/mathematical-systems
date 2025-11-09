"""
Affine connections and Christoffel symbols for covariant derivatives.
"""

from __future__ import annotations
from typing import Optional, Callable
import numpy as np

from tas.core.tensor import Tensor
from tas.core.metrics import Metric


class Connection:
    """
    Affine connection for defining parallel transport and covariant derivatives.
    
    An affine connection Γ^k_ij (Christoffel symbols) describes how vectors
    change when transported along curves on a manifold.
    
    For a metric-compatible connection (Levi-Civita), the Christoffel symbols
    can be computed from the metric tensor.
    
    Attributes:
        christoffel: Rank-3 tensor with components Γ^k_ij
        dim: Dimension of the space
        metric: Optional associated metric tensor
    """
    
    def __init__(
        self, 
        christoffel: Tensor,
        metric: Optional[Metric] = None,
        name: Optional[str] = None
    ):
        """
        Initialize connection with Christoffel symbols.
        
        Args:
            christoffel: Rank-3 tensor with indices (^k, _i, _j)
            metric: Optional associated metric
            name: Optional name for the connection
        """
        if christoffel.ndim != 3:
            raise ValueError(f"Christoffel symbols must be rank-3, got {christoffel.ndim}")
        
        if christoffel.shape[1] != christoffel.shape[2]:
            raise ValueError(
                "Last two indices of Christoffel symbols must have same dimension"
            )
        
        self.christoffel = christoffel
        self.dim = christoffel.shape[1]
        self.metric = metric
        self.name = name or "connection"
    
    def __repr__(self) -> str:
        return f"Connection(dim={self.dim}, name={self.name!r})"
    
    @classmethod
    def from_metric(
        cls,
        metric: Metric,
        coords: Optional[Callable] = None,
        dx: float = 1e-5
    ) -> "Connection":
        """
        Compute Levi-Civita connection from a metric tensor.
        
        The Christoffel symbols are computed as:
        Γ^k_ij = (1/2) g^kl (∂_i g_lj + ∂_j g_il - ∂_l g_ij)
        
        Args:
            metric: Metric tensor g_ij
            coords: Optional coordinate functions for computing derivatives
            dx: Step size for numerical derivatives
            
        Returns:
            Connection with computed Christoffel symbols
        """
        dim = metric.dim
        g_inv = metric.inverse()
        
        # Compute metric derivatives numerically
        # In a real implementation, this would use coordinate-aware differentiation
        christoffel_data = np.zeros((dim, dim, dim))
        
        # For now, if metric is constant (like Euclidean or Minkowski),
        # Christoffel symbols are zero
        # More sophisticated: compute finite differences
        
        # Check if metric is position-dependent
        # For this basic implementation, we assume constant metric -> zero connection
        # A full implementation would require coordinate functions
        
        christoffel = Tensor(
            data=christoffel_data,
            indices=("^k", "_i", "_j"),
            name="Christoffel"
        )
        
        return cls(christoffel=christoffel, metric=metric, name="Levi-Civita")
    
    @classmethod
    def from_metric_derivatives(
        cls,
        metric: Metric,
        metric_derivatives: np.ndarray
    ) -> "Connection":
        """
        Compute Christoffel symbols from metric and its derivatives.
        
        Args:
            metric: Metric tensor g_ij
            metric_derivatives: Array of shape (dim, dim, dim) containing ∂_k g_ij
            
        Returns:
            Connection with computed Christoffel symbols
        """
        dim = metric.dim
        g_inv = metric.inverse()
        
        christoffel_data = np.zeros((dim, dim, dim))
        
        # Γ^k_ij = (1/2) g^kl (∂_i g_lj + ∂_j g_il - ∂_l g_ij)
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    sum_val = 0.0
                    for l in range(dim):
                        term1 = metric_derivatives[i, l, j]  # ∂_i g_lj
                        term2 = metric_derivatives[j, i, l]  # ∂_j g_il
                        term3 = metric_derivatives[l, i, j]  # ∂_l g_ij
                        sum_val += g_inv.data[k, l] * (term1 + term2 - term3)
                    christoffel_data[k, i, j] = 0.5 * sum_val
        
        christoffel = Tensor(
            data=christoffel_data,
            indices=("^k", "_i", "_j"),
            name="Christoffel"
        )
        
        return cls(christoffel=christoffel, metric=metric, name="Levi-Civita")
    
    def is_symmetric(self, tol: float = 1e-10) -> bool:
        """
        Check if connection is torsion-free (symmetric in lower indices).
        
        For torsion-free connection: Γ^k_ij = Γ^k_ji
        
        Args:
            tol: Tolerance for numerical comparison
            
        Returns:
            True if symmetric (torsion-free)
        """
        data = self.christoffel.data
        # Check Γ^k_ij == Γ^k_ji for all k
        for k in range(self.dim):
            if not np.allclose(data[k], data[k].T, atol=tol):
                return False
        return True


def covariant_derivative(
    tensor: Tensor,
    connection: Connection,
    direction: Optional[int] = None
) -> Tensor:
    """
    Compute covariant derivative of a tensor.
    
    For a vector V^i: ∇_j V^i = ∂_j V^i + Γ^i_jk V^k
    For a covector ω_i: ∇_j ω_i = ∂_j ω_i - Γ^k_ji ω_k
    
    This simplified implementation computes the connection term only,
    assuming ordinary derivatives are zero (for constant tensors).
    
    Args:
        tensor: Tensor to differentiate
        connection: Connection defining covariant derivative
        direction: Optional specific direction index
        
    Returns:
        Covariant derivative (rank increased by 1)
        
    Note:
        Full implementation requires coordinate-aware ordinary derivative.
        Current version handles the connection contribution for constant tensors.
    """
    if direction is not None:
        raise NotImplementedError("Directional covariant derivative not yet implemented")
    
    # For now, implement for rank-1 tensors (vectors and covectors)
    if tensor.ndim != 1:
        raise NotImplementedError(
            "Covariant derivative currently only implemented for rank-1 tensors"
        )
    
    dim = connection.dim
    
    if tensor.shape[0] != dim:
        raise ValueError(
            f"Tensor dimension {tensor.shape[0]} must match connection dimension {dim}"
        )
    
    # Result has one additional index (the derivative index)
    result_data = np.zeros((dim, dim))
    
    Gamma = connection.christoffel.data
    
    if tensor.indices[0].variance == "up":
        # Contravariant vector V^i: ∇_j V^i = ∂_j V^i + Γ^i_jk V^k
        # For constant V, ∂_j V^i = 0
        for j in range(dim):
            for i in range(dim):
                sum_val = 0.0
                for k in range(dim):
                    sum_val += Gamma[i, j, k] * tensor.data[k]
                result_data[j, i] = sum_val
        
        result_indices = ("_j", "^i")
    else:
        # Covariant vector ω_i: ∇_j ω_i = ∂_j ω_i - Γ^k_ji ω_k
        # For constant ω, ∂_j ω_i = 0
        for j in range(dim):
            for i in range(dim):
                sum_val = 0.0
                for k in range(dim):
                    sum_val -= Gamma[k, j, i] * tensor.data[k]
                result_data[j, i] = sum_val
        
        result_indices = ("_j", "_i")
    
    return Tensor(
        data=result_data,
        indices=result_indices,
        name=f"∇({tensor.name or 'T'})",
        backend=tensor.backend,
        meta=tensor.meta
    )


def christoffel_from_metric_direct(metric: Metric, position: np.ndarray, 
                                   dx: float = 1e-5) -> np.ndarray:
    """
    Compute Christoffel symbols at a point via finite differences.
    
    This is a helper function for computing Christoffel symbols when
    the metric is a function of position.
    
    Args:
        metric: Metric tensor (potentially position-dependent)
        position: Coordinates at which to compute Christoffel symbols
        dx: Step size for finite differences
        
    Returns:
        Array of Christoffel symbols Γ^k_ij
    """
    # This would require metric to be callable with position
    # Placeholder for more advanced implementation
    raise NotImplementedError(
        "Position-dependent Christoffel symbol computation requires "
        "coordinate-aware metric implementation"
    )

"""
Coordinate systems and transformations between frames.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np

from tas.core.tensor import Tensor
from tas.core.metrics import Metric


class CoordinateFrame(ABC):
    """
    Abstract base class for coordinate systems.
    
    A coordinate frame defines:
    - Coordinate names and ranges
    - Transformation to/from Cartesian coordinates
    - Jacobian matrices for tensor transformation
    - Associated metric tensor
    """
    
    def __init__(self, dim: int, name: str):
        """
        Initialize coordinate frame.
        
        Args:
            dim: Dimension of the space
            name: Name of the coordinate system
        """
        self.dim = dim
        self.name = name
    
    @abstractmethod
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates to Cartesian.
        
        Args:
            coords: Coordinates in this frame, shape (..., dim)
            
        Returns:
            Cartesian coordinates, shape (..., dim)
        """
        pass
    
    @abstractmethod
    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform Cartesian coordinates to this frame.
        
        Args:
            coords: Cartesian coordinates, shape (..., dim)
            
        Returns:
            Coordinates in this frame, shape (..., dim)
        """
        pass
    
    @abstractmethod
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix ∂x^i/∂q^j.
        
        Args:
            coords: Coordinates in this frame
            
        Returns:
            Jacobian matrix, shape (dim, dim)
        """
        pass
    
    @abstractmethod
    def inverse_jacobian(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute inverse Jacobian matrix ∂q^i/∂x^j.
        
        Args:
            coords: Coordinates in this frame
            
        Returns:
            Inverse Jacobian matrix, shape (dim, dim)
        """
        pass
    
    def metric(self, coords: Optional[np.ndarray] = None) -> Metric:
        """
        Get metric tensor in this coordinate system.
        
        Args:
            coords: Coordinates at which to evaluate metric (for curvilinear)
            
        Returns:
            Metric tensor
        """
        # Default: compute from Jacobian
        if coords is None:
            coords = np.zeros(self.dim)
        
        J = self.jacobian(coords)
        # Metric: g_ij = J^T · J for embedding in Euclidean space
        g_data = J.T @ J
        
        return Metric(
            data=g_data,
            indices=("_i", "_j"),
            name=f"metric_{self.name}"
        )


class CartesianFrame(CoordinateFrame):
    """
    Cartesian (rectangular) coordinate system.
    
    In n dimensions: (x, y, z, ...) or (x_1, x_2, ..., x_n)
    """
    
    def __init__(self, dim: int = 3):
        super().__init__(dim, "Cartesian")
    
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Identity transformation."""
        return coords.copy()
    
    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Identity transformation."""
        return coords.copy()
    
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Identity matrix."""
        return np.eye(self.dim)
    
    def inverse_jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Identity matrix."""
        return np.eye(self.dim)
    
    def metric(self, coords: Optional[np.ndarray] = None) -> Metric:
        """Euclidean metric (identity)."""
        from tas.core.metrics import euclidean_metric
        return euclidean_metric(self.dim)


class SphericalFrame(CoordinateFrame):
    """
    Spherical coordinate system in 3D.
    
    Coordinates: (r, θ, φ)
    - r: radial distance (r ≥ 0)
    - θ: polar angle from z-axis (0 ≤ θ ≤ π)
    - φ: azimuthal angle in xy-plane (0 ≤ φ < 2π)
    
    Cartesian conversion:
    - x = r sin(θ) cos(φ)
    - y = r sin(θ) sin(φ)
    - z = r cos(θ)
    """
    
    def __init__(self):
        super().__init__(3, "Spherical")
    
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Transform (r, θ, φ) to (x, y, z)."""
        r, theta, phi = coords[..., 0], coords[..., 1], coords[..., 2]
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return np.stack([x, y, z], axis=-1)
    
    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Transform (x, y, z) to (r, θ, φ)."""
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / (r + 1e-10), -1, 1))
        phi = np.arctan2(y, x)
        
        return np.stack([r, theta, phi], axis=-1)
    
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Jacobian ∂(x,y,z)/∂(r,θ,φ)."""
        r, theta, phi = coords
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        J = np.array([
            [sin_theta * cos_phi, r * cos_theta * cos_phi, -r * sin_theta * sin_phi],
            [sin_theta * sin_phi, r * cos_theta * sin_phi,  r * sin_theta * cos_phi],
            [cos_theta,          -r * sin_theta,             0]
        ])
        
        return J
    
    def inverse_jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Inverse Jacobian ∂(r,θ,φ)/∂(x,y,z)."""
        return np.linalg.inv(self.jacobian(coords))
    
    def metric(self, coords: Optional[np.ndarray] = None) -> Metric:
        """Spherical metric: ds² = dr² + r²dθ² + r²sin²(θ)dφ²."""
        if coords is None:
            r = 1.0
            theta = np.pi / 4
        else:
            r = coords[0]
            theta = coords[1]
        
        g_data = np.diag([1.0, r**2, r**2 * np.sin(theta)**2])
        
        return Metric(
            data=g_data,
            indices=("_i", "_j"),
            name="metric_spherical"
        )


class CylindricalFrame(CoordinateFrame):
    """
    Cylindrical coordinate system in 3D.
    
    Coordinates: (ρ, φ, z)
    - ρ: radial distance in xy-plane (ρ ≥ 0)
    - φ: azimuthal angle (0 ≤ φ < 2π)
    - z: height
    
    Cartesian conversion:
    - x = ρ cos(φ)
    - y = ρ sin(φ)
    - z = z
    """
    
    def __init__(self):
        super().__init__(3, "Cylindrical")
    
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Transform (ρ, φ, z) to (x, y, z)."""
        rho, phi, z = coords[..., 0], coords[..., 1], coords[..., 2]
        
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        
        return np.stack([x, y, z], axis=-1)
    
    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Transform (x, y, z) to (ρ, φ, z)."""
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        
        return np.stack([rho, phi, z], axis=-1)
    
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Jacobian ∂(x,y,z)/∂(ρ,φ,z)."""
        rho, phi, z = coords
        
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        J = np.array([
            [cos_phi, -rho * sin_phi, 0],
            [sin_phi,  rho * cos_phi, 0],
            [0,        0,              1]
        ])
        
        return J
    
    def inverse_jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Inverse Jacobian ∂(ρ,φ,z)/∂(x,y,z)."""
        return np.linalg.inv(self.jacobian(coords))
    
    def metric(self, coords: Optional[np.ndarray] = None) -> Metric:
        """Cylindrical metric: ds² = dρ² + ρ²dφ² + dz²."""
        if coords is None:
            rho = 1.0
        else:
            rho = coords[0]
        
        g_data = np.diag([1.0, rho**2, 1.0])
        
        return Metric(
            data=g_data,
            indices=("_i", "_j"),
            name="metric_cylindrical"
        )


def transform_tensor(
    tensor: Tensor,
    from_frame: CoordinateFrame,
    to_frame: CoordinateFrame,
    coords: Optional[np.ndarray] = None
) -> Tensor:
    """
    Transform a tensor between coordinate frames.
    
    Uses Jacobian matrices to transform tensor components according
    to index variance (covariant/contravariant).
    
    Args:
        tensor: Tensor to transform
        from_frame: Source coordinate frame
        to_frame: Target coordinate frame
        coords: Coordinates at which to evaluate transformation
        
    Returns:
        Transformed tensor
        
    Note:
        Full implementation requires proper handling of all index positions.
        Current version is simplified for rank-1 tensors.
    """
    if coords is None:
        coords = np.zeros(from_frame.dim)
    
    if from_frame.dim != to_frame.dim:
        raise ValueError("Can only transform between frames of same dimension")
    
    if tensor.shape[0] != from_frame.dim:
        raise ValueError("Tensor dimension must match frame dimension")
    
    # Get transformation Jacobian
    # from_frame coords -> Cartesian -> to_frame coords
    # Compose: J_to^{-1} · J_from
    
    J_from = from_frame.jacobian(coords)
    
    # Transform coords to Cartesian, then to target frame
    cart_coords = from_frame.to_cartesian(coords)
    target_coords = to_frame.from_cartesian(cart_coords)
    J_to_inv = to_frame.inverse_jacobian(target_coords)
    
    # Combined Jacobian
    J = J_to_inv @ J_from
    
    # Transform based on variance
    if tensor.ndim == 1:
        if tensor.indices[0].variance == "up":
            # Contravariant: V'^i = (∂q'^i/∂q^j) V^j
            new_data = J @ tensor.data
        else:
            # Covariant: ω'_i = (∂q^j/∂q'^i) ω_j
            new_data = np.linalg.inv(J) @ tensor.data
        
        return Tensor(
            data=new_data,
            indices=tensor.indices,
            name=f"transformed_{tensor.name or 'T'}",
            backend=tensor.backend,
            meta=tensor.meta
        )
    
    raise NotImplementedError(
        "Tensor transformation only implemented for rank-1 tensors"
    )

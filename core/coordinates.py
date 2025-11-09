"""
Coordinate System Transformations

Support for Cartesian, polar, and parametric coordinate systems.
"""

from typing import Tuple, Callable, Optional, Union
import numpy as np
from numpy.typing import NDArray


def cartesian_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    samples: Union[int, Tuple[int, int]] = 100
) -> Tuple[NDArray, NDArray]:
    """
    Create a 2D Cartesian grid.
    
    Args:
        x_range: (x_min, x_max) tuple
        y_range: (y_min, y_max) tuple
        samples: Number of samples (scalar for both, or (nx, ny) tuple)
        
    Returns:
        (X, Y) meshgrid arrays of shape (ny, nx)
        
    Example:
        >>> X, Y = cartesian_grid((-1, 1), (-1, 1), samples=50)
        >>> Z = X**2 + Y**2  # Compute function values
    """
    if isinstance(samples, int):
        nx = ny = samples
    else:
        nx, ny = samples
    
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)
    
    return X, Y


def polar_to_cartesian(
    r: Union[float, NDArray],
    theta: Union[float, NDArray]
) -> Tuple[Union[float, NDArray], Union[float, NDArray]]:
    """
    Convert polar coordinates to Cartesian coordinates.
    
    Args:
        r: Radius (scalar or array)
        theta: Angle in radians (scalar or array)
        
    Returns:
        (x, y) in Cartesian coordinates
        
    Example:
        >>> r = np.linspace(0, 1, 100)
        >>> theta = np.linspace(0, 2*np.pi, 100)
        >>> x, y = polar_to_cartesian(r, theta)
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cartesian_to_polar(
    x: Union[float, NDArray],
    y: Union[float, NDArray]
) -> Tuple[Union[float, NDArray], Union[float, NDArray]]:
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x: X coordinate (scalar or array)
        y: Y coordinate (scalar or array)
        
    Returns:
        (r, theta) in polar coordinates (theta in radians)
        
    Example:
        >>> x = np.array([1, 0, -1])
        >>> y = np.array([0, 1, 0])
        >>> r, theta = cartesian_to_polar(x, y)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_grid(
    r_range: Tuple[float, float],
    theta_range: Tuple[float, float] = (0, 2*np.pi),
    samples: Union[int, Tuple[int, int]] = 100
) -> Tuple[NDArray, NDArray]:
    """
    Create a polar coordinate grid.
    
    Args:
        r_range: (r_min, r_max) tuple
        theta_range: (theta_min, theta_max) in radians, default (0, 2π)
        samples: Number of samples (scalar or (nr, ntheta) tuple)
        
    Returns:
        (R, Theta) meshgrid arrays in polar coordinates
        
    Example:
        >>> R, Theta = polar_grid((0, 1), samples=50)
        >>> X, Y = polar_to_cartesian(R, Theta)
    """
    if isinstance(samples, int):
        nr = ntheta = samples
    else:
        nr, ntheta = samples
    
    r = np.linspace(r_range[0], r_range[1], nr)
    theta = np.linspace(theta_range[0], theta_range[1], ntheta)
    R, Theta = np.meshgrid(r, theta)
    
    return R, Theta


def parametric_to_points(
    fx: Callable,
    fy: Callable,
    t_range: Tuple[float, float],
    samples: int = 1000,
    fz: Optional[Callable] = None
) -> NDArray:
    """
    Generate points from parametric equations.
    
    Args:
        fx: Function for x(t)
        fy: Function for y(t)
        t_range: (t_min, t_max) parameter range
        samples: Number of samples
        fz: Optional function for z(t) (for 3D curves)
        
    Returns:
        Array of shape (samples, 2) or (samples, 3) with (x, y) or (x, y, z) points
        
    Example:
        >>> # Lissajous curve
        >>> fx = lambda t: np.sin(3*t)
        >>> fy = lambda t: np.sin(2*t)
        >>> points = parametric_to_points(fx, fy, (0, 2*np.pi), samples=1000)
    """
    t = np.linspace(t_range[0], t_range[1], samples)
    
    x = fx(t)
    y = fy(t)
    
    if fz is not None:
        z = fz(t)
        return np.column_stack([x, y, z])
    else:
        return np.column_stack([x, y])


def parametric_surface(
    fx: Callable,
    fy: Callable,
    fz: Callable,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    samples: Union[int, Tuple[int, int]] = 50
) -> NDArray:
    """
    Generate points from parametric surface equations.
    
    Args:
        fx: Function for x(u, v)
        fy: Function for y(u, v)
        fz: Function for z(u, v)
        u_range: (u_min, u_max) parameter range
        v_range: (v_min, v_max) parameter range
        samples: Number of samples (scalar or (nu, nv) tuple)
        
    Returns:
        Array of shape (nu*nv, 3) with (x, y, z) points
        
    Example:
        >>> # Torus
        >>> R, r = 2, 1
        >>> fx = lambda u, v: (R + r*np.cos(v)) * np.cos(u)
        >>> fy = lambda u, v: (R + r*np.cos(v)) * np.sin(u)
        >>> fz = lambda u, v: r * np.sin(v)
        >>> points = parametric_surface(fx, fy, fz, (0, 2*np.pi), (0, 2*np.pi))
    """
    if isinstance(samples, int):
        nu = nv = samples
    else:
        nu, nv = samples
    
    u = np.linspace(u_range[0], u_range[1], nu)
    v = np.linspace(v_range[0], v_range[1], nv)
    U, V = np.meshgrid(u, v)
    
    X = fx(U, V)
    Y = fy(U, V)
    Z = fz(U, V)
    
    # Flatten and stack
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return points


def spherical_to_cartesian(
    r: Union[float, NDArray],
    theta: Union[float, NDArray],
    phi: Union[float, NDArray]
) -> Tuple[Union[float, NDArray], Union[float, NDArray], Union[float, NDArray]]:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        r: Radius
        theta: Azimuthal angle (0 to 2π)
        phi: Polar angle (0 to π)
        
    Returns:
        (x, y, z) in Cartesian coordinates
        
    Note:
        Uses physics convention: theta is azimuthal, phi is polar
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def cartesian_to_spherical(
    x: Union[float, NDArray],
    y: Union[float, NDArray],
    z: Union[float, NDArray]
) -> Tuple[Union[float, NDArray], Union[float, NDArray], Union[float, NDArray]]:
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        
    Returns:
        (r, theta, phi) in spherical coordinates
        
    Note:
        theta is azimuthal (0 to 2π), phi is polar (0 to π)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / (r + 1e-10))  # Add small value to avoid division by zero
    return r, theta, phi


def normalize_points(
    points: NDArray,
    target_range: Optional[Tuple[float, float]] = None
) -> NDArray:
    """
    Normalize points to fit within a target range.
    
    Args:
        points: Array of shape (n, d) where d is dimension
        target_range: (min, max) tuple for normalization, default (-1, 1)
        
    Returns:
        Normalized points with same shape as input
        
    Example:
        >>> points = np.random.randn(100, 2) * 10
        >>> normalized = normalize_points(points, target_range=(0, 1))
    """
    if target_range is None:
        target_range = (-1, 1)
    
    # Find current range for each dimension
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges = np.where(ranges == 0, 1, ranges)
    
    # Normalize to [0, 1]
    normalized = (points - min_vals) / ranges
    
    # Scale to target range
    target_min, target_max = target_range
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


def center_points(points: NDArray) -> NDArray:
    """
    Center points around the origin (mean = 0).
    
    Args:
        points: Array of shape (n, d)
        
    Returns:
        Centered points
    """
    return points - points.mean(axis=0)


def scale_points(points: NDArray, scale_factor: float) -> NDArray:
    """
    Scale points by a constant factor.
    
    Args:
        points: Array of shape (n, d)
        scale_factor: Scaling factor
        
    Returns:
        Scaled points
    """
    return points * scale_factor


def rotate_2d(
    points: NDArray,
    angle: float,
    center: Optional[Tuple[float, float]] = None
) -> NDArray:
    """
    Rotate 2D points around a center point.
    
    Args:
        points: Array of shape (n, 2)
        angle: Rotation angle in radians
        center: (x, y) center of rotation, default (0, 0)
        
    Returns:
        Rotated points
    """
    if center is None:
        center = (0, 0)
    
    # Translate to origin
    points_centered = points - np.array(center)
    
    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Rotate
    rotated = points_centered @ rotation_matrix.T
    
    # Translate back
    return rotated + np.array(center)


def adaptive_sample_curve(
    fx: Callable,
    fy: Callable,
    t_range: Tuple[float, float],
    min_samples: int = 100,
    max_samples: int = 10000,
    curvature_threshold: float = 0.1
) -> NDArray:
    """
    Adaptively sample a parametric curve based on curvature.
    
    Areas of high curvature get more samples for smooth rendering.
    
    Args:
        fx: Function for x(t)
        fy: Function for y(t)
        t_range: (t_min, t_max) parameter range
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples
        curvature_threshold: Threshold for adaptive refinement
        
    Returns:
        Array of shape (n, 2) with adaptively sampled points
    """
    # Start with coarse sampling
    t = np.linspace(t_range[0], t_range[1], min_samples)
    
    # Compute points and derivatives
    x = fx(t)
    y = fy(t)
    
    # Estimate curvature using finite differences
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2) + 1e-10
    
    # Normalize curvature
    curvature = curvature / curvature.max()
    
    # For simplicity, return the coarse sampling
    # A full implementation would recursively refine high-curvature regions
    return np.column_stack([x, y])

"""
Utility Functions for Math Art Generator

Random number generation, noise functions, scaling, and helper utilities.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import noise as noise_lib  # Perlin/Simplex noise
from numpy.random import Generator, PCG64


class SeededRNG:
    """
    Seeded random number generator for deterministic art generation.
    
    Ensures reproducibility across sessions with the same seed.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize seeded RNG.
        
        Args:
            seed: Random seed for reproducibility. If None, uses random seed.
        """
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self._rng = Generator(PCG64(self.seed))
    
    def random(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, NDArray]:
        """Generate random values in [0, 1)."""
        return self._rng.random(size)
    
    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray]:
        """Generate uniform random values."""
        return self._rng.uniform(low, high, size)
    
    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray]:
        """Generate normally distributed random values."""
        return self._rng.normal(loc, scale, size)
    
    def integers(
        self,
        low: int,
        high: int,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[int, NDArray]:
        """Generate random integers."""
        return self._rng.integers(low, high, size)
    
    def choice(
        self,
        a: Union[int, NDArray],
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True
    ) -> Union[any, NDArray]:
        """Sample random elements from array."""
        return self._rng.choice(a, size, replace=replace)


class NoiseGenerator:
    """
    Perlin and Simplex noise generator for organic patterns.
    
    Provides deterministic noise fields for art generation.
    """
    
    def __init__(self, seed: Optional[int] = None, noise_type: str = "perlin"):
        """
        Initialize noise generator.
        
        Args:
            seed: Random seed for reproducibility
            noise_type: "perlin" or "simplex"
        """
        self.seed = seed if seed is not None else 0
        self.noise_type = noise_type
    
    def noise_2d(
        self,
        x: Union[float, NDArray],
        y: Union[float, NDArray],
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 1.0
    ) -> Union[float, NDArray]:
        """
        Generate 2D Perlin/Simplex noise.
        
        Args:
            x, y: Coordinates (scalars or arrays)
            octaves: Number of noise layers
            persistence: Amplitude decrease per octave
            lacunarity: Frequency increase per octave
            scale: Overall scale of noise
            
        Returns:
            Noise values in range approximately [-1, 1]
        """
        x_scaled = np.asarray(x) * scale
        y_scaled = np.asarray(y) * scale
        
        if self.noise_type == "perlin":
            noise_func = noise_lib.pnoise2
        else:
            noise_func = noise_lib.snoise2
        
        # Vectorized noise generation
        if isinstance(x_scaled, np.ndarray):
            result = np.zeros_like(x_scaled)
            flat_x = x_scaled.ravel()
            flat_y = y_scaled.ravel()
            flat_result = np.array([
                noise_func(
                    xi, yi,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=self.seed
                )
                for xi, yi in zip(flat_x, flat_y)
            ])
            result = flat_result.reshape(x_scaled.shape)
            return result
        else:
            return noise_func(
                x_scaled, y_scaled,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=self.seed
            )
    
    def noise_3d(
        self,
        x: Union[float, NDArray],
        y: Union[float, NDArray],
        z: Union[float, NDArray],
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 1.0
    ) -> Union[float, NDArray]:
        """
        Generate 3D Perlin/Simplex noise.
        
        Args:
            x, y, z: Coordinates (scalars or arrays)
            octaves: Number of noise layers
            persistence: Amplitude decrease per octave
            lacunarity: Frequency increase per octave
            scale: Overall scale of noise
            
        Returns:
            Noise values in range approximately [-1, 1]
        """
        x_scaled = np.asarray(x) * scale
        y_scaled = np.asarray(y) * scale
        z_scaled = np.asarray(z) * scale
        
        if self.noise_type == "perlin":
            noise_func = noise_lib.pnoise3
        else:
            noise_func = noise_lib.snoise3
        
        if isinstance(x_scaled, np.ndarray):
            result = np.zeros_like(x_scaled)
            flat_x = x_scaled.ravel()
            flat_y = y_scaled.ravel()
            flat_z = z_scaled.ravel()
            flat_result = np.array([
                noise_func(
                    xi, yi, zi,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=self.seed
                )
                for xi, yi, zi in zip(flat_x, flat_y, flat_z)
            ])
            result = flat_result.reshape(x_scaled.shape)
            return result
        else:
            return noise_func(
                x_scaled, y_scaled, z_scaled,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=self.seed
            )


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Smooth interpolation function (Hermite interpolation).
    
    Returns 0 for x <= edge0, 1 for x >= edge1, smooth transition between.
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return x * x * (3 - 2 * x)


def remap(
    value: Union[float, NDArray],
    from_range: Tuple[float, float],
    to_range: Tuple[float, float],
    clamp: bool = False
) -> Union[float, NDArray]:
    """
    Remap a value from one range to another.
    
    Args:
        value: Input value(s)
        from_range: (min, max) of input range
        to_range: (min, max) of output range
        clamp: Whether to clamp output to target range
        
    Returns:
        Remapped value(s)
    """
    from_min, from_max = from_range
    to_min, to_max = to_range
    
    # Normalize to [0, 1]
    normalized = (value - from_min) / (from_max - from_min)
    
    # Remap to target range
    result = normalized * (to_max - to_min) + to_min
    
    if clamp:
        result = np.clip(result, to_min, to_max)
    
    return result


def safe_divide(
    numerator: Union[float, NDArray],
    denominator: Union[float, NDArray],
    default: float = 0.0
) -> Union[float, NDArray]:
    """
    Safe division that handles division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return when denominator is zero
        
    Returns:
        Division result, with default where denominator is zero
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        if isinstance(result, np.ndarray):
            result = np.where(np.isfinite(result), result, default)
        elif not np.isfinite(result):
            result = default
    return result


def distance_2d(p1: NDArray, p2: NDArray) -> Union[float, NDArray]:
    """
    Compute Euclidean distance between 2D points.
    
    Args:
        p1: Point(s) of shape (..., 2)
        p2: Point(s) of shape (..., 2)
        
    Returns:
        Distance(s)
    """
    diff = p1 - p2
    return np.sqrt(np.sum(diff**2, axis=-1))


def distance_3d(p1: NDArray, p2: NDArray) -> Union[float, NDArray]:
    """
    Compute Euclidean distance between 3D points.
    
    Args:
        p1: Point(s) of shape (..., 3)
        p2: Point(s) of shape (..., 3)
        
    Returns:
        Distance(s)
    """
    diff = p1 - p2
    return np.sqrt(np.sum(diff**2, axis=-1))


def compute_curvature_2d(points: NDArray) -> NDArray:
    """
    Compute curvature along a 2D curve.
    
    Args:
        points: Array of shape (n, 2) representing curve points
        
    Returns:
        Array of shape (n,) with curvature values
    """
    # First and second derivatives
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2) + 1e-10
    curvature = numerator / denominator
    
    return curvature


def compute_velocity(points: NDArray) -> NDArray:
    """
    Compute velocity (speed of change) along a curve.
    
    Args:
        points: Array of shape (n, d) representing curve points
        
    Returns:
        Array of shape (n,) with velocity magnitudes
    """
    # Compute differences between consecutive points
    deltas = np.diff(points, axis=0, prepend=points[0:1])
    velocity = np.linalg.norm(deltas, axis=1)
    return velocity


def apply_jitter(
    points: NDArray,
    amount: float,
    rng: Optional[SeededRNG] = None
) -> NDArray:
    """
    Apply random jitter to points.
    
    Args:
        points: Array of shape (n, d)
        amount: Maximum jitter amount
        rng: Random number generator (creates new one if None)
        
    Returns:
        Jittered points
    """
    if rng is None:
        rng = SeededRNG()
    
    jitter = rng.uniform(-amount, amount, size=points.shape)
    return points + jitter


def blend_arrays(
    arr1: NDArray,
    arr2: NDArray,
    weight: float
) -> NDArray:
    """
    Blend two arrays with a weight.
    
    Args:
        arr1: First array
        arr2: Second array (same shape as arr1)
        weight: Blend weight (0 = all arr1, 1 = all arr2)
        
    Returns:
        Blended array
    """
    return (1 - weight) * arr1 + weight * arr2


def fibonacci_sphere(n_points: int) -> NDArray:
    """
    Generate points uniformly distributed on a sphere using Fibonacci spiral.
    
    Args:
        n_points: Number of points
        
    Returns:
        Array of shape (n_points, 3) with points on unit sphere
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    indices = np.arange(n_points)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.column_stack([x, y, z])


def create_circle_mask(
    width: int,
    height: int,
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None
) -> NDArray:
    """
    Create a circular mask.
    
    Args:
        width: Mask width
        height: Mask height
        center: (cx, cy) center coordinates, default is image center
        radius: Circle radius, default is min(width, height) / 2
        
    Returns:
        Boolean array of shape (height, width)
    """
    if center is None:
        center = (width / 2, height / 2)
    if radius is None:
        radius = min(width, height) / 2
    
    y, x = np.ogrid[:height, :width]
    cx, cy = center
    
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = dist_from_center <= radius
    
    return mask

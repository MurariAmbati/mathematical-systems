"""
Function evaluation engine for sampling mathematical functions.

Provides uniform and adaptive sampling strategies for converting mathematical
functions into discrete time series suitable for audio synthesis.
"""

from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
from scipy.interpolate import interp1d


class FunctionEngine:
    """
    Core engine for evaluating and sampling mathematical functions.
    
    Supports:
    - Uniform sampling (fixed time step)
    - Adaptive sampling (based on function curvature)
    - Multiple sampling strategies
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the function engine.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 44100)
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate  # Time step
        
    def uniform_sample(
        self,
        func: Callable,
        duration: float,
        *args: Any,
        normalize: bool = True,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Sample a function uniformly over time.
        
        Args:
            func: Callable function to sample (should accept time as first argument)
            duration: Duration in seconds
            *args: Additional positional arguments to pass to func
            normalize: Whether to normalize output to [-1, 1]
            **kwargs: Additional keyword arguments to pass to func
            
        Returns:
            NumPy array of sampled values
        """
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Evaluate function
        try:
            values = func(t, *args, **kwargs)
        except Exception as e:
            raise ValueError(f"Function evaluation failed: {str(e)}")
        
        # Ensure proper array shape
        values = np.asarray(values, dtype=np.float64)
        if values.shape != t.shape:
            raise ValueError(f"Function output shape {values.shape} doesn't match time shape {t.shape}")
        
        # Normalize if requested
        if normalize:
            values = self._normalize(values)
            
        return values
    
    def adaptive_sample(
        self,
        func: Callable,
        duration: float,
        *args: Any,
        tolerance: float = 0.01,
        min_samples: Optional[int] = None,
        max_samples: Optional[int] = None,
        normalize: bool = True,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Sample a function adaptively based on curvature.
        
        Uses more samples where the function changes rapidly and fewer where it's smooth.
        
        Args:
            func: Callable function to sample
            duration: Duration in seconds
            *args: Additional positional arguments to pass to func
            tolerance: Error tolerance for adaptive sampling (smaller = more samples)
            min_samples: Minimum number of samples
            max_samples: Maximum number of samples
            normalize: Whether to normalize output to [-1, 1]
            **kwargs: Additional keyword arguments to pass to func
            
        Returns:
            NumPy array of adaptively sampled and interpolated values
        """
        # Set default sample bounds
        if min_samples is None:
            min_samples = int(duration * self.sample_rate * 0.1)  # 10% of uniform
        if max_samples is None:
            max_samples = int(duration * self.sample_rate * 2)  # 2x uniform
            
        # Start with coarse sampling
        initial_samples = max(100, min_samples)
        t_coarse = np.linspace(0, duration, initial_samples)
        
        try:
            values_coarse = func(t_coarse, *args, **kwargs)
        except Exception as e:
            raise ValueError(f"Function evaluation failed: {str(e)}")
        
        values_coarse = np.asarray(values_coarse, dtype=np.float64)
        
        # Compute approximate curvature using second differences
        curvature = self._compute_curvature(values_coarse)
        
        # Determine sampling density based on curvature
        sample_density = self._compute_sample_density(curvature, tolerance)
        
        # Generate adaptive time points
        t_adaptive = self._generate_adaptive_points(
            t_coarse, 
            sample_density, 
            min_samples, 
            max_samples
        )
        
        # Evaluate at adaptive points
        values_adaptive = func(t_adaptive, *args, **kwargs)
        values_adaptive = np.asarray(values_adaptive, dtype=np.float64)
        
        # Interpolate to uniform grid for audio output
        interpolator = interp1d(
            t_adaptive, 
            values_adaptive, 
            kind='cubic',
            fill_value='extrapolate'
        )
        
        t_uniform = np.linspace(0, duration, int(duration * self.sample_rate), endpoint=False)
        values = interpolator(t_uniform)
        
        # Normalize if requested
        if normalize:
            values = self._normalize(values)
            
        return values
    
    def _compute_curvature(self, values: np.ndarray) -> np.ndarray:
        """
        Compute approximate curvature using second differences.
        
        Args:
            values: Array of function values
            
        Returns:
            Array of curvature estimates
        """
        # Compute first differences
        first_diff = np.diff(values)
        
        # Compute second differences (curvature proxy)
        second_diff = np.diff(first_diff)
        
        # Pad to match original length
        curvature = np.abs(second_diff)
        curvature = np.pad(curvature, (1, 1), mode='edge')
        
        return curvature
    
    def _compute_sample_density(
        self, 
        curvature: np.ndarray, 
        tolerance: float
    ) -> np.ndarray:
        """
        Compute sampling density based on curvature.
        
        Args:
            curvature: Array of curvature values
            tolerance: Error tolerance
            
        Returns:
            Array of normalized sampling densities [0, 1]
        """
        # Normalize curvature to [0, 1]
        max_curv = np.max(curvature)
        if max_curv > 0:
            norm_curv = curvature / max_curv
        else:
            norm_curv = np.ones_like(curvature)
        
        # Scale by tolerance (smaller tolerance = higher density)
        density = 1.0 + norm_curv / tolerance
        
        # Normalize to [0, 1]
        density = (density - density.min()) / (density.max() - density.min() + 1e-10)
        
        return density
    
    def _generate_adaptive_points(
        self,
        t: np.ndarray,
        density: np.ndarray,
        min_samples: int,
        max_samples: int
    ) -> np.ndarray:
        """
        Generate adaptive sampling points based on density.
        
        Args:
            t: Original time array
            density: Sampling density array
            min_samples: Minimum number of samples
            max_samples: Maximum number of samples
            
        Returns:
            Array of adaptive time points
        """
        # Compute cumulative density
        cumulative_density = np.cumsum(density)
        cumulative_density /= cumulative_density[-1]  # Normalize to [0, 1]
        
        # Determine number of samples
        total_density = np.sum(density)
        num_samples = int(total_density * len(t))
        num_samples = np.clip(num_samples, min_samples, max_samples)
        
        # Generate uniform points in density space
        uniform_density = np.linspace(0, 1, num_samples)
        
        # Map back to time using interpolation
        t_adaptive = np.interp(uniform_density, cumulative_density, t)
        
        return t_adaptive
    
    def _normalize(self, values: np.ndarray, target_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
        """
        Normalize values to a target range.
        
        Args:
            values: Array of values to normalize
            target_range: Tuple of (min, max) for target range
            
        Returns:
            Normalized array
        """
        v_min, v_max = values.min(), values.max()
        
        # Avoid division by zero
        if v_max - v_min < 1e-10:
            return np.zeros_like(values)
        
        # Normalize to [0, 1]
        normalized = (values - v_min) / (v_max - v_min)
        
        # Scale to target range
        t_min, t_max = target_range
        scaled = normalized * (t_max - t_min) + t_min
        
        return scaled
    
    def sample_multidimensional(
        self,
        func: Callable,
        duration: float,
        num_dimensions: int,
        *args: Any,
        method: str = 'uniform',
        normalize: bool = True,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Sample a multidimensional function (e.g., parametric curves).
        
        Args:
            func: Function that returns array of shape (num_dimensions,) for each time point
            duration: Duration in seconds
            num_dimensions: Number of output dimensions
            *args: Additional positional arguments
            method: Sampling method ('uniform' or 'adaptive')
            normalize: Whether to normalize each dimension
            **kwargs: Additional keyword arguments
            
        Returns:
            Array of shape (num_samples, num_dimensions)
        """
        if method == 'uniform':
            sampler = self.uniform_sample
        elif method == 'adaptive':
            sampler = self.adaptive_sample
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Sample the function
        result = sampler(func, duration, *args, normalize=False, **kwargs)
        
        # Reshape if necessary
        if result.ndim == 1:
            # Function returned 1D array, assume it needs reshaping
            num_samples = len(result) // num_dimensions
            result = result[:num_samples * num_dimensions].reshape(num_samples, num_dimensions)
        
        # Normalize each dimension if requested
        if normalize:
            for i in range(result.shape[1]):
                result[:, i] = self._normalize(result[:, i])
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get engine metadata.
        
        Returns:
            Dictionary containing engine configuration
        """
        return {
            'sample_rate': self.sample_rate,
            'time_step': self.dt,
            'nyquist_frequency': self.sample_rate / 2
        }

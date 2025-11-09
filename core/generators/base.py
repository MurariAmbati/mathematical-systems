"""
Base Art Generator Class

Abstract base class for all art generators.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray
from core.utils import SeededRNG


class ArtGenerator(ABC):
    """
    Abstract base class for mathematical art generators.
    
    All generators must implement the generate() method which returns
    point arrays representing the artwork.
    """
    
    # Unique identifier for this generator type
    id: str = "base"
    
    # Human-readable name
    name: str = "Base Generator"
    
    def __init__(
        self,
        samples: int = 1000,
        seed: Optional[int] = None,
        **params
    ):
        """
        Initialize art generator.
        
        Args:
            samples: Number of points to generate
            seed: Random seed for reproducibility
            **params: Additional generator-specific parameters
        """
        self.samples = samples
        self.seed = seed
        self.params = params
        self.rng = SeededRNG(seed)
        self._points: Optional[NDArray] = None
    
    @abstractmethod
    def generate(self) -> NDArray:
        """
        Generate art points.
        
        Returns:
            Array of shape (n, 2) or (n, 3) containing point coordinates
            
        This method must be implemented by all subclasses.
        """
        pass
    
    def get_points(self, regenerate: bool = False) -> NDArray:
        """
        Get generated points (cached).
        
        Args:
            regenerate: Force regeneration even if cached
            
        Returns:
            Generated points array
        """
        if self._points is None or regenerate:
            self._points = self.generate()
        return self._points
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for this generator.
        
        Returns:
            Dictionary with generator configuration
        """
        return {
            "id": self.id,
            "name": self.name,
            "samples": self.samples,
            "seed": self.seed,
            "params": self.params,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ArtGenerator':
        """
        Create generator instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Generator instance
        """
        samples = config.get("samples", 1000)
        seed = config.get("seed", None)
        params = config.get("params", {})
        
        return cls(samples=samples, seed=seed, **params)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(samples={self.samples}, seed={self.seed})"


class ParametricGenerator(ArtGenerator):
    """
    Base class for parametric curve/surface generators.
    
    Subclasses need to implement the parametric equations.
    """
    
    def __init__(
        self,
        t_range: tuple = (0, 2*np.pi),
        samples: int = 1000,
        seed: Optional[int] = None,
        **params
    ):
        """
        Initialize parametric generator.
        
        Args:
            t_range: (t_min, t_max) parameter range
            samples: Number of samples
            seed: Random seed
            **params: Additional parameters
        """
        super().__init__(samples=samples, seed=seed, **params)
        self.t_range = t_range
    
    def get_parameter_values(self) -> NDArray:
        """Get array of parameter values."""
        return np.linspace(self.t_range[0], self.t_range[1], self.samples)


class CartesianGenerator(ArtGenerator):
    """
    Base class for Cartesian coordinate generators.
    
    Generates points on a 2D grid and evaluates functions.
    """
    
    def __init__(
        self,
        x_range: tuple = (-1, 1),
        y_range: tuple = (-1, 1),
        samples: int = 100,
        seed: Optional[int] = None,
        **params
    ):
        """
        Initialize Cartesian generator.
        
        Args:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            samples: Number of samples per dimension
            seed: Random seed
            **params: Additional parameters
        """
        super().__init__(samples=samples, seed=seed, **params)
        self.x_range = x_range
        self.y_range = y_range
    
    def get_grid(self) -> tuple[NDArray, NDArray]:
        """Get X, Y meshgrid."""
        from core.coordinates import cartesian_grid
        return cartesian_grid(self.x_range, self.y_range, self.samples)


class PolarGenerator(ArtGenerator):
    """
    Base class for polar coordinate generators.
    
    Generates points using r = f(theta) equations.
    """
    
    def __init__(
        self,
        theta_range: tuple = (0, 2*np.pi),
        samples: int = 1000,
        seed: Optional[int] = None,
        **params
    ):
        """
        Initialize polar generator.
        
        Args:
            theta_range: (theta_min, theta_max) in radians
            samples: Number of samples
            seed: Random seed
            **params: Additional parameters
        """
        super().__init__(samples=samples, seed=seed, **params)
        self.theta_range = theta_range
    
    def get_theta_values(self) -> NDArray:
        """Get array of theta values."""
        return np.linspace(self.theta_range[0], self.theta_range[1], self.samples)


class IterativeGenerator(ArtGenerator):
    """
    Base class for iterative/recursive generators.
    
    Used for fractals, attractors, and other iterative systems.
    """
    
    def __init__(
        self,
        iterations: int = 10000,
        seed: Optional[int] = None,
        **params
    ):
        """
        Initialize iterative generator.
        
        Args:
            iterations: Number of iterations
            seed: Random seed
            **params: Additional parameters
        """
        super().__init__(samples=iterations, seed=seed, **params)
        self.iterations = iterations

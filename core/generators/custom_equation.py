"""
Custom Equation Generator

Generates art from user-defined mathematical expressions.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union
from core.generators.base import CartesianGenerator
from core.parser import parse_equation
from core.evaluator import Evaluator
from core.utils import NoiseGenerator


class CustomEquation(CartesianGenerator):
    """
    Generate art from custom mathematical expressions in Cartesian coordinates.
    
    Evaluates f(x, y) over a 2D grid and creates various visualizations.
    """
    
    id = "custom_equation"
    name = "Custom Equation"
    
    def __init__(
        self,
        expr: Union[str, 'ParsedExpression'],
        x_range: tuple = (-3, 3),
        y_range: tuple = (-3, 3),
        samples: int = 100,
        threshold: Optional[float] = None,
        mode: str = "grid",
        seed: int = None,
        **params
    ):
        """
        Initialize custom equation generator.
        
        Args:
            expr: Mathematical expression as string or ParsedExpression
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            samples: Number of samples per dimension
            threshold: Optional threshold for level curves
            mode: "grid" (all points), "contour" (level curves), or "scatter" (random)
            seed: Random seed
        """
        super().__init__(
            x_range=x_range,
            y_range=y_range,
            samples=samples,
            seed=seed,
            expr=expr, threshold=threshold, mode=mode,
            **params
        )
        
        self.expr = expr
        self.threshold = threshold
        self.mode = mode
        
        # Create evaluator
        if isinstance(expr, str):
            self.evaluator = Evaluator(expr)
        else:
            self.evaluator = Evaluator(expr)
    
    def generate(self) -> NDArray:
        """
        Generate custom equation art points.
        
        Returns:
            Array of shape (n, 2) or (n, 3) with point coordinates and values
        """
        if self.mode == "grid":
            return self._generate_grid()
        elif self.mode == "contour":
            return self._generate_contour()
        elif self.mode == "scatter":
            return self._generate_scatter()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _generate_grid(self) -> NDArray:
        """Generate all grid points with function values."""
        X, Y = self.get_grid()
        Z = self.evaluator.evaluate(x=X, y=Y)
        
        # Flatten and stack
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Filter by threshold if specified
        if self.threshold is not None:
            mask = np.abs(points[:, 2]) < self.threshold
            points = points[mask]
        
        return points
    
    def _generate_contour(self) -> NDArray:
        """Generate points along level curves."""
        X, Y = self.get_grid()
        Z = self.evaluator.evaluate(x=X, y=Y)
        
        # Find points where function value is close to threshold
        if self.threshold is None:
            threshold = 0.0
        else:
            threshold = self.threshold
        
        # Simple approach: find grid points close to threshold
        epsilon = 0.1
        mask = np.abs(Z - threshold) < epsilon
        
        x_contour = X[mask]
        y_contour = Y[mask]
        
        return np.column_stack([x_contour, y_contour])
    
    def _generate_scatter(self) -> NDArray:
        """Generate random scatter points weighted by function value."""
        # Generate random points
        x = self.rng.uniform(self.x_range[0], self.x_range[1], self.samples * 10)
        y = self.rng.uniform(self.y_range[0], self.y_range[1], self.samples * 10)
        
        # Evaluate function
        z = self.evaluator.evaluate(x=x, y=y)
        
        # Weight by function value (keep points with higher |z|)
        weights = np.abs(z)
        weights = weights / weights.sum()
        
        # Sample points based on weights
        indices = self.rng.choice(len(x), size=self.samples, replace=False, p=weights)
        
        return np.column_stack([x[indices], y[indices]])


class NoiseField(CartesianGenerator):
    """
    Generate art from Perlin/Simplex noise fields.
    """
    
    id = "noise_field"
    name = "Noise Field"
    
    def __init__(
        self,
        x_range: tuple = (-5, 5),
        y_range: tuple = (-5, 5),
        samples: int = 100,
        noise_type: str = "perlin",
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 1.0,
        seed: int = None,
        **params
    ):
        """
        Initialize noise field generator.
        
        Args:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            samples: Number of samples per dimension
            noise_type: "perlin" or "simplex"
            octaves: Number of noise octaves
            persistence: Amplitude decrease per octave
            lacunarity: Frequency increase per octave
            scale: Overall noise scale
            seed: Random seed
        """
        super().__init__(
            x_range=x_range,
            y_range=y_range,
            samples=samples,
            seed=seed,
            noise_type=noise_type, octaves=octaves,
            persistence=persistence, lacunarity=lacunarity, scale=scale,
            **params
        )
        
        self.noise_gen = NoiseGenerator(seed=seed, noise_type=noise_type)
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.scale = scale
    
    def generate(self) -> NDArray:
        """Generate noise field points."""
        X, Y = self.get_grid()
        
        # Generate noise values
        Z = self.noise_gen.noise_2d(
            X, Y,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            scale=self.scale
        )
        
        # Flatten and stack
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        return points


class HybridGenerator(CartesianGenerator):
    """
    Combine analytical expression with noise for organic effects.
    """
    
    id = "hybrid"
    name = "Hybrid Expression + Noise"
    
    def __init__(
        self,
        expr: str,
        x_range: tuple = (-3, 3),
        y_range: tuple = (-3, 3),
        samples: int = 100,
        noise_weight: float = 0.3,
        noise_scale: float = 1.0,
        seed: int = None,
        **params
    ):
        """
        Initialize hybrid generator.
        
        Args:
            expr: Mathematical expression
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            samples: Number of samples per dimension
            noise_weight: Weight of noise (0 to 1)
            noise_scale: Scale of noise features
            seed: Random seed
        """
        super().__init__(
            x_range=x_range,
            y_range=y_range,
            samples=samples,
            seed=seed,
            expr=expr, noise_weight=noise_weight, noise_scale=noise_scale,
            **params
        )
        
        self.evaluator = Evaluator(expr)
        self.noise_gen = NoiseGenerator(seed=seed)
        self.noise_weight = noise_weight
        self.noise_scale = noise_scale
    
    def generate(self) -> NDArray:
        """Generate hybrid expression + noise points."""
        X, Y = self.get_grid()
        
        # Evaluate analytical expression
        Z_expr = self.evaluator.evaluate(x=X, y=Y)
        
        # Generate noise
        Z_noise = self.noise_gen.noise_2d(
            X, Y,
            octaves=3,
            persistence=0.5,
            lacunarity=2.0,
            scale=self.noise_scale
        )
        
        # Blend
        Z = (1 - self.noise_weight) * Z_expr + self.noise_weight * Z_noise
        
        # Flatten and stack
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        return points

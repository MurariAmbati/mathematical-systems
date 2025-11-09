"""
Polar Pattern Generator

Generates patterns using polar coordinate equations r = f(θ).
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Callable
from core.generators.base import PolarGenerator
from core.parser import ParsedExpression, parse_equation
from core.evaluator import Evaluator


class PolarPattern(PolarGenerator):
    """
    Generate patterns from polar equations r = f(θ).
    
    Supports both string expressions and callable functions.
    """
    
    id = "polar_pattern"
    name = "Polar Pattern"
    
    def __init__(
        self,
        expr: Union[str, Callable],
        theta_range: tuple = (0, 2*np.pi),
        samples: int = 2000,
        time_param: Optional[float] = None,
        seed: int = None,
        **params
    ):
        """
        Initialize polar pattern generator.
        
        Args:
            expr: Polar equation as string (e.g., "sin(5*theta)") or callable
            theta_range: (theta_min, theta_max) in radians
            samples: Number of points
            time_param: Optional time parameter for animations
            seed: Random seed
        """
        super().__init__(
            theta_range=theta_range,
            samples=samples,
            seed=seed,
            expr=expr, time_param=time_param,
            **params
        )
        
        self.expr = expr
        self.time_param = time_param if time_param is not None else 0.0
        
        # Parse expression if it's a string
        if isinstance(expr, str):
            self.evaluator = Evaluator(expr)
        else:
            self.evaluator = None
    
    def generate(self, t: Optional[float] = None) -> NDArray:
        """
        Generate polar pattern points.
        
        Args:
            t: Optional time parameter (overrides time_param if provided)
        
        Returns:
            Array of shape (samples, 2) with (x, y) coordinates
        """
        theta = self.get_theta_values()
        time = t if t is not None else self.time_param
        
        # Evaluate r = f(theta, t)
        if self.evaluator is not None:
            # String expression - use evaluator
            r = self.evaluator.evaluate(theta=theta, t=time)
        else:
            # Callable function
            r = self.expr(theta, time)
        
        # Convert to Cartesian coordinates
        from core.coordinates import polar_to_cartesian
        x, y = polar_to_cartesian(r, theta)
        
        return np.column_stack([x, y])


class MultiPolarPattern(PolarGenerator):
    """
    Generate patterns from multiple polar equations overlaid.
    """
    
    id = "multi_polar_pattern"
    name = "Multi Polar Pattern"
    
    def __init__(
        self,
        expressions: list[str],
        theta_range: tuple = (0, 2*np.pi),
        samples: int = 2000,
        seed: int = None,
        **params
    ):
        """
        Initialize multi-polar pattern generator.
        
        Args:
            expressions: List of polar equations
            theta_range: (theta_min, theta_max) in radians
            samples: Number of points per expression
            seed: Random seed
        """
        super().__init__(
            theta_range=theta_range,
            samples=samples,
            seed=seed,
            expressions=expressions,
            **params
        )
        
        self.expressions = expressions
        self.evaluators = [Evaluator(expr) for expr in expressions]
    
    def generate(self) -> NDArray:
        """
        Generate multi-polar pattern points.
        
        Returns:
            Array of shape (n_expressions * samples, 2)
        """
        all_points = []
        
        for evaluator in self.evaluators:
            theta = self.get_theta_values()
            r = evaluator.evaluate(theta=theta, t=0)
            
            from core.coordinates import polar_to_cartesian
            x, y = polar_to_cartesian(r, theta)
            
            points = np.column_stack([x, y])
            all_points.append(points)
        
        return np.vstack(all_points)


class MaunderRose(PolarGenerator):
    """
    Maurer rose pattern - a variation of rose curves with modular arithmetic.
    
    Polar equation: r = sin(n*θ)
    But we plot points at angles: θ, θ+d, θ+2d, ...
    where d = 360° * (360°/n)
    """
    
    id = "maurer_rose"
    name = "Maurer Rose"
    
    def __init__(
        self,
        n: float = 2.0,
        d: float = 29.0,
        samples: int = 360,
        seed: int = None,
        **params
    ):
        """
        Initialize Maurer rose generator.
        
        Args:
            n: Numerator for rose equation
            d: Angle increment in degrees
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            theta_range=(0, 2*np.pi),
            samples=samples,
            seed=seed,
            n=n, d=d,
            **params
        )
        
        self.n = n
        self.d = d
    
    def generate(self) -> NDArray:
        """Generate Maurer rose points."""
        # Create angle sequence with modular arithmetic
        angles_deg = np.arange(self.samples) * self.d
        theta = np.deg2rad(angles_deg)
        
        # Rose equation
        r = np.sin(self.n * theta)
        
        # Convert to Cartesian
        from core.coordinates import polar_to_cartesian
        x, y = polar_to_cartesian(r, theta)
        
        return np.column_stack([x, y])


class Cardioid(PolarGenerator):
    """
    Cardioid curve (heart-shaped).
    
    Polar equation: r = a(1 + cos(θ))
    """
    
    id = "cardioid"
    name = "Cardioid"
    
    def __init__(
        self,
        a: float = 1.0,
        samples: int = 1000,
        seed: int = None,
        **params
    ):
        """
        Initialize cardioid generator.
        
        Args:
            a: Scale parameter
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            theta_range=(0, 2*np.pi),
            samples=samples,
            seed=seed,
            a=a,
            **params
        )
        
        self.a = a
    
    def generate(self) -> NDArray:
        """Generate cardioid points."""
        theta = self.get_theta_values()
        r = self.a * (1 + np.cos(theta))
        
        from core.coordinates import polar_to_cartesian
        x, y = polar_to_cartesian(r, theta)
        
        return np.column_stack([x, y])


class Limacon(PolarGenerator):
    """
    Limaçon curve (generalized cardioid).
    
    Polar equation: r = a + b*cos(θ)
    """
    
    id = "limacon"
    name = "Limaçon"
    
    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.5,
        samples: int = 1000,
        seed: int = None,
        **params
    ):
        """
        Initialize limaçon generator.
        
        Args:
            a: Constant term
            b: Cosine coefficient
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            theta_range=(0, 2*np.pi),
            samples=samples,
            seed=seed,
            a=a, b=b,
            **params
        )
        
        self.a = a
        self.b = b
    
    def generate(self) -> NDArray:
        """Generate limaçon points."""
        theta = self.get_theta_values()
        r = self.a + self.b * np.cos(theta)
        
        from core.coordinates import polar_to_cartesian
        x, y = polar_to_cartesian(r, theta)
        
        return np.column_stack([x, y])


class ArchimedeanSpiral(PolarGenerator):
    """
    Archimedean spiral.
    
    Polar equation: r = a + b*θ
    """
    
    id = "archimedean_spiral"
    name = "Archimedean Spiral"
    
    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        n_turns: int = 5,
        samples: int = 2000,
        seed: int = None,
        **params
    ):
        """
        Initialize Archimedean spiral generator.
        
        Args:
            a: Starting radius offset
            b: Growth rate
            n_turns: Number of spiral turns
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            theta_range=(0, 2*np.pi*n_turns),
            samples=samples,
            seed=seed,
            a=a, b=b, n_turns=n_turns,
            **params
        )
        
        self.a = a
        self.b = b
    
    def generate(self) -> NDArray:
        """Generate Archimedean spiral points."""
        theta = self.get_theta_values()
        r = self.a + self.b * theta
        
        from core.coordinates import polar_to_cartesian
        x, y = polar_to_cartesian(r, theta)
        
        return np.column_stack([x, y])


class LogarithmicSpiral(PolarGenerator):
    """
    Logarithmic (equiangular) spiral.
    
    Polar equation: r = a * e^(b*θ)
    """
    
    id = "logarithmic_spiral"
    name = "Logarithmic Spiral"
    
    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.2,
        n_turns: int = 3,
        samples: int = 2000,
        seed: int = None,
        **params
    ):
        """
        Initialize logarithmic spiral generator.
        
        Args:
            a: Initial radius
            b: Growth rate
            n_turns: Number of spiral turns
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            theta_range=(0, 2*np.pi*n_turns),
            samples=samples,
            seed=seed,
            a=a, b=b, n_turns=n_turns,
            **params
        )
        
        self.a = a
        self.b = b
    
    def generate(self) -> NDArray:
        """Generate logarithmic spiral points."""
        theta = self.get_theta_values()
        r = self.a * np.exp(self.b * theta)
        
        from core.coordinates import polar_to_cartesian
        x, y = polar_to_cartesian(r, theta)
        
        return np.column_stack([x, y])

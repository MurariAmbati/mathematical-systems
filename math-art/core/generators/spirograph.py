"""
Spirograph Generator

Generates hypotrochoid and epitrochoid curves (spirograph patterns).
"""

import numpy as np
from numpy.typing import NDArray
from core.generators.base import ParametricGenerator


class Spirograph(ParametricGenerator):
    """
    Spirograph pattern generator using hypotrochoid/epitrochoid equations.
    
    A spirograph is created by tracing a point attached to a circle rolling
    inside (hypotrochoid) or outside (epitrochoid) another circle.
    
    Parametric equations:
        Hypotrochoid (inner):
            x(t) = (R-r)*cos(t) + a*cos((R-r)/r * t)
            y(t) = (R-r)*sin(t) - a*sin((R-r)/r * t)
        
        Epitrochoid (outer):
            x(t) = (R+r)*cos(t) - a*cos((R+r)/r * t)
            y(t) = (R+r)*sin(t) - a*sin((R+r)/r * t)
    """
    
    id = "spirograph"
    name = "Spirograph"
    
    def __init__(
        self,
        R: float = 5.0,
        r: float = 3.0,
        a: float = 2.0,
        mode: str = "hypo",
        samples: int = 5000,
        seed: int = None,
        **params
    ):
        """
        Initialize spirograph generator.
        
        Args:
            R: Radius of fixed circle
            r: Radius of rolling circle
            a: Distance from center of rolling circle to drawing point
            mode: "hypo" for hypotrochoid (inner) or "epi" for epitrochoid (outer)
            samples: Number of points
            seed: Random seed
        """
        # Calculate number of rotations needed for closed curve
        # The curve closes when t = 2π * lcm(R, r) / r
        from math import gcd
        lcm_val = abs(R * r) / gcd(int(R * 10), int(r * 10))
        t_max = 2 * np.pi * lcm_val / r
        
        super().__init__(
            t_range=(0, t_max),
            samples=samples,
            seed=seed,
            R=R, r=r, a=a, mode=mode,
            **params
        )
        
        self.R = R
        self.r = r
        self.a = a
        self.mode = mode.lower()
    
    def generate(self) -> NDArray:
        """
        Generate spirograph points.
        
        Returns:
            Array of shape (samples, 2) with (x, y) coordinates
        """
        t = self.get_parameter_values()
        
        if self.mode == "hypo":
            # Hypotrochoid (inner rolling)
            x = (self.R - self.r) * np.cos(t) + self.a * np.cos((self.R - self.r) / self.r * t)
            y = (self.R - self.r) * np.sin(t) - self.a * np.sin((self.R - self.r) / self.r * t)
        elif self.mode == "epi":
            # Epitrochoid (outer rolling)
            x = (self.R + self.r) * np.cos(t) - self.a * np.cos((self.R + self.r) / self.r * t)
            y = (self.R + self.r) * np.sin(t) - self.a * np.sin((self.R + self.r) / self.r * t)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'hypo' or 'epi'")
        
        points = np.column_stack([x, y])
        return points


class GeneralizedSpirograph(ParametricGenerator):
    """
    Generalized spirograph with custom frequency ratios.
    
    Uses the general form:
        x(t) = A*cos(f1*t) + B*cos(f2*t)
        y(t) = C*sin(f3*t) + D*sin(f4*t)
    """
    
    id = "generalized_spirograph"
    name = "Generalized Spirograph"
    
    def __init__(
        self,
        A: float = 1.0,
        B: float = 0.5,
        C: float = 1.0,
        D: float = 0.5,
        f1: float = 1.0,
        f2: float = 3.0,
        f3: float = 2.0,
        f4: float = 4.0,
        samples: int = 5000,
        seed: int = None,
        **params
    ):
        """
        Initialize generalized spirograph.
        
        Args:
            A, B, C, D: Amplitude coefficients
            f1, f2, f3, f4: Frequency coefficients
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            t_range=(0, 2*np.pi*10),  # Multiple periods
            samples=samples,
            seed=seed,
            A=A, B=B, C=C, D=D, f1=f1, f2=f2, f3=f3, f4=f4,
            **params
        )
        
        self.A, self.B, self.C, self.D = A, B, C, D
        self.f1, self.f2, self.f3, self.f4 = f1, f2, f3, f4
    
    def generate(self) -> NDArray:
        """Generate generalized spirograph points."""
        t = self.get_parameter_values()
        
        x = self.A * np.cos(self.f1 * t) + self.B * np.cos(self.f2 * t)
        y = self.C * np.sin(self.f3 * t) + self.D * np.sin(self.f4 * t)
        
        return np.column_stack([x, y])


class RosePattern(ParametricGenerator):
    """
    Rose pattern generator (rhodonea curve).
    
    Polar equation: r = A*cos(k*θ)
    """
    
    id = "rose_pattern"
    name = "Rose Pattern"
    
    def __init__(
        self,
        A: float = 1.0,
        k: float = 5.0,
        samples: int = 2000,
        seed: int = None,
        **params
    ):
        """
        Initialize rose pattern.
        
        Args:
            A: Amplitude
            k: Number of petals (if k is odd) or 2k petals (if k is even)
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            t_range=(0, 2*np.pi),
            samples=samples,
            seed=seed,
            A=A, k=k,
            **params
        )
        
        self.A = A
        self.k = k
    
    def generate(self) -> NDArray:
        """Generate rose pattern points."""
        theta = self.get_parameter_values()
        
        # Polar equation
        r = self.A * np.cos(self.k * theta)
        
        # Convert to Cartesian
        from core.coordinates import polar_to_cartesian
        x, y = polar_to_cartesian(r, theta)
        
        return np.column_stack([x, y])

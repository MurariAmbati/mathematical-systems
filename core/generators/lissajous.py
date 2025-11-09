"""
Lissajous Curves Generator

Generates Lissajous figures - parametric curves that create beautiful patterns.
"""

import numpy as np
from numpy.typing import NDArray
from core.generators.base import ParametricGenerator


class Lissajous(ParametricGenerator):
    """
    Lissajous curve generator.
    
    Parametric equations:
        x(t) = A*sin(a*t + δ)
        y(t) = B*sin(b*t)
    
    where A, B are amplitudes, a, b are frequency ratios, and δ is phase shift.
    """
    
    id = "lissajous"
    name = "Lissajous Curve"
    
    def __init__(
        self,
        A: float = 1.0,
        B: float = 1.0,
        a: float = 3.0,
        b: float = 2.0,
        delta: float = np.pi/2,
        samples: int = 2000,
        seed: int = None,
        **params
    ):
        """
        Initialize Lissajous curve generator.
        
        Args:
            A: X amplitude
            B: Y amplitude
            a: X frequency
            b: Y frequency
            delta: Phase shift (in radians)
            samples: Number of points
            seed: Random seed
        """
        # Calculate period for closed curve
        from math import gcd
        if isinstance(a, int) and isinstance(b, int):
            period_factor = max(a, b) / gcd(int(a), int(b))
        else:
            period_factor = max(a, b)
        
        super().__init__(
            t_range=(0, 2*np.pi*period_factor),
            samples=samples,
            seed=seed,
            A=A, B=B, a=a, b=b, delta=delta,
            **params
        )
        
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.delta = delta
    
    def generate(self) -> NDArray:
        """
        Generate Lissajous curve points.
        
        Returns:
            Array of shape (samples, 2) with (x, y) coordinates
        """
        t = self.get_parameter_values()
        
        x = self.A * np.sin(self.a * t + self.delta)
        y = self.B * np.sin(self.b * t)
        
        return np.column_stack([x, y])


class Lissajous3D(ParametricGenerator):
    """
    3D Lissajous curve generator.
    
    Parametric equations:
        x(t) = A*sin(a*t + δx)
        y(t) = B*sin(b*t + δy)
        z(t) = C*sin(c*t + δz)
    """
    
    id = "lissajous_3d"
    name = "3D Lissajous Curve"
    
    def __init__(
        self,
        A: float = 1.0,
        B: float = 1.0,
        C: float = 1.0,
        a: float = 3.0,
        b: float = 2.0,
        c: float = 1.0,
        delta_x: float = 0.0,
        delta_y: float = np.pi/2,
        delta_z: float = 0.0,
        samples: int = 3000,
        seed: int = None,
        **params
    ):
        """
        Initialize 3D Lissajous curve generator.
        
        Args:
            A, B, C: Amplitudes for x, y, z
            a, b, c: Frequencies for x, y, z
            delta_x, delta_y, delta_z: Phase shifts (in radians)
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            t_range=(0, 2*np.pi*10),
            samples=samples,
            seed=seed,
            A=A, B=B, C=C, a=a, b=b, c=c,
            delta_x=delta_x, delta_y=delta_y, delta_z=delta_z,
            **params
        )
        
        self.A, self.B, self.C = A, B, C
        self.a, self.b, self.c = a, b, c
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
    
    def generate(self) -> NDArray:
        """
        Generate 3D Lissajous curve points.
        
        Returns:
            Array of shape (samples, 3) with (x, y, z) coordinates
        """
        t = self.get_parameter_values()
        
        x = self.A * np.sin(self.a * t + self.delta_x)
        y = self.B * np.sin(self.b * t + self.delta_y)
        z = self.C * np.sin(self.c * t + self.delta_z)
        
        return np.column_stack([x, y, z])


class BowditchCurve(ParametricGenerator):
    """
    Bowditch curve (generalized Lissajous with cosine terms).
    
    Parametric equations:
        x(t) = A*sin(a*t + δ) + A2*cos(a2*t)
        y(t) = B*sin(b*t) + B2*cos(b2*t)
    """
    
    id = "bowditch"
    name = "Bowditch Curve"
    
    def __init__(
        self,
        A: float = 1.0,
        B: float = 1.0,
        A2: float = 0.5,
        B2: float = 0.5,
        a: float = 3.0,
        b: float = 2.0,
        a2: float = 5.0,
        b2: float = 4.0,
        delta: float = 0.0,
        samples: int = 3000,
        seed: int = None,
        **params
    ):
        """
        Initialize Bowditch curve generator.
        
        Args:
            A, B: Primary amplitudes
            A2, B2: Secondary amplitudes
            a, b: Primary frequencies
            a2, b2: Secondary frequencies
            delta: Phase shift
            samples: Number of points
            seed: Random seed
        """
        super().__init__(
            t_range=(0, 2*np.pi*10),
            samples=samples,
            seed=seed,
            A=A, B=B, A2=A2, B2=B2, a=a, b=b, a2=a2, b2=b2, delta=delta,
            **params
        )
        
        self.A, self.B = A, B
        self.A2, self.B2 = A2, B2
        self.a, self.b = a, b
        self.a2, self.b2 = a2, b2
        self.delta = delta
    
    def generate(self) -> NDArray:
        """Generate Bowditch curve points."""
        t = self.get_parameter_values()
        
        x = self.A * np.sin(self.a * t + self.delta) + self.A2 * np.cos(self.a2 * t)
        y = self.B * np.sin(self.b * t) + self.B2 * np.cos(self.b2 * t)
        
        return np.column_stack([x, y])

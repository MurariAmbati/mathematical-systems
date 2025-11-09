"""
Attractor Generators

Generates points from chaotic dynamical systems (strange attractors).
"""

import numpy as np
from numpy.typing import NDArray
from core.generators.base import IterativeGenerator


class LorenzAttractor(IterativeGenerator):
    """
    Lorenz attractor - one of the most famous chaotic systems.
    
    Differential equations:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
    
    where σ (sigma), ρ (rho), and β (beta) are system parameters.
    """
    
    id = "lorenz_attractor"
    name = "Lorenz Attractor"
    
    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8/3,
        dt: float = 0.01,
        initial: tuple = (1.0, 1.0, 1.0),
        iterations: int = 10000,
        seed: int = None,
        **params
    ):
        """
        Initialize Lorenz attractor.
        
        Args:
            sigma: Prandtl number (default 10.0)
            rho: Rayleigh number (default 28.0)
            beta: Geometric factor (default 8/3)
            dt: Time step for integration
            initial: Initial (x, y, z) position
            iterations: Number of iterations
            seed: Random seed
        """
        super().__init__(
            iterations=iterations,
            seed=seed,
            sigma=sigma, rho=rho, beta=beta, dt=dt, initial=initial,
            **params
        )
        
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.initial = np.array(initial)
    
    def generate(self) -> NDArray:
        """
        Generate Lorenz attractor points.
        
        Returns:
            Array of shape (iterations, 3) with (x, y, z) coordinates
        """
        points = np.zeros((self.iterations, 3))
        points[0] = self.initial
        
        for i in range(1, self.iterations):
            x, y, z = points[i-1]
            
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            
            points[i] = points[i-1] + self.dt * np.array([dx, dy, dz])
        
        return points


class CliffordAttractor(IterativeGenerator):
    """
    Clifford attractor - produces intricate fractal-like patterns.
    
    Iterative equations:
        x_{n+1} = sin(a*y_n) + c*cos(a*x_n)
        y_{n+1} = sin(b*x_n) + d*cos(b*y_n)
    """
    
    id = "clifford_attractor"
    name = "Clifford Attractor"
    
    def __init__(
        self,
        a: float = -1.4,
        b: float = 1.6,
        c: float = 1.0,
        d: float = 0.7,
        initial: tuple = (0.0, 0.0),
        iterations: int = 50000,
        seed: int = None,
        **params
    ):
        """
        Initialize Clifford attractor.
        
        Args:
            a, b, c, d: System parameters
            initial: Initial (x, y) position
            iterations: Number of iterations
            seed: Random seed
        """
        super().__init__(
            iterations=iterations,
            seed=seed,
            a=a, b=b, c=c, d=d, initial=initial,
            **params
        )
        
        self.a, self.b, self.c, self.d = a, b, c, d
        self.initial = np.array(initial)
    
    def generate(self) -> NDArray:
        """
        Generate Clifford attractor points.
        
        Returns:
            Array of shape (iterations, 2) with (x, y) coordinates
        """
        points = np.zeros((self.iterations, 2))
        points[0] = self.initial
        
        for i in range(1, self.iterations):
            x, y = points[i-1]
            
            x_new = np.sin(self.a * y) + self.c * np.cos(self.a * x)
            y_new = np.sin(self.b * x) + self.d * np.cos(self.b * y)
            
            points[i] = [x_new, y_new]
        
        return points


class IkedaAttractor(IterativeGenerator):
    """
    Ikeda attractor - models laser physics dynamics.
    
    Iterative equations:
        t_n = 0.4 - 6/(1 + x_n^2 + y_n^2)
        x_{n+1} = 1 + u*(x_n*cos(t_n) - y_n*sin(t_n))
        y_{n+1} = u*(x_n*sin(t_n) + y_n*cos(t_n))
    """
    
    id = "ikeda_attractor"
    name = "Ikeda Attractor"
    
    def __init__(
        self,
        u: float = 0.918,
        initial: tuple = (0.1, 0.1),
        iterations: int = 20000,
        seed: int = None,
        **params
    ):
        """
        Initialize Ikeda attractor.
        
        Args:
            u: System parameter (typically around 0.9)
            initial: Initial (x, y) position
            iterations: Number of iterations
            seed: Random seed
        """
        super().__init__(
            iterations=iterations,
            seed=seed,
            u=u, initial=initial,
            **params
        )
        
        self.u = u
        self.initial = np.array(initial)
    
    def generate(self) -> NDArray:
        """
        Generate Ikeda attractor points.
        
        Returns:
            Array of shape (iterations, 2) with (x, y) coordinates
        """
        points = np.zeros((self.iterations, 2))
        points[0] = self.initial
        
        for i in range(1, self.iterations):
            x, y = points[i-1]
            
            t = 0.4 - 6 / (1 + x**2 + y**2)
            
            x_new = 1 + self.u * (x * np.cos(t) - y * np.sin(t))
            y_new = self.u * (x * np.sin(t) + y * np.cos(t))
            
            points[i] = [x_new, y_new]
        
        return points


class DeJongAttractor(IterativeGenerator):
    """
    De Jong (Peter de Jong) attractor.
    
    Iterative equations:
        x_{n+1} = sin(a*y_n) - cos(b*x_n)
        y_{n+1} = sin(c*x_n) - cos(d*y_n)
    """
    
    id = "dejong_attractor"
    name = "De Jong Attractor"
    
    def __init__(
        self,
        a: float = 1.4,
        b: float = -2.3,
        c: float = 2.4,
        d: float = -2.1,
        initial: tuple = (0.0, 0.0),
        iterations: int = 50000,
        seed: int = None,
        **params
    ):
        """
        Initialize De Jong attractor.
        
        Args:
            a, b, c, d: System parameters
            initial: Initial (x, y) position
            iterations: Number of iterations
            seed: Random seed
        """
        super().__init__(
            iterations=iterations,
            seed=seed,
            a=a, b=b, c=c, d=d, initial=initial,
            **params
        )
        
        self.a, self.b, self.c, self.d = a, b, c, d
        self.initial = np.array(initial)
    
    def generate(self) -> NDArray:
        """Generate De Jong attractor points."""
        points = np.zeros((self.iterations, 2))
        points[0] = self.initial
        
        for i in range(1, self.iterations):
            x, y = points[i-1]
            
            x_new = np.sin(self.a * y) - np.cos(self.b * x)
            y_new = np.sin(self.c * x) - np.cos(self.d * y)
            
            points[i] = [x_new, y_new]
        
        return points


class HoffAttractor(IterativeGenerator):
    """
    Hoff attractor - creates complex symmetric patterns.
    
    Iterative equations:
        x_{n+1} = y_n - sign(x_n)*sqrt(|b*x_n - c|)
        y_{n+1} = a - x_n
    """
    
    id = "hoff_attractor"
    name = "Hoff Attractor"
    
    def __init__(
        self,
        a: float = 0.3,
        b: float = 0.2,
        c: float = 1.0,
        initial: tuple = (0.0, 0.0),
        iterations: int = 30000,
        seed: int = None,
        **params
    ):
        """
        Initialize Hoff attractor.
        
        Args:
            a, b, c: System parameters
            initial: Initial (x, y) position
            iterations: Number of iterations
            seed: Random seed
        """
        super().__init__(
            iterations=iterations,
            seed=seed,
            a=a, b=b, c=c, initial=initial,
            **params
        )
        
        self.a, self.b, self.c = a, b, c
        self.initial = np.array(initial)
    
    def generate(self) -> NDArray:
        """Generate Hoff attractor points."""
        points = np.zeros((self.iterations, 2))
        points[0] = self.initial
        
        for i in range(1, self.iterations):
            x, y = points[i-1]
            
            x_new = y - np.sign(x) * np.sqrt(np.abs(self.b * x - self.c))
            y_new = self.a - x
            
            points[i] = [x_new, y_new]
        
        return points

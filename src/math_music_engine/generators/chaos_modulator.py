"""
Chaos modulator generator.

Generates modulation signals from chaotic dynamical systems.
"""

from typing import Optional, Dict, Any, Literal
import numpy as np


class ChaosModulator:
    """
    Modulation generator using chaotic systems.
    
    Supports:
    - Logistic map
    - Lorenz attractor
    - Rössler attractor
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize chaos modulator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
    def logistic_map(
        self,
        r: float,
        x0: float,
        duration: float,
        iterations_per_sample: int = 10
    ) -> np.ndarray:
        """
        Generate signal from logistic map.
        
        Mathematical equation: x(n+1) = r * x(n) * (1 - x(n))
        Chaotic for r ≈ 3.57 to 4.0
        
        Args:
            r: Control parameter (typically 3.5 - 4.0)
            x0: Initial condition (0 < x0 < 1)
            duration: Duration in seconds
            iterations_per_sample: Number of map iterations per audio sample
            
        Returns:
            Chaotic modulation signal
        """
        num_samples = int(duration * self.sample_rate)
        signal = np.zeros(num_samples)
        
        x = x0
        for i in range(num_samples):
            # Iterate the map multiple times per sample for smoother output
            for _ in range(iterations_per_sample):
                x = r * x * (1 - x)
            signal[i] = x
        
        # Normalize to [-1, 1]
        signal = 2 * signal - 1
        
        return signal
    
    def lorenz_attractor(
        self,
        duration: float,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        initial_state: Optional[np.ndarray] = None,
        dt: float = 0.01,
        component: Literal['x', 'y', 'z'] = 'x'
    ) -> np.ndarray:
        """
        Generate signal from Lorenz attractor.
        
        Mathematical equations:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
        
        Args:
            duration: Duration in seconds
            sigma: Prandtl number (default: 10.0)
            rho: Rayleigh number (default: 28.0)
            beta: Geometric parameter (default: 8/3)
            initial_state: Initial [x, y, z] state (default: [1, 1, 1])
            dt: Integration time step
            component: Which component to return ('x', 'y', or 'z')
            
        Returns:
            Chaotic modulation signal
        """
        if initial_state is None:
            initial_state = np.array([1.0, 1.0, 1.0])
        
        # Calculate number of integration steps
        num_steps = int(duration / dt)
        
        # Integrate Lorenz equations
        state = initial_state.copy()
        trajectory = np.zeros((num_steps, 3))
        
        for i in range(num_steps):
            x, y, z = state
            
            # Lorenz equations
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            # Euler integration
            state += dt * np.array([dx, dy, dz])
            trajectory[i] = state
        
        # Select component
        component_idx = {'x': 0, 'y': 1, 'z': 2}[component]
        signal = trajectory[:, component_idx]
        
        # Resample to audio sample rate
        num_samples = int(duration * self.sample_rate)
        resampled = np.interp(
            np.linspace(0, len(signal) - 1, num_samples),
            np.arange(len(signal)),
            signal
        )
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(resampled))
        if max_val > 0:
            resampled /= max_val
        
        return resampled
    
    def rossler_attractor(
        self,
        duration: float,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        initial_state: Optional[np.ndarray] = None,
        dt: float = 0.01,
        component: Literal['x', 'y', 'z'] = 'x'
    ) -> np.ndarray:
        """
        Generate signal from Rössler attractor.
        
        Mathematical equations:
        dx/dt = -y - z
        dy/dt = x + ay
        dz/dt = b + z(x - c)
        
        Args:
            duration: Duration in seconds
            a: Parameter a (default: 0.2)
            b: Parameter b (default: 0.2)
            c: Parameter c (default: 5.7)
            initial_state: Initial [x, y, z] state (default: [1, 1, 1])
            dt: Integration time step
            component: Which component to return ('x', 'y', or 'z')
            
        Returns:
            Chaotic modulation signal
        """
        if initial_state is None:
            initial_state = np.array([1.0, 1.0, 1.0])
        
        # Calculate number of integration steps
        num_steps = int(duration / dt)
        
        # Integrate Rössler equations
        state = initial_state.copy()
        trajectory = np.zeros((num_steps, 3))
        
        for i in range(num_steps):
            x, y, z = state
            
            # Rössler equations
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            # Euler integration
            state += dt * np.array([dx, dy, dz])
            trajectory[i] = state
        
        # Select component
        component_idx = {'x': 0, 'y': 1, 'z': 2}[component]
        signal = trajectory[:, component_idx]
        
        # Resample to audio sample rate
        num_samples = int(duration * self.sample_rate)
        resampled = np.interp(
            np.linspace(0, len(signal) - 1, num_samples),
            np.arange(len(signal)),
            signal
        )
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(resampled))
        if max_val > 0:
            resampled /= max_val
        
        return resampled
    
    def henon_map(
        self,
        a: float,
        b: float,
        x0: float,
        y0: float,
        duration: float,
        component: Literal['x', 'y'] = 'x'
    ) -> np.ndarray:
        """
        Generate signal from Hénon map.
        
        Mathematical equations:
        x(n+1) = 1 - a*x(n)² + y(n)
        y(n+1) = b*x(n)
        
        Args:
            a: Parameter a (typically 1.4)
            b: Parameter b (typically 0.3)
            x0: Initial x value
            y0: Initial y value
            duration: Duration in seconds
            component: Which component to return ('x' or 'y')
            
        Returns:
            Chaotic modulation signal
        """
        num_samples = int(duration * self.sample_rate)
        x_vals = np.zeros(num_samples)
        y_vals = np.zeros(num_samples)
        
        x, y = x0, y0
        
        for i in range(num_samples):
            x_new = 1 - a * x**2 + y
            y_new = b * x
            x, y = x_new, y_new
            x_vals[i] = x
            y_vals[i] = y
        
        # Select component
        signal = x_vals if component == 'x' else y_vals
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
        
        return signal
    
    def generate(
        self,
        system: Literal['logistic', 'lorenz', 'rossler', 'henon'],
        duration: float,
        **kwargs
    ) -> np.ndarray:
        """
        Generate chaotic signal from specified system.
        
        Args:
            system: Chaotic system type
            duration: Duration in seconds
            **kwargs: System-specific parameters
            
        Returns:
            Chaotic modulation signal
        """
        if system == 'logistic':
            r = kwargs.get('r', 3.9)
            x0 = kwargs.get('x0', 0.5)
            return self.logistic_map(r, x0, duration)
            
        elif system == 'lorenz':
            return self.lorenz_attractor(duration, **kwargs)
            
        elif system == 'rossler':
            return self.rossler_attractor(duration, **kwargs)
            
        elif system == 'henon':
            a = kwargs.get('a', 1.4)
            b = kwargs.get('b', 0.3)
            x0 = kwargs.get('x0', 0.0)
            y0 = kwargs.get('y0', 0.0)
            return self.henon_map(a, b, x0, y0, duration)
            
        else:
            raise ValueError(f"Unknown chaotic system: {system}")
    
    def get_metadata(self, system: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get generator metadata.
        
        Args:
            system: Chaotic system name
            parameters: System parameters
            
        Returns:
            Dictionary containing generator configuration
        """
        return {
            'generator': 'ChaosModulator',
            'system': system,
            'parameters': parameters,
            'sample_rate': self.sample_rate,
            'mathematical_basis': f'{system} chaotic system'
        }

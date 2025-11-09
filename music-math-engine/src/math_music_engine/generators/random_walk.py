"""
Random walk generator.

Generates stochastic musical patterns using random walks and Markov processes.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np


class RandomWalk:
    """
    Stochastic generator using random walks and Markov processes.
    
    Mathematical basis: Brownian motion, Markov chains, stochastic processes
    """
    
    def __init__(self, sample_rate: int = 44100, seed: Optional[int] = None):
        """
        Initialize random walk generator.
        
        Args:
            sample_rate: Audio sample rate
            seed: Random seed for reproducibility
        """
        self.sample_rate = sample_rate
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        
    def brownian_motion(
        self,
        duration: float,
        step_size: float = 1.0,
        bias: float = 0.0
    ) -> np.ndarray:
        """
        Generate 1D Brownian motion (random walk).
        
        Args:
            duration: Duration in seconds
            step_size: Standard deviation of each step
            bias: Drift parameter (directional bias)
            
        Returns:
            Random walk signal
        """
        num_samples = int(duration * self.sample_rate)
        
        # Generate random steps
        steps = np.random.normal(bias, step_size, num_samples)
        
        # Cumulative sum gives random walk
        walk = np.cumsum(steps)
        
        # Normalize
        walk = (walk - walk.mean()) / (walk.std() + 1e-10)
        
        return walk
    
    def geometric_brownian_motion(
        self,
        duration: float,
        mu: float = 0.0,
        sigma: float = 1.0,
        initial_value: float = 1.0
    ) -> np.ndarray:
        """
        Generate geometric Brownian motion.
        
        Used in finance, produces exponential growth/decay patterns.
        dS = μS dt + σS dW
        
        Args:
            duration: Duration in seconds
            mu: Drift coefficient
            sigma: Volatility coefficient
            initial_value: Initial value
            
        Returns:
            Geometric Brownian motion signal
        """
        num_samples = int(duration * self.sample_rate)
        dt = 1.0 / self.sample_rate
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), num_samples)
        
        # Initialize
        S = np.zeros(num_samples)
        S[0] = initial_value
        
        # Integrate
        for i in range(1, num_samples):
            S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[i])
        
        # Normalize
        S = (S - S.mean()) / (S.std() + 1e-10)
        
        return S
    
    def levy_flight(
        self,
        duration: float,
        alpha: float = 1.5
    ) -> np.ndarray:
        """
        Generate Lévy flight (heavy-tailed random walk).
        
        Args:
            duration: Duration in seconds
            alpha: Stability parameter (0 < alpha <= 2)
                  alpha = 2: Brownian motion
                  alpha = 1: Cauchy distribution
            
        Returns:
            Lévy flight signal
        """
        num_samples = int(duration * self.sample_rate)
        
        # Generate Lévy-distributed steps using inverse transform
        # Approximation using stable distribution
        u = np.random.uniform(-np.pi/2, np.pi/2, num_samples)
        v = np.random.exponential(1, num_samples)
        
        if alpha == 1:
            steps = np.tan(u)
        else:
            steps = (np.sin(alpha * u) / (np.cos(u) ** (1/alpha)) * 
                    (np.cos(u - alpha * u) / v) ** ((1 - alpha) / alpha))
        
        # Clip extreme values
        steps = np.clip(steps, -10, 10)
        
        # Cumulative sum
        walk = np.cumsum(steps)
        
        # Normalize
        walk = (walk - walk.mean()) / (walk.std() + 1e-10)
        
        return walk
    
    def markov_melody(
        self,
        num_notes: int,
        transition_matrix: Optional[np.ndarray] = None,
        states: Optional[List[int]] = None,
        initial_state: Optional[int] = None
    ) -> List[int]:
        """
        Generate melody using Markov chain.
        
        Args:
            num_notes: Number of notes to generate
            transition_matrix: State transition probabilities
                             Shape: (n_states, n_states)
                             If None, generates random matrix
            states: List of MIDI note values for each state
                   If None, uses [60, 62, 64, 65, 67, 69, 71, 72]
            initial_state: Initial state index (random if None)
            
        Returns:
            List of MIDI note numbers
        """
        # Default states (C major scale)
        if states is None:
            states = [60, 62, 64, 65, 67, 69, 71, 72]
        
        n_states = len(states)
        
        # Generate random transition matrix if not provided
        if transition_matrix is None:
            transition_matrix = np.random.rand(n_states, n_states)
            # Normalize rows to sum to 1
            transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize
        if initial_state is None:
            current_state = np.random.randint(0, n_states)
        else:
            current_state = initial_state
        
        melody = [states[current_state]]
        
        # Generate sequence
        for _ in range(num_notes - 1):
            # Sample next state based on transition probabilities
            current_state = np.random.choice(
                n_states,
                p=transition_matrix[current_state]
            )
            melody.append(states[current_state])
        
        return melody
    
    def ornstein_uhlenbeck_process(
        self,
        duration: float,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Generate Ornstein-Uhlenbeck process (mean-reverting random walk).
        
        dX = θ(μ - X)dt + σdW
        
        Args:
            duration: Duration in seconds
            theta: Mean reversion rate
            mu: Long-term mean
            sigma: Volatility
            
        Returns:
            O-U process signal
        """
        num_samples = int(duration * self.sample_rate)
        dt = 1.0 / self.sample_rate
        
        X = np.zeros(num_samples)
        X[0] = mu
        
        for i in range(1, num_samples):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW
        
        # Normalize
        X = (X - X.mean()) / (X.std() + 1e-10)
        
        return X
    
    def perlin_noise(
        self,
        duration: float,
        octaves: int = 4,
        persistence: float = 0.5
    ) -> np.ndarray:
        """
        Generate Perlin noise (smooth random walk).
        
        Args:
            duration: Duration in seconds
            octaves: Number of octaves (detail levels)
            persistence: Amplitude decay per octave
            
        Returns:
            Perlin noise signal
        """
        num_samples = int(duration * self.sample_rate)
        
        def interpolate(a: float, b: float, x: float) -> float:
            """Cosine interpolation."""
            ft = x * np.pi
            f = (1 - np.cos(ft)) * 0.5
            return a * (1 - f) + b * f
        
        def noise_octave(samples: int, frequency: float) -> np.ndarray:
            """Generate one octave of noise."""
            num_points = int(samples / frequency) + 1
            random_values = np.random.rand(num_points) * 2 - 1
            
            result = np.zeros(samples)
            for i in range(samples):
                x = i / frequency
                x0 = int(x)
                x1 = x0 + 1
                
                if x1 < num_points:
                    result[i] = interpolate(random_values[x0], random_values[x1], x - x0)
                else:
                    result[i] = random_values[x0]
            
            return result
        
        # Combine octaves
        signal = np.zeros(num_samples)
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(octaves):
            signal += amplitude * noise_octave(num_samples, frequency)
            amplitude *= persistence
            frequency *= 2
        
        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-10)
        
        return signal
    
    def random_walk_melody(
        self,
        num_notes: int,
        base_note: int = 60,
        step_range: Tuple[int, int] = (-3, 3),
        bounds: Tuple[int, int] = (48, 84)
    ) -> List[int]:
        """
        Generate melody using bounded random walk.
        
        Args:
            num_notes: Number of notes
            base_note: Starting MIDI note
            step_range: Range of possible interval steps
            bounds: (min_note, max_note) bounds
            
        Returns:
            List of MIDI note numbers
        """
        melody = [base_note]
        current_note = base_note
        
        for _ in range(num_notes - 1):
            # Random step
            step = np.random.randint(step_range[0], step_range[1] + 1)
            current_note += step
            
            # Reflect at boundaries
            if current_note < bounds[0]:
                current_note = bounds[0] + (bounds[0] - current_note)
            elif current_note > bounds[1]:
                current_note = bounds[1] - (current_note - bounds[1])
            
            melody.append(current_note)
        
        return melody
    
    def biased_random_walk(
        self,
        duration: float,
        bias_function: Optional[callable] = None
    ) -> np.ndarray:
        """
        Generate random walk with time-varying bias.
        
        Args:
            duration: Duration in seconds
            bias_function: Function f(t) returning bias at time t
                          If None, uses sine wave bias
            
        Returns:
            Biased random walk signal
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        if bias_function is None:
            # Default: sinusoidal bias
            bias = np.sin(2 * np.pi * t / duration)
        else:
            bias = bias_function(t)
        
        # Generate steps with bias
        steps = np.random.randn(num_samples) + bias
        walk = np.cumsum(steps)
        
        # Normalize
        walk = (walk - walk.mean()) / (walk.std() + 1e-10)
        
        return walk
    
    def get_metadata(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get generator metadata.
        
        Args:
            method: Generation method
            parameters: Method parameters
            
        Returns:
            Dictionary containing generator configuration
        """
        return {
            'generator': 'RandomWalk',
            'method': method,
            'parameters': parameters,
            'sample_rate': self.sample_rate,
            'seed': self.seed,
            'mathematical_basis': 'Stochastic processes and random walks'
        }

"""
Fibonacci rhythm generator.

Generates rhythmic patterns based on Fibonacci ratios and sequences.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np


class FibonacciRhythm:
    """
    Rhythm generator based on Fibonacci sequence and golden ratio.
    
    Mathematical basis: Fibonacci sequence F(n) = F(n-1) + F(n-2)
    Golden ratio φ = (1 + √5) / 2 ≈ 1.618
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize Fibonacci rhythm generator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
    def generate_sequence(self, length: int, start: int = 1) -> List[int]:
        """
        Generate Fibonacci sequence.
        
        Args:
            length: Number of terms
            start: Starting index (default: 1)
            
        Returns:
            List of Fibonacci numbers
        """
        if length <= 0:
            return []
        if length == 1:
            return [1]
        
        fib = [1, 1]
        for _ in range(2, length):
            fib.append(fib[-1] + fib[-2])
        
        return fib[start-1:start-1+length] if start > 1 else fib[:length]
    
    def generate_rhythm_pattern(
        self,
        num_beats: int,
        subdivision: int = 16
    ) -> np.ndarray:
        """
        Generate a rhythm pattern using Fibonacci sequence.
        
        Args:
            num_beats: Number of beats in the pattern
            subdivision: Subdivision per beat (e.g., 16 for sixteenth notes)
            
        Returns:
            Binary rhythm pattern (1 = hit, 0 = rest)
        """
        total_slots = num_beats * subdivision
        pattern = np.zeros(total_slots, dtype=int)
        
        # Generate Fibonacci sequence
        fib_seq = self.generate_sequence(num_beats)
        
        # Place hits based on Fibonacci numbers
        current_pos = 0
        for fib_num in fib_seq:
            if current_pos < total_slots:
                pattern[current_pos] = 1
                current_pos += fib_num % subdivision + 1
        
        return pattern
    
    def generate_timing(
        self,
        num_events: int,
        duration: float,
        method: str = "ratios"
    ) -> np.ndarray:
        """
        Generate event timings based on Fibonacci relationships.
        
        Args:
            num_events: Number of events
            duration: Total duration in seconds
            method: Timing method:
                   - "ratios": Use Fibonacci ratios
                   - "sequence": Use Fibonacci sequence directly
                   - "golden": Use golden ratio subdivisions
            
        Returns:
            Array of event times in seconds
        """
        if method == "ratios":
            # Use Fibonacci ratios for timing
            fib_seq = self.generate_sequence(num_events)
            ratios = np.array(fib_seq, dtype=float)
            ratios /= ratios.sum()  # Normalize to sum to 1
            cumulative = np.cumsum(ratios)
            times = cumulative * duration
            
        elif method == "sequence":
            # Use Fibonacci numbers as time intervals
            fib_seq = self.generate_sequence(num_events)
            intervals = np.array(fib_seq, dtype=float)
            intervals = intervals / intervals.sum() * duration
            times = np.cumsum(intervals)
            
        elif method == "golden":
            # Use golden ratio for recursive subdivision
            times = []
            remaining = duration
            current = 0.0
            
            for _ in range(num_events):
                # Divide remaining time by golden ratio
                interval = remaining / self.phi
                current += interval
                times.append(current)
                remaining -= interval
            
            times = np.array(times)
        else:
            # Default: uniform distribution
            times = np.linspace(0, duration, num_events, endpoint=False)
        
        return times
    
    def generate_tempo_curve(
        self,
        duration: float,
        base_tempo: float = 120.0,
        variation: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a tempo curve based on Fibonacci ratios.
        
        Args:
            duration: Duration in seconds
            base_tempo: Base tempo in BPM
            variation: Tempo variation factor (0-1)
            
        Returns:
            Tuple of (time_points, tempo_values)
        """
        num_points = 100
        t = np.linspace(0, duration, num_points)
        
        # Generate Fibonacci-based tempo modulation
        fib_length = 8
        fib_seq = self.generate_sequence(fib_length)
        fib_normalized = np.array(fib_seq) / max(fib_seq)
        
        # Interpolate Fibonacci pattern across duration
        tempo_pattern = np.interp(
            t,
            np.linspace(0, duration, fib_length),
            fib_normalized
        )
        
        # Apply variation
        tempo_values = base_tempo * (1 + variation * (tempo_pattern - 0.5))
        
        return t, tempo_values
    
    def generate_pulse_train(
        self,
        pattern: np.ndarray,
        tempo: float,
        duration: float,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert rhythm pattern to audio pulse train.
        
        Args:
            pattern: Binary rhythm pattern
            tempo: Tempo in BPM
            duration: Duration in seconds
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            Audio signal with pulses
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        num_samples = int(duration * sample_rate)
        signal = np.zeros(num_samples)
        
        # Calculate time per subdivision
        beats_per_second = tempo / 60.0
        time_per_subdivision = 1.0 / (beats_per_second * len(pattern))
        
        # Generate pulses
        pulse_duration = 0.01  # 10ms pulse
        pulse_samples = int(pulse_duration * sample_rate)
        
        for i, hit in enumerate(pattern):
            if hit:
                # Calculate pulse position
                pulse_time = i * time_per_subdivision
                pulse_sample = int(pulse_time * sample_rate)
                
                # Add pulse (exponential decay)
                if pulse_sample + pulse_samples < num_samples:
                    t_pulse = np.arange(pulse_samples) / sample_rate
                    pulse = np.exp(-t_pulse * 100)  # Decay
                    signal[pulse_sample:pulse_sample + pulse_samples] += pulse
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
        
        return signal
    
    def generate_euclidean_rhythm(
        self,
        pulses: int,
        steps: int
    ) -> np.ndarray:
        """
        Generate Euclidean rhythm (distributes pulses evenly across steps).
        
        Uses Fibonacci-inspired algorithm for distribution.
        
        Args:
            pulses: Number of pulses (hits)
            steps: Total number of steps
            
        Returns:
            Binary rhythm pattern
        """
        if pulses >= steps:
            return np.ones(steps, dtype=int)
        
        pattern = np.zeros(steps, dtype=int)
        
        # Distribute pulses as evenly as possible
        for i in range(pulses):
            position = int(i * steps / pulses)
            pattern[position] = 1
        
        return pattern
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get generator metadata.
        
        Returns:
            Dictionary containing generator configuration
        """
        return {
            'generator': 'FibonacciRhythm',
            'sample_rate': self.sample_rate,
            'golden_ratio': self.phi,
            'mathematical_basis': 'Fibonacci sequence and golden ratio'
        }

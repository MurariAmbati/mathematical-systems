"""
Fourier synthesis generator.

Generates audio from Fourier series coefficients for harmonic timbre shaping.
"""

from typing import List, Optional, Dict, Any
import numpy as np


class FourierSynth:
    """
    Fourier synthesis generator using harmonic series.
    
    Mathematical basis: Fourier series representation
    f(t) = a₀/2 + Σ(aₙ·cos(nωt) + bₙ·sin(nωt))
    """
    
    def __init__(
        self,
        harmonics: Optional[List[float]] = None,
        sample_rate: int = 44100
    ):
        """
        Initialize Fourier synthesizer.
        
        Args:
            harmonics: List of harmonic amplitudes (relative to fundamental)
                      If None, defaults to [1.0] (pure sine wave)
            sample_rate: Audio sample rate
        """
        self.harmonics = harmonics if harmonics is not None else [1.0]
        self.sample_rate = sample_rate
        
    def generate(
        self,
        fundamental_freq: float,
        duration: float,
        phase_offsets: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate audio using Fourier synthesis.
        
        Args:
            fundamental_freq: Fundamental frequency in Hz
            duration: Duration in seconds
            phase_offsets: Optional phase offsets for each harmonic (in radians)
            
        Returns:
            Generated audio signal
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Initialize signal
        signal = np.zeros(num_samples)
        
        # Set default phase offsets
        if phase_offsets is None:
            phase_offsets = [0.0] * len(self.harmonics)
        elif len(phase_offsets) < len(self.harmonics):
            phase_offsets.extend([0.0] * (len(self.harmonics) - len(phase_offsets)))
        
        # Add each harmonic
        for n, (amplitude, phase) in enumerate(zip(self.harmonics, phase_offsets), start=1):
            if amplitude > 0:
                harmonic_freq = fundamental_freq * n
                signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t + phase)
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
        
        return signal
    
    def generate_from_spectrum(
        self,
        frequency_spectrum: Dict[float, float],
        duration: float
    ) -> np.ndarray:
        """
        Generate audio from a frequency spectrum.
        
        Args:
            frequency_spectrum: Dictionary mapping frequencies (Hz) to amplitudes
            duration: Duration in seconds
            
        Returns:
            Generated audio signal
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        signal = np.zeros(num_samples)
        
        for freq, amplitude in frequency_spectrum.items():
            if amplitude > 0:
                signal += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
        
        return signal
    
    def set_harmonics_from_formula(
        self,
        formula: str,
        num_harmonics: int = 16
    ):
        """
        Set harmonic amplitudes from a mathematical formula.
        
        Args:
            formula: Formula for nth harmonic amplitude
                    Supported: "1/n" (sawtooth), "1/n**2" (triangle), 
                              "odd" (square wave), etc.
            num_harmonics: Number of harmonics to generate
        """
        harmonics = []
        
        for n in range(1, num_harmonics + 1):
            if formula == "sawtooth" or formula == "1/n":
                amplitude = 1.0 / n
            elif formula == "triangle" or formula == "1/n**2":
                if n % 2 == 1:  # Only odd harmonics
                    amplitude = 1.0 / (n ** 2)
                else:
                    amplitude = 0.0
            elif formula == "square" or formula == "odd":
                if n % 2 == 1:  # Only odd harmonics
                    amplitude = 1.0 / n
                else:
                    amplitude = 0.0
            elif formula == "exponential":
                amplitude = np.exp(-n / num_harmonics)
            else:
                # Try to evaluate as Python expression
                try:
                    amplitude = eval(formula, {"n": n, "np": np})
                except:
                    amplitude = 1.0 / n
            
            harmonics.append(amplitude)
        
        self.harmonics = harmonics
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get generator metadata.
        
        Returns:
            Dictionary containing generator configuration
        """
        return {
            'generator': 'FourierSynth',
            'harmonics': self.harmonics,
            'num_harmonics': len(self.harmonics),
            'sample_rate': self.sample_rate,
            'mathematical_basis': 'Fourier series'
        }


def create_harmonic_series(
    harmonic_type: str,
    num_harmonics: int = 16
) -> List[float]:
    """
    Create predefined harmonic series.
    
    Args:
        harmonic_type: Type of harmonic series
                      ("sawtooth", "square", "triangle", "exponential")
        num_harmonics: Number of harmonics
        
    Returns:
        List of harmonic amplitudes
    """
    synth = FourierSynth()
    synth.set_harmonics_from_formula(harmonic_type, num_harmonics)
    return synth.harmonics

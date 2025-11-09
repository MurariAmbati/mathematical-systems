"""
Mathematical generators for the Math-Music-Engine.
"""

from .fourier_synth import FourierSynth, create_harmonic_series
from .fibonacci_rhythm import FibonacciRhythm
from .chaos_modulator import ChaosModulator
from .fractal_melody import FractalMelody, LSYSTEM_PRESETS
from .prime_sequence import PrimeSequence
from .random_walk import RandomWalk

__all__ = [
    'FourierSynth',
    'create_harmonic_series',
    'FibonacciRhythm',
    'ChaosModulator',
    'FractalMelody',
    'LSYSTEM_PRESETS',
    'PrimeSequence',
    'RandomWalk'
]

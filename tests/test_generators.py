"""
Tests for generators module.
"""

import pytest
import numpy as np
from math_music_engine.generators import (
    FourierSynth,
    FibonacciRhythm,
    ChaosModulator,
    FractalMelody,
    PrimeSequence,
    RandomWalk
)


class TestFourierSynth:
    """Test suite for FourierSynth."""
    
    def test_initialization(self):
        """Test Fourier synth initialization."""
        synth = FourierSynth(harmonics=[1.0, 0.5, 0.25])
        assert len(synth.harmonics) == 3
        assert synth.sample_rate == 44100
    
    def test_generate(self):
        """Test audio generation."""
        synth = FourierSynth(harmonics=[1.0, 0.5])
        signal = synth.generate(440.0, duration=1.0)
        
        assert len(signal) == 44100
        assert np.max(np.abs(signal)) <= 1.0


class TestFibonacciRhythm:
    """Test suite for FibonacciRhythm."""
    
    def test_fibonacci_sequence(self):
        """Test Fibonacci sequence generation."""
        gen = FibonacciRhythm()
        fib = gen.generate_sequence(8)
        
        assert fib == [1, 1, 2, 3, 5, 8, 13, 21]
    
    def test_rhythm_pattern(self):
        """Test rhythm pattern generation."""
        gen = FibonacciRhythm()
        pattern = gen.generate_rhythm_pattern(num_beats=4, subdivision=16)
        
        assert len(pattern) == 64
        assert np.sum(pattern) > 0  # Should have some hits


class TestChaosModulator:
    """Test suite for ChaosModulator."""
    
    def test_logistic_map(self):
        """Test logistic map generation."""
        gen = ChaosModulator()
        signal = gen.logistic_map(r=3.9, x0=0.5, duration=1.0)
        
        assert len(signal) == 44100
        assert np.all(np.abs(signal) <= 1.0)
    
    def test_lorenz_attractor(self):
        """Test Lorenz attractor generation."""
        gen = ChaosModulator()
        signal = gen.lorenz_attractor(duration=1.0)
        
        assert len(signal) == 44100
        assert np.max(np.abs(signal)) <= 1.0


class TestFractalMelody:
    """Test suite for FractalMelody."""
    
    def test_lsystem_generation(self):
        """Test L-system generation."""
        gen = FractalMelody()
        lsystem = gen.generate_lsystem("A", {"A": "AB", "B": "A"}, iterations=3)
        
        assert len(lsystem) > 0
        assert 'A' in lsystem or 'B' in lsystem
    
    def test_recursive_melody(self):
        """Test recursive melody generation."""
        gen = FractalMelody()
        melody = gen.generate_recursive_melody([1, 2], recursion_depth=2)
        
        assert len(melody) > 2


class TestPrimeSequence:
    """Test suite for PrimeSequence."""
    
    def test_prime_generation(self):
        """Test prime number generation."""
        gen = PrimeSequence()
        primes = gen.generate_primes(10)
        
        assert len(primes) == 10
        assert primes[:5] == [2, 3, 5, 7, 11]
    
    def test_prime_melody(self):
        """Test prime melody generation."""
        gen = PrimeSequence()
        melody = gen.prime_melody(num_notes=10, method="gaps")
        
        assert len(melody) == 10
        assert all(0 <= note <= 127 for note in melody)


class TestRandomWalk:
    """Test suite for RandomWalk."""
    
    def test_brownian_motion(self):
        """Test Brownian motion generation."""
        gen = RandomWalk(seed=42)
        signal = gen.brownian_motion(duration=1.0)
        
        assert len(signal) == 44100
        assert np.max(np.abs(signal)) <= 10.0  # Should be normalized
    
    def test_markov_melody(self):
        """Test Markov melody generation."""
        gen = RandomWalk(seed=42)
        melody = gen.markov_melody(num_notes=20)
        
        assert len(melody) == 20
        assert all(0 <= note <= 127 for note in melody)


if __name__ == '__main__':
    pytest.main([__file__])

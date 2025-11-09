"""
Tests for synthesis module.
"""

import pytest
import numpy as np
from math_music_engine.core import (
    Oscillator,
    WaveformType,
    ADSR,
    AdditiveSynthesizer,
    FrequencyModulation,
    AmplitudeModulation
)


class TestOscillator:
    """Test suite for Oscillator."""
    
    def test_sine_generation(self):
        """Test sine wave generation."""
        osc = Oscillator(sample_rate=44100)
        signal = osc.generate(WaveformType.SINE, 440.0, duration=1.0)
        
        assert len(signal) == 44100
        assert np.max(np.abs(signal)) <= 1.0
    
    def test_square_generation(self):
        """Test square wave generation."""
        osc = Oscillator()
        signal = osc.generate(WaveformType.SQUARE, 440.0, duration=0.1)
        
        assert len(signal) == 4410
        # Square wave should have values close to -1 or 1
        assert np.max(signal) > 0.9
        assert np.min(signal) < -0.9


class TestADSR:
    """Test suite for ADSR envelope."""
    
    def test_envelope_generation(self):
        """Test ADSR envelope generation."""
        adsr = ADSR(attack=0.1, decay=0.1, sustain=0.7, release=0.2)
        envelope = adsr.generate(duration=1.0)
        
        assert len(envelope) == 44100
        assert np.max(envelope) <= 1.0
        assert np.min(envelope) >= 0.0
    
    def test_envelope_shape(self):
        """Test envelope has correct shape."""
        adsr = ADSR(attack=0.1, decay=0.1, sustain=0.5, release=0.1)
        envelope = adsr.generate(duration=1.0, gate_duration=0.5)
        
        # Envelope should start at 0
        assert envelope[0] == 0.0
        # Should reach peak during attack
        assert np.max(envelope) > 0.9


class TestAdditiveSynthesizer:
    """Test suite for AdditiveSynthesizer."""
    
    def test_synthesis(self):
        """Test additive synthesis."""
        synth = AdditiveSynthesizer()
        signal = synth.synthesize(
            fundamental_freq=440.0,
            harmonics=[1.0, 0.5, 0.25],
            duration=1.0
        )
        
        assert len(signal) == 44100
        assert np.max(np.abs(signal)) <= 1.0


class TestFrequencyModulation:
    """Test suite for FM synthesis."""
    
    def test_fm_synthesis(self):
        """Test FM synthesis."""
        fm = FrequencyModulation()
        signal = fm.synthesize(
            carrier_freq=440.0,
            modulator_freq=220.0,
            modulation_index=2.0,
            duration=1.0
        )
        
        assert len(signal) == 44100
        assert np.max(np.abs(signal)) <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])

"""
Tests for mapping engine module.
"""

import pytest
import numpy as np
from math_music_engine.core import MappingEngine, MappingMode, Scale


class TestMappingEngine:
    """Test suite for MappingEngine."""
    
    def test_initialization(self):
        """Test mapper initialization."""
        mapper = MappingEngine()
        assert mapper.freq_min == 20.0
        assert mapper.freq_max == 20000.0
        assert mapper.sample_rate == 44100
    
    def test_linear_frequency_mapping(self):
        """Test linear frequency mapping."""
        mapper = MappingEngine()
        values = np.array([0.0, 0.5, 1.0])
        freqs = mapper.map_to_frequency(values, mode=MappingMode.LINEAR, 
                                       freq_min=100, freq_max=200)
        
        assert np.allclose(freqs[0], 100)
        assert np.allclose(freqs[1], 150)
        assert np.allclose(freqs[2], 200)
    
    def test_logarithmic_frequency_mapping(self):
        """Test logarithmic frequency mapping."""
        mapper = MappingEngine()
        values = np.array([0.0, 0.5, 1.0])
        freqs = mapper.map_to_frequency(values, mode=MappingMode.LOGARITHMIC, 
                                       freq_min=100, freq_max=400)
        
        assert np.isclose(freqs[0], 100)
        assert np.isclose(freqs[2], 400)
        assert np.isclose(freqs[1], 200)  # Geometric mean of 100 and 400
    
    def test_amplitude_mapping(self):
        """Test amplitude mapping."""
        mapper = MappingEngine()
        values = np.array([0.0, 0.5, 1.0])
        amps = mapper.map_to_amplitude(values)
        
        assert np.all(amps >= 0)
        assert np.all(amps <= 1)
    
    def test_midi_mapping(self):
        """Test MIDI note mapping."""
        mapper = MappingEngine()
        values = np.array([0.0, 0.5, 1.0])
        midi = mapper.map_to_midi(values, root_note="C4")
        
        assert np.all(midi >= 0)
        assert np.all(midi <= 127)
        assert midi[0] < midi[2]  # Increasing
    
    def test_scale_quantization(self):
        """Test scale quantization."""
        mapper = MappingEngine()
        values = np.linspace(0, 1, 20)
        midi = mapper.map_to_midi(values, scale=Scale.MAJOR, root_note="C4")
        
        # All notes should be in C major scale
        assert np.all(midi >= 0)
        assert np.all(midi <= 127)
    
    def test_note_to_frequency_conversion(self):
        """Test note name to frequency conversion."""
        mapper = MappingEngine()
        
        # A4 should be 440 Hz
        freq = mapper._note_to_frequency("A4")
        assert np.isclose(freq, 440.0)
        
        # C4 (middle C) should be ~261.63 Hz
        freq = mapper._note_to_frequency("C4")
        assert np.isclose(freq, 261.63, atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__])

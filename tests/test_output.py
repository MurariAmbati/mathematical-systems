"""
Tests for output manager module.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from math_music_engine.core import OutputManager


class TestOutputManager:
    """Test suite for OutputManager."""
    
    def test_initialization(self):
        """Test output manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = OutputManager(tmpdir)
            assert mgr.output_dir == Path(tmpdir)
    
    def test_export_audio(self):
        """Test audio export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = OutputManager(tmpdir)
            
            # Create test signal
            signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
            
            # Export
            filepath = mgr.export_audio(signal, "test.wav", sample_rate=44100)
            
            assert filepath.exists()
            assert filepath.suffix == '.wav'
    
    def test_export_metadata(self):
        """Test metadata export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = OutputManager(tmpdir)
            
            metadata = {
                'test_key': 'test_value',
                'number': 42
            }
            
            filepath = mgr.export_metadata(metadata, "test_metadata.json")
            
            assert filepath.exists()
            
            # Load and verify
            with open(filepath) as f:
                loaded = json.load(f)
            
            assert loaded['test_key'] == 'test_value'
            assert loaded['number'] == 42
            assert 'export_timestamp' in loaded
    
    def test_create_reproducibility_metadata(self):
        """Test reproducibility metadata creation."""
        mgr = OutputManager()
        
        metadata = mgr.create_reproducibility_metadata(
            expression="sin(2*pi*t)",
            parameters={'duration': 10},
            seed=42
        )
        
        assert metadata['expression'] == "sin(2*pi*t)"
        assert metadata['parameters']['duration'] == 10
        assert metadata['seed'] == 42
        assert 'timestamp' in metadata


if __name__ == '__main__':
    pytest.main([__file__])

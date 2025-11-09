"""
Mapping engine for transforming mathematical outputs to musical parameters.

Maps mathematical function outputs to:
- Frequency (pitch)
- Amplitude
- Time
- Timbre
"""

from typing import Optional, Dict, Any, List, Callable
import numpy as np
from enum import Enum


class MappingMode(Enum):
    """Enumeration of mapping modes."""
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    QUANTIZED = "quantized"


class Scale(Enum):
    """Musical scales."""
    CHROMATIC = "chromatic"
    MAJOR = "major"
    MINOR = "minor"
    HARMONIC_MINOR = "harmonic_minor"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    WHOLE_TONE = "whole_tone"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"


# Scale intervals in semitones from root
SCALE_INTERVALS = {
    Scale.CHROMATIC: list(range(12)),
    Scale.MAJOR: [0, 2, 4, 5, 7, 9, 11],
    Scale.MINOR: [0, 2, 3, 5, 7, 8, 10],
    Scale.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
    Scale.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
    Scale.PENTATONIC_MINOR: [0, 3, 5, 7, 10],
    Scale.WHOLE_TONE: [0, 2, 4, 6, 8, 10],
    Scale.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    Scale.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
    Scale.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
    Scale.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    Scale.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
}


class MappingEngine:
    """
    Engine for mapping mathematical outputs to musical parameters.
    
    All mappings are deterministic and invertible (where applicable).
    """
    
    def __init__(
        self,
        freq_min: float = 20.0,
        freq_max: float = 20000.0,
        sample_rate: int = 44100
    ):
        """
        Initialize the mapping engine.
        
        Args:
            freq_min: Minimum frequency in Hz
            freq_max: Maximum frequency in Hz
            sample_rate: Audio sample rate
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sample_rate = sample_rate
        
        # MIDI note number range
        self.midi_min = 0
        self.midi_max = 127
        
    def map_to_frequency(
        self,
        values: np.ndarray,
        mode: MappingMode = MappingMode.LOGARITHMIC,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        scale: Optional[Scale] = None,
        root_note: str = "A4",
    ) -> np.ndarray:
        """
        Map values to frequencies.
        
        Args:
            values: Input values (assumed normalized to [-1, 1] or [0, 1])
            mode: Mapping mode (linear, logarithmic, exponential, quantized)
            freq_min: Minimum frequency (default: self.freq_min)
            freq_max: Maximum frequency (default: self.freq_max)
            scale: Musical scale for quantization (optional)
            root_note: Root note for scale (e.g., "A4", "C3")
            
        Returns:
            Array of frequencies in Hz
        """
        if freq_min is None:
            freq_min = self.freq_min
        if freq_max is None:
            freq_max = self.freq_max
            
        # Normalize values to [0, 1]
        normalized = self._normalize_to_unit(values)
        
        if mode == MappingMode.LINEAR:
            frequencies = freq_min + normalized * (freq_max - freq_min)
            
        elif mode == MappingMode.LOGARITHMIC:
            # Logarithmic mapping (perceptually linear pitch)
            log_min = np.log2(freq_min)
            log_max = np.log2(freq_max)
            frequencies = 2 ** (log_min + normalized * (log_max - log_min))
            
        elif mode == MappingMode.EXPONENTIAL:
            # Exponential mapping
            frequencies = freq_min * (freq_max / freq_min) ** normalized
            
        elif mode == MappingMode.QUANTIZED:
            # First map logarithmically, then quantize
            log_min = np.log2(freq_min)
            log_max = np.log2(freq_max)
            frequencies = 2 ** (log_min + normalized * (log_max - log_min))
            
            if scale is not None:
                frequencies = self._quantize_to_scale(frequencies, scale, root_note)
        else:
            raise ValueError(f"Unknown mapping mode: {mode}")
            
        return frequencies
    
    def map_to_amplitude(
        self,
        values: np.ndarray,
        mode: MappingMode = MappingMode.LINEAR,
        amp_min: float = 0.0,
        amp_max: float = 1.0,
        compression: float = 1.0
    ) -> np.ndarray:
        """
        Map values to amplitudes.
        
        Args:
            values: Input values
            mode: Mapping mode
            amp_min: Minimum amplitude
            amp_max: Maximum amplitude
            compression: Compression factor (< 1 = compress, > 1 = expand)
            
        Returns:
            Array of amplitudes [0, 1]
        """
        # Normalize to [0, 1]
        normalized = self._normalize_to_unit(values)
        
        # Apply compression
        if compression != 1.0:
            normalized = normalized ** compression
        
        if mode == MappingMode.LINEAR:
            amplitudes = amp_min + normalized * (amp_max - amp_min)
            
        elif mode == MappingMode.LOGARITHMIC:
            # Logarithmic amplitude (decibel-like)
            # Map to dB scale, then back to linear
            db_min = 20 * np.log10(amp_min + 1e-10)
            db_max = 20 * np.log10(amp_max + 1e-10)
            db_values = db_min + normalized * (db_max - db_min)
            amplitudes = 10 ** (db_values / 20)
            
        elif mode == MappingMode.EXPONENTIAL:
            amplitudes = amp_min + (amp_max - amp_min) * (np.exp(normalized) - 1) / (np.e - 1)
            
        else:
            amplitudes = amp_min + normalized * (amp_max - amp_min)
        
        # Clip to valid range
        amplitudes = np.clip(amplitudes, 0.0, 1.0)
        
        return amplitudes
    
    def map_to_midi(
        self,
        values: np.ndarray,
        scale: Optional[Scale] = None,
        root_note: str = "C4",
        octave_range: int = 4
    ) -> np.ndarray:
        """
        Map values to MIDI note numbers.
        
        Args:
            values: Input values (normalized to [-1, 1] or [0, 1])
            scale: Musical scale for quantization
            root_note: Root note (e.g., "C4")
            octave_range: Number of octaves to span
            
        Returns:
            Array of MIDI note numbers (0-127)
        """
        # Normalize to [0, 1]
        normalized = self._normalize_to_unit(values)
        
        # Get root MIDI note
        root_midi = self._note_to_midi(root_note)
        
        if scale is None:
            # Chromatic mapping
            midi_notes = root_midi + normalized * (octave_range * 12)
        else:
            # Scale-quantized mapping
            intervals = SCALE_INTERVALS[scale]
            total_notes = len(intervals) * octave_range
            
            # Map to scale degree
            scale_degrees = (normalized * total_notes).astype(int)
            scale_degrees = np.clip(scale_degrees, 0, total_notes - 1)
            
            # Convert scale degree to MIDI note
            octaves = scale_degrees // len(intervals)
            notes_in_octave = scale_degrees % len(intervals)
            midi_notes = root_midi + octaves * 12 + np.array([intervals[n] for n in notes_in_octave])
        
        # Clip to MIDI range
        midi_notes = np.clip(midi_notes, self.midi_min, self.midi_max).astype(int)
        
        return midi_notes
    
    def map_to_timbre(
        self,
        values: np.ndarray,
        num_harmonics: int = 8,
        mode: str = "linear"
    ) -> np.ndarray:
        """
        Map values to timbre (harmonic weights).
        
        Args:
            values: Input values (can be multidimensional)
            num_harmonics: Number of harmonics
            mode: Distribution mode ("linear", "exponential", "random")
            
        Returns:
            Array of shape (len(values), num_harmonics) with normalized harmonic weights
        """
        values = np.asarray(values)
        
        if values.ndim == 1:
            # Single dimension - distribute across harmonics
            harmonic_weights = np.zeros((len(values), num_harmonics))
            
            if mode == "linear":
                # Linear distribution
                for i, v in enumerate(values):
                    weights = np.linspace(1.0, 0.1, num_harmonics) * abs(v)
                    harmonic_weights[i] = weights / (np.sum(weights) + 1e-10)
                    
            elif mode == "exponential":
                # Exponential decay
                for i, v in enumerate(values):
                    weights = np.exp(-np.arange(num_harmonics) * 0.5) * abs(v)
                    harmonic_weights[i] = weights / (np.sum(weights) + 1e-10)
                    
            elif mode == "random":
                # Controlled random
                for i, v in enumerate(values):
                    weights = np.random.rand(num_harmonics) * abs(v)
                    harmonic_weights[i] = weights / (np.sum(weights) + 1e-10)
        else:
            # Multiple dimensions - use directly as harmonic weights
            harmonic_weights = values[:, :num_harmonics]
            # Normalize each row
            row_sums = np.sum(np.abs(harmonic_weights), axis=1, keepdims=True)
            harmonic_weights = harmonic_weights / (row_sums + 1e-10)
        
        return harmonic_weights
    
    def _normalize_to_unit(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to [0, 1] range.
        
        Args:
            values: Input array
            
        Returns:
            Normalized array in [0, 1]
        """
        v_min, v_max = values.min(), values.max()
        
        if v_max - v_min < 1e-10:
            return np.ones_like(values) * 0.5
        
        return (values - v_min) / (v_max - v_min)
    
    def _quantize_to_scale(
        self,
        frequencies: np.ndarray,
        scale: Scale,
        root_note: str
    ) -> np.ndarray:
        """
        Quantize frequencies to nearest scale notes.
        
        Args:
            frequencies: Array of frequencies in Hz
            scale: Musical scale
            root_note: Root note
            
        Returns:
            Quantized frequencies
        """
        root_freq = self._note_to_frequency(root_note)
        intervals = SCALE_INTERVALS[scale]
        
        quantized = np.zeros_like(frequencies)
        
        for i, freq in enumerate(frequencies):
            # Convert to semitones from root
            semitones = 12 * np.log2(freq / root_freq)
            
            # Find nearest scale note
            octave = int(semitones // 12)
            note_in_octave = semitones % 12
            
            # Find closest interval
            closest_interval = min(intervals, key=lambda x: abs(x - note_in_octave))
            
            # Convert back to frequency
            quantized_semitones = octave * 12 + closest_interval
            quantized[i] = root_freq * (2 ** (quantized_semitones / 12))
        
        return quantized
    
    def _note_to_midi(self, note: str) -> int:
        """
        Convert note name to MIDI number.
        
        Args:
            note: Note name (e.g., "C4", "A#3", "Bb4")
            
        Returns:
            MIDI note number
        """
        note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        
        # Parse note
        note_name = note[0].upper()
        octave_str = note[-1]
        
        # Handle accidentals
        if len(note) == 3:
            accidental = note[1]
            if accidental == '#':
                offset = 1
            elif accidental == 'b':
                offset = -1
            else:
                offset = 0
        else:
            offset = 0
        
        octave = int(octave_str)
        midi = (octave + 1) * 12 + note_map[note_name] + offset
        
        return midi
    
    def _note_to_frequency(self, note: str) -> float:
        """
        Convert note name to frequency.
        
        Args:
            note: Note name (e.g., "A4", "C3")
            
        Returns:
            Frequency in Hz
        """
        midi = self._note_to_midi(note)
        # A4 (MIDI 69) = 440 Hz
        return 440.0 * (2 ** ((midi - 69) / 12))
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get mapping engine metadata.
        
        Returns:
            Dictionary containing mapping configuration
        """
        return {
            'freq_min': self.freq_min,
            'freq_max': self.freq_max,
            'sample_rate': self.sample_rate,
            'midi_range': (self.midi_min, self.midi_max)
        }

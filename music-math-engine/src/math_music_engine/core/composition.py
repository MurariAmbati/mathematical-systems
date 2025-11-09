"""
Composition layer for managing multiple mathematical streams.

Combines multiple mathematical functions into polyphonic structures.
"""

from typing import List, Callable, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum


class CompositionMode(Enum):
    """Modes for combining multiple signals."""
    SUM = "sum"
    MULTIPLY = "multiply"
    MODULATE = "modulate"
    INTERLEAVE = "interleave"


class Voice:
    """
    Represents a single voice in a composition.
    """
    
    def __init__(
        self,
        signal: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        amplitude: float = 1.0
    ):
        """
        Initialize a voice.
        
        Args:
            signal: Audio signal array
            metadata: Optional metadata about the voice
            amplitude: Voice amplitude/gain
        """
        self.signal = np.asarray(signal, dtype=np.float64)
        self.metadata = metadata or {}
        self.amplitude = amplitude
        
    def apply_gain(self, gain: float) -> 'Voice':
        """
        Apply gain to the voice.
        
        Args:
            gain: Gain factor
            
        Returns:
            New Voice with gain applied
        """
        return Voice(
            self.signal * gain,
            self.metadata.copy(),
            self.amplitude * gain
        )
    
    def apply_pan(self, pan: float, stereo: bool = True) -> np.ndarray:
        """
        Apply stereo panning.
        
        Args:
            pan: Pan position (-1 = left, 0 = center, 1 = right)
            stereo: Whether to create stereo output
            
        Returns:
            Mono or stereo signal
        """
        if not stereo:
            return self.signal
        
        # Convert pan to left/right gains
        # Pan law: constant power panning
        pan = np.clip(pan, -1, 1)
        angle = (pan + 1) * np.pi / 4  # Map [-1, 1] to [0, pi/2]
        
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)
        
        # Create stereo signal
        stereo_signal = np.column_stack([
            self.signal * left_gain,
            self.signal * right_gain
        ])
        
        return stereo_signal


class Composition:
    """
    Manages multiple voices and their combination.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize composition.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.voices: List[Voice] = []
        self.metadata: Dict[str, Any] = {
            'sample_rate': sample_rate,
            'num_voices': 0
        }
        
    def add_voice(
        self,
        signal: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        amplitude: float = 1.0
    ) -> int:
        """
        Add a voice to the composition.
        
        Args:
            signal: Audio signal
            metadata: Optional voice metadata
            amplitude: Voice amplitude
            
        Returns:
            Voice index
        """
        voice = Voice(signal, metadata, amplitude)
        self.voices.append(voice)
        self.metadata['num_voices'] = len(self.voices)
        return len(self.voices) - 1
    
    def remove_voice(self, index: int):
        """
        Remove a voice from the composition.
        
        Args:
            index: Voice index
        """
        if 0 <= index < len(self.voices):
            self.voices.pop(index)
            self.metadata['num_voices'] = len(self.voices)
    
    def mix(
        self,
        mode: CompositionMode = CompositionMode.SUM,
        normalize: bool = True,
        stereo: bool = False,
        pan_positions: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Mix all voices together.
        
        Args:
            mode: Composition mode (how to combine voices)
            normalize: Whether to normalize the output
            stereo: Whether to create stereo output
            pan_positions: Pan position for each voice (if stereo)
            
        Returns:
            Mixed signal (mono or stereo)
        """
        if not self.voices:
            return np.array([])
        
        # Determine maximum length
        max_length = max(len(v.signal) for v in self.voices)
        
        # Prepare pan positions
        if stereo and pan_positions is None:
            # Auto-distribute voices across stereo field
            if len(self.voices) == 1:
                pan_positions = [0.0]
            else:
                pan_positions = np.linspace(-1, 1, len(self.voices)).tolist()
        elif not stereo:
            pan_positions = [0.0] * len(self.voices)
        
        if mode == CompositionMode.SUM:
            mixed = self._mix_sum(max_length, stereo, pan_positions)
            
        elif mode == CompositionMode.MULTIPLY:
            mixed = self._mix_multiply(max_length, stereo, pan_positions)
            
        elif mode == CompositionMode.MODULATE:
            mixed = self._mix_modulate(max_length, stereo, pan_positions)
            
        elif mode == CompositionMode.INTERLEAVE:
            mixed = self._mix_interleave(stereo, pan_positions)
            
        else:
            raise ValueError(f"Unknown composition mode: {mode}")
        
        # Normalize if requested
        if normalize:
            max_val = np.max(np.abs(mixed))
            if max_val > 0:
                mixed /= max_val
        
        return mixed
    
    def _mix_sum(
        self,
        max_length: int,
        stereo: bool,
        pan_positions: List[float]
    ) -> np.ndarray:
        """Mix voices by summing."""
        if stereo:
            mixed = np.zeros((max_length, 2))
            for voice, pan in zip(self.voices, pan_positions):
                stereo_signal = voice.apply_pan(pan, stereo=True)
                # Pad if necessary
                if len(stereo_signal) < max_length:
                    padding = max_length - len(stereo_signal)
                    stereo_signal = np.vstack([
                        stereo_signal,
                        np.zeros((padding, 2))
                    ])
                mixed += stereo_signal * voice.amplitude
        else:
            mixed = np.zeros(max_length)
            for voice in self.voices:
                signal = voice.signal
                if len(signal) < max_length:
                    signal = np.pad(signal, (0, max_length - len(signal)))
                mixed += signal * voice.amplitude
        
        return mixed
    
    def _mix_multiply(
        self,
        max_length: int,
        stereo: bool,
        pan_positions: List[float]
    ) -> np.ndarray:
        """Mix voices by multiplication (ring modulation)."""
        if stereo:
            mixed = np.ones((max_length, 2))
            for voice, pan in zip(self.voices, pan_positions):
                stereo_signal = voice.apply_pan(pan, stereo=True)
                if len(stereo_signal) < max_length:
                    padding = max_length - len(stereo_signal)
                    stereo_signal = np.vstack([
                        stereo_signal,
                        np.ones((padding, 2))
                    ])
                mixed *= stereo_signal
        else:
            mixed = np.ones(max_length)
            for voice in self.voices:
                signal = voice.signal
                if len(signal) < max_length:
                    signal = np.pad(signal, (0, max_length - len(signal)), constant_values=1)
                mixed *= signal
        
        return mixed
    
    def _mix_modulate(
        self,
        max_length: int,
        stereo: bool,
        pan_positions: List[float]
    ) -> np.ndarray:
        """Mix voices with first voice as carrier, others as modulators."""
        if not self.voices:
            return np.array([])
        
        carrier = self.voices[0]
        
        if stereo:
            mixed = carrier.apply_pan(pan_positions[0], stereo=True)
            if len(mixed) < max_length:
                padding = max_length - len(mixed)
                mixed = np.vstack([mixed, np.zeros((padding, 2))])
            
            for voice, pan in zip(self.voices[1:], pan_positions[1:]):
                mod_signal = voice.apply_pan(pan, stereo=True)
                if len(mod_signal) < max_length:
                    padding = max_length - len(mod_signal)
                    mod_signal = np.vstack([mod_signal, np.zeros((padding, 2))])
                # Apply modulation
                mixed = mixed * (1 + mod_signal * voice.amplitude)
        else:
            mixed = carrier.signal.copy()
            if len(mixed) < max_length:
                mixed = np.pad(mixed, (0, max_length - len(mixed)))
            
            for voice in self.voices[1:]:
                signal = voice.signal
                if len(signal) < max_length:
                    signal = np.pad(signal, (0, max_length - len(signal)))
                # Apply modulation
                mixed = mixed * (1 + signal * voice.amplitude)
        
        return mixed
    
    def _mix_interleave(
        self,
        stereo: bool,
        pan_positions: List[float]
    ) -> np.ndarray:
        """Mix voices by interleaving (concatenation)."""
        if stereo:
            segments = []
            for voice, pan in zip(self.voices, pan_positions):
                stereo_signal = voice.apply_pan(pan, stereo=True)
                segments.append(stereo_signal)
            mixed = np.vstack(segments)
        else:
            segments = [voice.signal for voice in self.voices]
            mixed = np.concatenate(segments)
        
        return mixed
    
    def apply_temporal_alignment(
        self,
        delays: List[float],
        crossfade: float = 0.0
    ):
        """
        Apply time delays to voices for temporal alignment.
        
        Args:
            delays: List of delays in seconds for each voice
            crossfade: Crossfade duration in seconds
        """
        if len(delays) != len(self.voices):
            raise ValueError("Number of delays must match number of voices")
        
        for i, (voice, delay) in enumerate(zip(self.voices, delays)):
            if delay > 0:
                # Add silence at beginning
                delay_samples = int(delay * self.sample_rate)
                padded_signal = np.pad(voice.signal, (delay_samples, 0))
                
                # Apply crossfade if specified
                if crossfade > 0:
                    fade_samples = int(crossfade * self.sample_rate)
                    fade_in = np.linspace(0, 1, min(fade_samples, len(voice.signal)))
                    padded_signal[delay_samples:delay_samples + len(fade_in)] *= fade_in
                
                self.voices[i].signal = padded_signal
    
    def apply_function_composition(
        self,
        modulator_idx: int,
        carrier_idx: int,
        composition_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> int:
        """
        Apply function composition between two voices.
        
        Args:
            modulator_idx: Index of modulator voice
            carrier_idx: Index of carrier voice
            composition_func: Function to combine signals (mod, carrier) -> result
            
        Returns:
            Index of new composed voice
        """
        if modulator_idx >= len(self.voices) or carrier_idx >= len(self.voices):
            raise ValueError("Invalid voice index")
        
        modulator = self.voices[modulator_idx].signal
        carrier = self.voices[carrier_idx].signal
        
        # Match lengths
        min_length = min(len(modulator), len(carrier))
        modulator = modulator[:min_length]
        carrier = carrier[:min_length]
        
        # Apply composition
        composed = composition_func(modulator, carrier)
        
        # Create metadata
        metadata = {
            'composition': {
                'modulator': self.voices[modulator_idx].metadata,
                'carrier': self.voices[carrier_idx].metadata
            }
        }
        
        return self.add_voice(composed, metadata)
    
    def get_voice(self, index: int) -> Voice:
        """
        Get a voice by index.
        
        Args:
            index: Voice index
            
        Returns:
            Voice object
        """
        if 0 <= index < len(self.voices):
            return self.voices[index]
        raise IndexError(f"Voice index {index} out of range")
    
    def clear(self):
        """Clear all voices."""
        self.voices.clear()
        self.metadata['num_voices'] = 0

"""
Signal synthesis layer for generating audio waveforms.

Provides oscillators, envelopes, and synthesis chains for audio generation.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from enum import Enum


class WaveformType(Enum):
    """Types of basic waveforms."""
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"
    NOISE = "noise"


class Oscillator:
    """
    Basic oscillator for waveform generation.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize oscillator.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.phase = 0.0
        
    def generate(
        self,
        waveform: WaveformType,
        frequency: np.ndarray,
        duration: Optional[float] = None,
        num_samples: Optional[int] = None,
        phase_offset: float = 0.0
    ) -> np.ndarray:
        """
        Generate a waveform.
        
        Args:
            waveform: Type of waveform to generate
            frequency: Frequency in Hz (can be array for frequency modulation)
            duration: Duration in seconds (alternative to num_samples)
            num_samples: Number of samples (alternative to duration)
            phase_offset: Initial phase offset in radians
            
        Returns:
            Generated waveform array
        """
        # Determine number of samples
        if num_samples is None and duration is None:
            raise ValueError("Must specify either duration or num_samples")
        if num_samples is None:
            num_samples = int(duration * self.sample_rate)
        
        # Handle constant or varying frequency
        if isinstance(frequency, (int, float)):
            frequency = np.full(num_samples, frequency, dtype=np.float64)
        else:
            frequency = np.asarray(frequency, dtype=np.float64)
            if len(frequency) != num_samples:
                # Interpolate to match num_samples
                frequency = np.interp(
                    np.linspace(0, len(frequency) - 1, num_samples),
                    np.arange(len(frequency)),
                    frequency
                )
        
        # Generate time-varying phase
        phase_increment = 2 * np.pi * frequency / self.sample_rate
        phase = np.cumsum(phase_increment) + phase_offset + self.phase
        
        # Generate waveform
        if waveform == WaveformType.SINE:
            signal = np.sin(phase)
            
        elif waveform == WaveformType.SQUARE:
            signal = np.sign(np.sin(phase))
            
        elif waveform == WaveformType.SAWTOOTH:
            signal = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
            
        elif waveform == WaveformType.TRIANGLE:
            sawtooth = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
            signal = 2 * np.abs(sawtooth) - 1
            
        elif waveform == WaveformType.NOISE:
            signal = np.random.uniform(-1, 1, num_samples)
            
        else:
            raise ValueError(f"Unknown waveform type: {waveform}")
        
        # Update phase for continuous generation
        self.phase = phase[-1] % (2 * np.pi)
        
        return signal
    
    def reset_phase(self):
        """Reset the oscillator phase to zero."""
        self.phase = 0.0


class ADSR:
    """
    ADSR (Attack, Decay, Sustain, Release) envelope generator.
    """
    
    def __init__(
        self,
        attack: float = 0.01,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.2,
        sample_rate: int = 44100
    ):
        """
        Initialize ADSR envelope.
        
        Args:
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0-1)
            release: Release time in seconds
            sample_rate: Audio sample rate
        """
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.sample_rate = sample_rate
        
    def generate(self, duration: float, gate_duration: Optional[float] = None) -> np.ndarray:
        """
        Generate ADSR envelope.
        
        Args:
            duration: Total duration in seconds
            gate_duration: Duration before release (if None, no release)
            
        Returns:
            Envelope array
        """
        num_samples = int(duration * self.sample_rate)
        envelope = np.zeros(num_samples)
        
        # Convert times to samples
        attack_samples = int(self.attack * self.sample_rate)
        decay_samples = int(self.decay * self.sample_rate)
        release_samples = int(self.release * self.sample_rate)
        
        if gate_duration is None:
            gate_duration = duration
        gate_samples = int(gate_duration * self.sample_rate)
        
        current_sample = 0
        
        # Attack phase
        attack_end = min(attack_samples, num_samples)
        if attack_samples > 0:
            envelope[current_sample:attack_end] = np.linspace(0, 1, attack_end - current_sample)
        current_sample = attack_end
        
        # Decay phase
        decay_end = min(current_sample + decay_samples, num_samples, gate_samples)
        if decay_samples > 0 and current_sample < decay_end:
            envelope[current_sample:decay_end] = np.linspace(1, self.sustain, decay_end - current_sample)
        current_sample = decay_end
        
        # Sustain phase
        sustain_end = min(gate_samples, num_samples)
        if current_sample < sustain_end:
            envelope[current_sample:sustain_end] = self.sustain
        current_sample = sustain_end
        
        # Release phase
        release_end = min(current_sample + release_samples, num_samples)
        if release_samples > 0 and current_sample < release_end:
            envelope[current_sample:release_end] = np.linspace(
                self.sustain, 0, release_end - current_sample
            )
        
        return envelope


class AdditiveSynthesizer:
    """
    Additive synthesis using multiple harmonics.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize additive synthesizer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.oscillator = Oscillator(sample_rate)
        
    def synthesize(
        self,
        fundamental_freq: float,
        harmonics: List[float],
        duration: float,
        envelope: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize using additive synthesis.
        
        Args:
            fundamental_freq: Fundamental frequency in Hz
            harmonics: List of harmonic amplitudes (relative to fundamental)
            duration: Duration in seconds
            envelope: Optional envelope to apply
            
        Returns:
            Synthesized signal
        """
        num_samples = int(duration * self.sample_rate)
        signal = np.zeros(num_samples)
        
        # Generate each harmonic
        for i, amplitude in enumerate(harmonics):
            if amplitude > 0:
                harmonic_freq = fundamental_freq * (i + 1)
                self.oscillator.reset_phase()
                harmonic_signal = self.oscillator.generate(
                    WaveformType.SINE,
                    harmonic_freq,
                    num_samples=num_samples
                )
                signal += amplitude * harmonic_signal
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
        
        # Apply envelope
        if envelope is not None:
            if len(envelope) == len(signal):
                signal *= envelope
            else:
                # Interpolate envelope to match signal length
                envelope_interp = np.interp(
                    np.linspace(0, len(envelope) - 1, len(signal)),
                    np.arange(len(envelope)),
                    envelope
                )
                signal *= envelope_interp
        
        return signal


class FrequencyModulation:
    """
    Frequency modulation (FM) synthesis.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize FM synthesizer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.carrier_osc = Oscillator(sample_rate)
        self.modulator_osc = Oscillator(sample_rate)
        
    def synthesize(
        self,
        carrier_freq: float,
        modulator_freq: float,
        modulation_index: float,
        duration: float,
        envelope: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize using FM synthesis.
        
        Args:
            carrier_freq: Carrier frequency in Hz
            modulator_freq: Modulator frequency in Hz
            modulation_index: Modulation index (depth)
            duration: Duration in seconds
            envelope: Optional envelope to apply
            
        Returns:
            Synthesized signal
        """
        num_samples = int(duration * self.sample_rate)
        
        # Generate modulator
        self.modulator_osc.reset_phase()
        modulator = self.modulator_osc.generate(
            WaveformType.SINE,
            modulator_freq,
            num_samples=num_samples
        )
        
        # Modulate carrier frequency
        modulated_freq = carrier_freq + (modulation_index * modulator_freq * modulator)
        
        # Generate carrier with modulated frequency
        self.carrier_osc.reset_phase()
        signal = self.carrier_osc.generate(
            WaveformType.SINE,
            modulated_freq,
            num_samples=num_samples
        )
        
        # Apply envelope
        if envelope is not None:
            if len(envelope) == len(signal):
                signal *= envelope
            else:
                envelope_interp = np.interp(
                    np.linspace(0, len(envelope) - 1, len(signal)),
                    np.arange(len(envelope)),
                    envelope
                )
                signal *= envelope_interp
        
        return signal


class AmplitudeModulation:
    """
    Amplitude modulation (AM) synthesis.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize AM synthesizer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.carrier_osc = Oscillator(sample_rate)
        self.modulator_osc = Oscillator(sample_rate)
        
    def synthesize(
        self,
        carrier_freq: float,
        modulator_freq: float,
        modulation_depth: float,
        duration: float,
        envelope: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize using AM synthesis.
        
        Args:
            carrier_freq: Carrier frequency in Hz
            modulator_freq: Modulator frequency in Hz
            modulation_depth: Modulation depth (0-1)
            duration: Duration in seconds
            envelope: Optional envelope to apply
            
        Returns:
            Synthesized signal
        """
        num_samples = int(duration * self.sample_rate)
        
        # Generate carrier
        self.carrier_osc.reset_phase()
        carrier = self.carrier_osc.generate(
            WaveformType.SINE,
            carrier_freq,
            num_samples=num_samples
        )
        
        # Generate modulator
        self.modulator_osc.reset_phase()
        modulator = self.modulator_osc.generate(
            WaveformType.SINE,
            modulator_freq,
            num_samples=num_samples
        )
        
        # Apply amplitude modulation
        # AM formula: carrier * (1 + depth * modulator)
        signal = carrier * (1 + modulation_depth * modulator)
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
        
        # Apply envelope
        if envelope is not None:
            if len(envelope) == len(signal):
                signal *= envelope
            else:
                envelope_interp = np.interp(
                    np.linspace(0, len(envelope) - 1, len(signal)),
                    np.arange(len(envelope)),
                    envelope
                )
                signal *= envelope_interp
        
        return signal


class Filter:
    """
    Simple digital filters.
    """
    
    @staticmethod
    def lowpass(signal: np.ndarray, cutoff: float, sample_rate: int) -> np.ndarray:
        """
        Apply simple lowpass filter.
        
        Args:
            signal: Input signal
            cutoff: Cutoff frequency in Hz
            sample_rate: Sample rate
            
        Returns:
            Filtered signal
        """
        from scipy import signal as sp_signal
        
        # Design lowpass filter
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = sp_signal.butter(4, normalized_cutoff, btype='low')
        
        # Apply filter
        filtered = sp_signal.filtfilt(b, a, signal)
        
        return filtered
    
    @staticmethod
    def highpass(signal: np.ndarray, cutoff: float, sample_rate: int) -> np.ndarray:
        """
        Apply simple highpass filter.
        
        Args:
            signal: Input signal
            cutoff: Cutoff frequency in Hz
            sample_rate: Sample rate
            
        Returns:
            Filtered signal
        """
        from scipy import signal as sp_signal
        
        # Design highpass filter
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = sp_signal.butter(4, normalized_cutoff, btype='high')
        
        # Apply filter
        filtered = sp_signal.filtfilt(b, a, signal)
        
        return filtered

"""
Advanced Demo: Multiple generators with visualization.
"""

import numpy as np
from math_music_engine.generators import (
    FourierSynth, 
    FractalMelody, 
    PrimeSequence,
    RandomWalk
)
from math_music_engine.core import (
    MappingEngine,
    MappingMode,
    Scale,
    Oscillator,
    WaveformType,
    OutputManager
)

print("Math-Music-Engine: Advanced Demo")
print("=" * 50)

output_mgr = OutputManager()
mapper = MappingEngine()

# 1. Fractal Melody
print("\n1. Generating fractal melody (Koch curve)...")
fractal = FractalMelody()
intervals = fractal.generate_koch_curve_melody(iterations=3, intervals=[0, 2, 4, 5, 7, 9, 11])
notes = fractal.intervals_to_notes(intervals, base_note=60)  # C4 = MIDI 60
print(f"   Generated {len(notes)} notes from fractal pattern")

# Convert MIDI notes to frequencies
freqs = [440.0 * (2 ** ((note - 69) / 12)) for note in notes]
print(f"   Frequency range: {min(freqs):.1f} - {max(freqs):.1f} Hz")

# Create audio by playing each note
sample_rate = 44100
note_duration = 0.2
osc = Oscillator(sample_rate=sample_rate)

signals = []
for freq in freqs:
    note_signal = osc.generate(WaveformType.SINE, freq, duration=note_duration)
    signals.append(note_signal)

fractal_audio = np.concatenate(signals)
output_mgr.export_audio(fractal_audio, "fractal_melody.wav", sample_rate=sample_rate)
print("   ✓ Saved: fractal_melody.wav")

# 2. Prime Sequence Melody
print("\n2. Generating prime sequence melody...")
prime_gen = PrimeSequence()
prime_notes = prime_gen.prime_melody(num_notes=20, base_note=48, method='gaps')
print(f"   Generated {len(prime_notes)} notes from prime gaps")

prime_freqs = [440.0 * (2 ** ((note - 69) / 12)) for note in prime_notes]
prime_signals = []
for freq in prime_freqs:
    note_signal = osc.generate(WaveformType.TRIANGLE, freq, duration=0.15)
    prime_signals.append(note_signal)

prime_audio = np.concatenate(prime_signals)
output_mgr.export_audio(prime_audio, "prime_melody.wav", sample_rate=sample_rate)
print("   ✓ Saved: prime_melody.wav")

# 3. Random Walk with Scale Quantization
print("\n3. Generating random walk melody (Brownian motion)...")
walk_gen = RandomWalk(sample_rate=sample_rate, seed=42)
walk_signal = walk_gen.brownian_motion(duration=10.0, step_size=0.1, bias=0.0)

# Sample values at regular intervals to create notes
num_notes = 30
note_indices = np.linspace(0, len(walk_signal) - 1, num_notes, dtype=int)
walk_values = walk_signal[note_indices]

# Normalize to [0, 1]
walk_values = (walk_values - walk_values.min()) / (walk_values.max() - walk_values.min())

# Map to MIDI notes with pentatonic scale
walk_notes = mapper.map_to_midi(
    walk_values,
    scale=Scale.PENTATONIC_MINOR,
    root_note='A3',
    octave_range=3
)
print(f"   Generated {len(walk_notes)} notes from random walk")

walk_freqs = [440.0 * (2 ** ((note - 69) / 12)) for note in walk_notes]
walk_signals = []
for freq in walk_freqs:
    note_signal = osc.generate(WaveformType.SAWTOOTH, freq, duration=0.25)
    walk_signals.append(note_signal)

walk_audio = np.concatenate(walk_signals)
output_mgr.export_audio(walk_audio, "random_walk_melody.wav", sample_rate=sample_rate)
print("   ✓ Saved: random_walk_melody.wav")

# 4. Fourier Synthesis with Rich Harmonics
print("\n4. Generating rich harmonic texture...")
# Create harmonic series that decays exponentially
harmonics = [1.0 / (n + 1) for n in range(16)]
fourier = FourierSynth(harmonics=harmonics, sample_rate=sample_rate)
fourier_audio = fourier.generate(fundamental_freq=110.0, duration=8.0)
output_mgr.export_audio(fourier_audio, "rich_harmonics.wav", sample_rate=sample_rate)
print(f"   ✓ Saved: rich_harmonics.wav (16 harmonics)")

print("\n" + "=" * 50)
print("✓ Advanced demo complete!")
print("\nGenerated files:")
print("  - fractal_melody.wav (Koch curve)")
print("  - prime_melody.wav (prime number gaps)")
print("  - random_walk_melody.wav (Brownian motion)")
print("  - rich_harmonics.wav (16 harmonic partials)")
print("=" * 50)

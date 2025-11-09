"""
Simple Demo: Generate a musical tone from a sine wave.

This is the simplest possible example - generates a 440 Hz tone.
"""

import numpy as np
from math_music_engine.core import Oscillator, WaveformType, OutputManager, ADSR

print("Math-Music-Engine: Simple Demo")
print("=" * 40)

# Create oscillator
osc = Oscillator(sample_rate=44100)

# Generate a 440 Hz sine wave (A4 note) for 2 seconds
print("Generating 440 Hz tone (A4)...")
signal = osc.generate(
    waveform=WaveformType.SINE,
    frequency=440.0,
    duration=2.0
)

# Apply an ADSR envelope to make it more musical
print("Applying ADSR envelope...")
adsr = ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
envelope = adsr.generate(duration=2.0, gate_duration=1.5)
signal = signal * envelope

# Export to WAV file
print("Exporting to simple_demo.wav...")
output_mgr = OutputManager()
output_mgr.export_audio(signal, "simple_demo.wav", sample_rate=44100)

print("✓ Complete! Generated: simple_demo.wav")
print("  Duration: 2 seconds")
print("  Frequency: 440 Hz (A4)")
print("  Waveform: Sine")

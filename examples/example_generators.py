"""
Example: Using generators.
"""

from math_music_engine.generators import FourierSynth, FibonacciRhythm, ChaosModulator
from math_music_engine.core import OutputManager, Composition, CompositionMode

# Create output manager
output_mgr = OutputManager()

print("=" * 50)
print("Math-Music-Engine: Generator Examples")
print("=" * 50)

# 1. Fourier Synth
print("\n1. Generating Fourier synthesis (harmonic series)...")
fourier = FourierSynth(harmonics=[1.0, 0.5, 0.25, 0.125], sample_rate=44100)
fourier_signal = fourier.generate(fundamental_freq=220.0, duration=5.0)
output_mgr.export_audio(fourier_signal, "fourier.wav")
print("   ✓ Saved: fourier.wav")

# 2. Fibonacci Rhythm
print("\n2. Generating Fibonacci rhythm pattern...")
fib_rhythm = FibonacciRhythm(sample_rate=44100)
pattern = fib_rhythm.generate_rhythm_pattern(num_beats=8, subdivision=16)
rhythm_signal = fib_rhythm.generate_pulse_train(pattern, tempo=120, duration=5.0)
output_mgr.export_audio(rhythm_signal, "fibonacci_rhythm.wav")
print("   ✓ Saved: fibonacci_rhythm.wav")

# 3. Chaos (Lorenz)
print("\n3. Generating chaotic modulation (Lorenz attractor)...")
chaos = ChaosModulator(sample_rate=44100)
chaos_signal = chaos.generate('lorenz', duration=5.0, sigma=10, rho=28, beta=8/3)
output_mgr.export_audio(chaos_signal, "chaos_lorenz.wav")
print("   ✓ Saved: chaos_lorenz.wav")

# Compose them together
print("\n4. Composing all signals together...")
comp = Composition(sample_rate=44100)
comp.add_voice(fourier_signal, amplitude=0.5)
comp.add_voice(rhythm_signal, amplitude=0.3)
comp.add_voice(chaos_signal, amplitude=0.2)

mixed = comp.mix(mode=CompositionMode.SUM, normalize=True)
output_mgr.export_audio(mixed, "composition.wav")
print("   ✓ Saved: composition.wav")

print("\n" + "=" * 50)
print("✓ All examples generated successfully!")
print("=" * 50)

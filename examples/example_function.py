"""
Example: Generate audio from mathematical function.
"""

import numpy as np
from math_music_engine.parser import expression_to_function
from math_music_engine.core import FunctionEngine, MappingEngine, MappingMode, Oscillator, WaveformType, OutputManager
from math_music_engine.visualization import plot_waveform, plot_frequency_spectrum

# Parse mathematical expression
def func(t):
    """Simple sinusoidal function"""
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

# Create engines
func_engine = FunctionEngine(sample_rate=44100)
mapper = MappingEngine(freq_min=220, freq_max=880)

# Sample the function - this gives us amplitude values over time
print("Sampling function...")
values = func_engine.uniform_sample(func, duration=5.0, normalize=True)

# Map values to frequencies (for frequency modulation)
print("Mapping to frequencies...")
frequencies = mapper.map_to_frequency(values, mode=MappingMode.LOGARITHMIC)

# Generate audio with frequency modulation
print("Generating audio...")
osc = Oscillator(sample_rate=44100)
signal = osc.generate(WaveformType.SINE, frequencies)

# Export
print("Exporting audio...")
output_mgr = OutputManager()
output_mgr.export_audio(signal, "example_function.wav")

# Create metadata
metadata = output_mgr.create_reproducibility_metadata(
    expression="sin(2*pi*t) + 0.5*sin(2*pi*3*t)",
    parameters={'duration': 5.0, 'sample_rate': 44100},
    mappings={'mode': 'logarithmic', 'freq_min': 220, 'freq_max': 880}
)
output_mgr.export_metadata(metadata, "example_function_metadata.json")

# Visualize
print("Creating visualizations...")
try:
    plot_waveform(signal, 44100, duration=1.0, output_path="example_waveform.png", show=False)
    plot_frequency_spectrum(signal, 44100, output_path="example_spectrum.png", show=False)
    print("✓ Visualizations saved!")
except Exception as e:
    print(f"Note: Visualization skipped ({e})")

print("✓ Complete! Generated: example_function.wav")

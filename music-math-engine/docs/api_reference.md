# API Reference

## Parser Module

### `ExpressionParser`

Parse mathematical expressions and convert to executable functions.

```python
from math_music_engine.parser import ExpressionParser

parser = ExpressionParser()
```

#### Methods

##### `parse(expression, variables=None)`
Parse a mathematical expression string.

**Parameters:**
- `expression` (str): Mathematical expression (e.g., "sin(2*pi*t)")
- `variables` (list[str], optional): Variable names

**Returns:**
- tuple: (SymPy expression, metadata dict)

**Example:**
```python
expr, meta = parser.parse("cos(2*pi*440*t)")
print(meta['variables'])  # ['t']
print(meta['contains_trig'])  # True
```

##### `to_numpy_function(expression, variables=None, use_cache=True)`
Convert expression to NumPy-compatible function.

**Returns:**
- tuple: (callable function, metadata dict)

**Example:**
```python
func, meta = parser.to_numpy_function("sin(t)")
import numpy as np
result = func(np.array([0, np.pi/2, np.pi]))
```

---

## Core Module

### `FunctionEngine`

Sample mathematical functions over time.

```python
from math_music_engine.core import FunctionEngine

engine = FunctionEngine(sample_rate=44100)
```

#### Methods

##### `uniform_sample(func, duration, *args, normalize=True, **kwargs)`
Sample function uniformly.

**Parameters:**
- `func` (callable): Function to sample
- `duration` (float): Duration in seconds
- `normalize` (bool): Normalize to [-1, 1]

**Returns:**
- np.ndarray: Sampled values

##### `adaptive_sample(func, duration, tolerance=0.01, ...)`
Sample function adaptively based on curvature.

**Example:**
```python
import numpy as np

def my_func(t):
    return np.sin(2 * np.pi * 440 * t)

signal = engine.uniform_sample(my_func, duration=1.0)
```

---

### `MappingEngine`

Map mathematical outputs to musical parameters.

```python
from math_music_engine.core import MappingEngine, MappingMode, Scale

mapper = MappingEngine(freq_min=220, freq_max=880)
```

#### Methods

##### `map_to_frequency(values, mode, freq_min=None, freq_max=None, scale=None, root_note='A4')`
Map values to frequencies.

**Parameters:**
- `values` (np.ndarray): Input values
- `mode` (MappingMode): Mapping mode
- `scale` (Scale, optional): Musical scale for quantization
- `root_note` (str): Root note (e.g., "C4")

**Returns:**
- np.ndarray: Frequencies in Hz

**Example:**
```python
values = np.linspace(0, 1, 100)
freqs = mapper.map_to_frequency(
    values,
    mode=MappingMode.LOGARITHMIC,
    scale=Scale.MAJOR,
    root_note="C4"
)
```

##### `map_to_amplitude(values, mode=MappingMode.LINEAR, ...)`
Map values to amplitudes [0, 1].

##### `map_to_midi(values, scale=None, root_note='C4', octave_range=4)`
Map values to MIDI note numbers.

---

### `Oscillator`

Generate basic waveforms.

```python
from math_music_engine.core import Oscillator, WaveformType

osc = Oscillator(sample_rate=44100)
```

#### Methods

##### `generate(waveform, frequency, duration=None, num_samples=None, phase_offset=0)`
Generate waveform.

**Parameters:**
- `waveform` (WaveformType): Type of waveform
- `frequency` (float or np.ndarray): Frequency in Hz
- `duration` (float): Duration in seconds
- `phase_offset` (float): Initial phase in radians

**Example:**
```python
signal = osc.generate(
    WaveformType.SINE,
    440.0,
    duration=1.0
)
```

**Waveform Types:**
- `WaveformType.SINE`
- `WaveformType.SQUARE`
- `WaveformType.SAWTOOTH`
- `WaveformType.TRIANGLE`
- `WaveformType.NOISE`

---

### `ADSR`

ADSR envelope generator.

```python
from math_music_engine.core import ADSR

adsr = ADSR(
    attack=0.01,
    decay=0.1,
    sustain=0.7,
    release=0.2
)
```

#### Methods

##### `generate(duration, gate_duration=None)`
Generate ADSR envelope.

**Example:**
```python
envelope = adsr.generate(duration=1.0, gate_duration=0.8)
signal = audio * envelope  # Apply envelope
```

---

### `Composition`

Manage multiple voices.

```python
from math_music_engine.core import Composition, CompositionMode

comp = Composition(sample_rate=44100)
```

#### Methods

##### `add_voice(signal, metadata=None, amplitude=1.0)`
Add a voice to the composition.

**Returns:**
- int: Voice index

##### `mix(mode=CompositionMode.SUM, normalize=True, stereo=False, pan_positions=None)`
Mix all voices together.

**Modes:**
- `CompositionMode.SUM`: Add voices
- `CompositionMode.MULTIPLY`: Multiply voices
- `CompositionMode.MODULATE`: Modulate carrier by others
- `CompositionMode.INTERLEAVE`: Concatenate voices

**Example:**
```python
comp.add_voice(signal1)
comp.add_voice(signal2)
mixed = comp.mix(mode=CompositionMode.SUM, normalize=True)
```

---

### `OutputManager`

Export audio and metadata.

```python
from math_music_engine.core import OutputManager

output = OutputManager(output_dir="./output")
```

#### Methods

##### `export_audio(signal, filename, sample_rate=44100, bit_depth=16, metadata=None)`
Export audio to WAV file.

**Example:**
```python
output.export_audio(
    signal,
    "output.wav",
    sample_rate=44100,
    metadata={'description': 'Test audio'}
)
```

##### `export_midi(notes, filename, tempo=120, metadata=None)`
Export MIDI file.

**Parameters:**
- `notes` (list[dict]): Note dictionaries with keys:
  - `'note'`: MIDI note number
  - `'start'`: Start time in seconds
  - `'duration'`: Duration in seconds
  - `'velocity'`: Velocity (0-127)

##### `export_metadata(metadata, filename)`
Export metadata JSON.

---

## Generators Module

### `FourierSynth`

Fourier synthesis generator.

```python
from math_music_engine.generators import FourierSynth

synth = FourierSynth(harmonics=[1.0, 0.5, 0.25, 0.125])
```

#### Methods

##### `generate(fundamental_freq, duration, phase_offsets=None)`
Generate audio using Fourier synthesis.

**Example:**
```python
signal = synth.generate(fundamental_freq=440.0, duration=2.0)
```

##### `set_harmonics_from_formula(formula, num_harmonics=16)`
Set harmonics from formula.

**Formulas:**
- `"sawtooth"` or `"1/n"`: Sawtooth wave
- `"square"` or `"odd"`: Square wave
- `"triangle"` or `"1/n**2"`: Triangle wave

---

### `FibonacciRhythm`

Fibonacci-based rhythm generator.

```python
from math_music_engine.generators import FibonacciRhythm

gen = FibonacciRhythm(sample_rate=44100)
```

#### Methods

##### `generate_rhythm_pattern(num_beats, subdivision=16)`
Generate binary rhythm pattern.

##### `generate_pulse_train(pattern, tempo, duration, sample_rate=None)`
Convert rhythm pattern to audio.

**Example:**
```python
pattern = gen.generate_rhythm_pattern(num_beats=4, subdivision=16)
signal = gen.generate_pulse_train(pattern, tempo=120, duration=4.0)
```

---

### `ChaosModulator`

Chaotic system modulation.

```python
from math_music_engine.generators import ChaosModulator

gen = ChaosModulator(sample_rate=44100)
```

#### Methods

##### `generate(system, duration, **kwargs)`
Generate chaotic signal.

**Systems:**
- `'logistic'`: Logistic map
- `'lorenz'`: Lorenz attractor
- `'rossler'`: Rössler attractor
- `'henon'`: Hénon map

**Example:**
```python
signal = gen.generate('lorenz', duration=10.0, sigma=10, rho=28)
```

---

### `FractalMelody`

Fractal and L-system melody generator.

```python
from math_music_engine.generators import FractalMelody

gen = FractalMelody()
```

#### Methods

##### `generate_lsystem(axiom, rules, iterations)`
Generate L-system string.

##### `generate_koch_curve_melody(iterations, intervals=None)`
Generate Koch curve-based melody.

**Example:**
```python
intervals = gen.generate_koch_curve_melody(iterations=3)
notes = gen.intervals_to_notes(intervals, base_note=60)
```

---

### `PrimeSequence`

Prime number-based generator.

```python
from math_music_engine.generators import PrimeSequence

gen = PrimeSequence()
```

#### Methods

##### `prime_melody(num_notes, base_note=60, method='gaps')`
Generate melody from primes.

**Methods:**
- `'direct'`: Use primes as intervals
- `'gaps'`: Use prime gaps
- `'residues'`: Use prime modulo 12

---

### `RandomWalk`

Stochastic process generator.

```python
from math_music_engine.generators import RandomWalk

gen = RandomWalk(sample_rate=44100, seed=42)
```

#### Methods

##### `brownian_motion(duration, step_size=1.0, bias=0.0)`
Generate Brownian motion.

##### `markov_melody(num_notes, transition_matrix=None, states=None)`
Generate Markov chain melody.

---

## Visualization Module

### Waveform Plots

```python
from math_music_engine.visualization import plot_waveform

plot_waveform(
    signal,
    sample_rate=44100,
    duration=1.0,
    output_path="waveform.png"
)
```

### Spectrograms

```python
from math_music_engine.visualization import plot_spectrogram

plot_spectrogram(
    signal,
    sample_rate=44100,
    window_size=2048,
    output_path="spectrogram.png"
)
```

### 3D Plots

```python
from math_music_engine.visualization import plot_function_surface

def my_function(x, y):
    return np.sin(x) * np.cos(y)

plot_function_surface(
    my_function,
    x_range=(-5, 5),
    y_range=(-5, 5)
)
```

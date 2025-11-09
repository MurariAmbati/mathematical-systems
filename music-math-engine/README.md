# math-music-engine

generate music from mathematical functions

## what it does

turns math into sound. deterministic, reproducible, and modular.

```python
from math_music_engine.core import Oscillator, WaveformType, OutputManager

# generate 440 hz tone
osc = Oscillator()
signal = osc.generate(WaveformType.SINE, 440.0, duration=2.0)

# save it
OutputManager().export_audio(signal, "tone.wav")
```

## install

```bash
pip install -e .
```

requires python ≥3.12

## quick start

```bash
# run demo
python examples/simple_demo.py

# use cli
mathmusic generate --generator fourier --duration 5 --output test.wav
mathmusic list
```

## generators

- **fourier** - harmonic synthesis
- **fibonacci** - rhythmic patterns from golden ratio
- **chaos** - lorenz, rössler, logistic map attractors
- **fractal** - self-similar melodies using l-systems
- **prime** - sequences from prime numbers
- **random** - brownian motion, lévy flights, markov chains

## examples

**basic tone**
```python
from math_music_engine.core import Oscillator, WaveformType, ADSR, OutputManager

osc = Oscillator()
signal = osc.generate(WaveformType.SINE, 440.0, duration=2.0)

# add envelope
adsr = ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
envelope = adsr.generate(duration=2.0)
signal = signal * envelope

OutputManager().export_audio(signal, "tone.wav")
```

**using generators**
```python
from math_music_engine.generators import FourierSynth
from math_music_engine.core import OutputManager

synth = FourierSynth(harmonics=[1.0, 0.5, 0.25, 0.125])
signal = synth.generate(fundamental_freq=220.0, duration=5.0)

OutputManager().export_audio(signal, "fourier.wav")
```

**scale quantization**
```python
from math_music_engine.core import MappingEngine, Scale
import numpy as np

mapper = MappingEngine()
values = np.random.random(20)

notes = mapper.map_to_midi(
    values,
    scale=Scale.PENTATONIC_MINOR,
    root_note='A3',
    octave_range=3
)
```

## features

- deterministic synthesis with seed control
- 6 mathematical generators
- musical scale quantization (major, minor, pentatonic, modes)
- wav, midi, json export
- waveform & spectrogram visualization
- command line interface
- comprehensive tests

## structure

```
src/math_music_engine/
├── parser/          # sympy expression parsing
├── core/            # engines and synthesis
├── generators/      # mathematical generators
├── visualization/   # plotting tools
└── interfaces/      # cli
```

## cli

```bash
# list generators
mathmusic list

# generate audio
mathmusic generate --generator fourier --duration 5 --output out.wav
mathmusic generate --generator chaos --duration 10 --output chaos.wav
```

## demos

```bash
python examples/simple_demo.py        # basic 440hz tone
python examples/example_generators.py # multiple generators
python examples/advanced_demo.py      # complex compositions
```

## documentation

- `examples/README.md` - walkthrough of all examples
- `docs/api_reference.md` - complete api documentation
- `docs/math_reference.md` - mathematical formulas
- `QUICK_REFERENCE.md` - one-page cheat sheet

## test

```bash
pytest                         # run all tests
pytest tests/test_mapping.py   # specific test
```

## dependencies

numpy, scipy, sympy, soundfile, mido, matplotlib, plotly

all installed automatically with `pip install -e .`

## license

[to be determined]

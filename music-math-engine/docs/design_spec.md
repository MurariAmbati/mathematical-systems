# Design Specification

## Overview

The Math-Music-Engine is a modular framework for generating music directly from mathematical functions, sequences, and structures. All musical outputs are deterministic functions of their mathematical definitions.

## Architecture

### Layer Structure

```
┌─────────────────────────────────────────┐
│         Interfaces Layer                │
│  (CLI, API, SuperCollider Bridge)      │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│       Generators Layer                   │
│  (Fourier, Fibonacci, Chaos, etc.)      │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Core Processing Layer            │
│ ┌─────────────┬──────────────────────┐ │
│ │   Parser    │  Function Engine     │ │
│ ├─────────────┼──────────────────────┤ │
│ │   Mapper    │  Synthesis           │ │
│ ├─────────────┼──────────────────────┤ │
│ │ Composition │  Output Manager      │ │
│ └─────────────┴──────────────────────┘ │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│      Visualization Layer                 │
│  (Waveforms, Spectrograms, 3D Plots)    │
└─────────────────────────────────────────┘
```

## Core Modules

### 1. Parser Module

**Purpose:** Parse mathematical expressions into executable functions.

**Components:**
- `ExpressionParser`: Main parser class using SymPy
- Expression validation and type checking
- Domain and continuity analysis
- Differentiation and integration support

**Input:** String representation of mathematical function
**Output:** Callable NumPy-compatible function

### 2. Function Engine

**Purpose:** Sample mathematical functions over time.

**Features:**
- Uniform sampling (fixed Δt)
- Adaptive sampling (curvature-dependent)
- Multidimensional function support
- Configurable sample rates

**Sampling Strategies:**
```python
# Uniform: Fixed time steps
t = linspace(0, duration, num_samples)

# Adaptive: More samples where function changes rapidly
curvature = compute_second_derivative(f)
density = curvature / tolerance
```

### 3. Mapping Engine

**Purpose:** Transform mathematical outputs to musical parameters.

**Mappings:**

| Parameter | Input Domain | Output Range | Transform |
|-----------|--------------|--------------|-----------|
| Frequency | ℝ | 20-20000 Hz | Logarithmic |
| Amplitude | [-∞, ∞] | [0, 1] | Sigmoid/Linear |
| MIDI Note | ℝ | 0-127 | Quantized |
| Timbre | ℝⁿ | [0, 1]ⁿ | Normalized |

**Scale Quantization:**
- Maps continuous frequencies to discrete scale degrees
- Supports major, minor, pentatonic, and modal scales
- Preserves melodic contour while enforcing scale constraints

### 4. Synthesis Layer

**Oscillators:**
- Sine, square, sawtooth, triangle
- Frequency modulation support
- Phase-accurate generation

**Envelopes:**
- ADSR (Attack, Decay, Sustain, Release)
- Custom envelope shapes
- Per-note or global application

**Synthesis Methods:**
- **Additive:** Sum of weighted harmonics
- **Subtractive:** Filtered rich waveforms
- **FM:** Frequency modulation
- **AM:** Amplitude modulation

### 5. Composition Layer

**Features:**
- Multi-voice management
- Voice mixing strategies (sum, multiply, modulate)
- Temporal alignment and phase locking
- Function composition (f ∘ g)
- Stereo panning

**Mixing Modes:**
```
SUM:      v_out = Σ v_i
MULTIPLY: v_out = Π v_i
MODULATE: v_out = v_carrier * (1 + Σ v_mod)
```

### 6. Output Manager

**Export Formats:**
- **WAV:** Uncompressed audio (16/24/32-bit)
- **MIDI:** Note events with timing
- **JSON:** Reproducibility metadata

**Metadata Schema:**
```json
{
  "version": "0.1.0",
  "timestamp": "ISO-8601",
  "expression": "mathematical expression",
  "generator": "generator name",
  "parameters": {...},
  "mappings": {...},
  "seed": 42
}
```

## Generators

### Fourier Synth
- **Basis:** Fourier series
- **Parameters:** Harmonic amplitudes, phases
- **Output:** Timbre-shaped audio

### Fibonacci Rhythm
- **Basis:** Fibonacci sequence
- **Parameters:** Tempo, subdivision
- **Output:** Rhythmic patterns

### Chaos Modulator
- **Systems:** Logistic, Lorenz, Rössler
- **Parameters:** System-specific constants
- **Output:** Dynamic modulation signals

### Fractal Melody
- **Basis:** L-systems, recursive patterns
- **Parameters:** Axiom, rules, iterations
- **Output:** Self-similar melodies

### Prime Sequence
- **Basis:** Prime numbers and properties
- **Parameters:** Method (direct, gaps, residues)
- **Output:** Note sequences

### Random Walk
- **Processes:** Brownian, Lévy, Ornstein-Uhlenbeck
- **Parameters:** Step size, bias, seed
- **Output:** Stochastic melodies

## Visualization

### Time Domain
- Waveform plots
- Multi-channel comparison
- Zoom and navigation

### Frequency Domain
- FFT spectrum
- Spectrogram (STFT)
- Mel-frequency spectrogram

### Mathematical
- 3D surface plots
- Parametric curves
- Phase space diagrams
- Attractor visualization

### Mapping
- Input-to-frequency curves
- Scale quantization effects
- Harmonic spectrum
- Parameter evolution

## Performance Requirements

### Latency
- Real-time mode: < 5 ms processing latency
- Batch mode: Optimized for throughput

### Precision
- Frequency deviation: ≤ 1 cent
- Phase accuracy: ≤ 0.1°
- Amplitude linearity: ≤ 0.1 dB error

### Scalability
- Support durations: 0.1s to 3600s
- Sample rates: 22.05 kHz to 192 kHz
- Voices: 1 to 64 simultaneous

## Testing Strategy

### Unit Tests
- Individual module functionality
- Edge cases and error handling
- ≥ 90% code coverage target

### Integration Tests
- End-to-end workflows
- Generator → synthesis → export
- Metadata reproducibility

### Validation Tests
- Frequency accuracy
- Amplitude calibration
- Deterministic output verification

### Performance Tests
- Latency measurements
- Memory usage profiling
- Long-duration stability

## Extensibility

### Adding New Generators
1. Inherit from base generator pattern
2. Implement `generate()` method
3. Provide `get_metadata()` method
4. Add to generator registry
5. Write unit tests

### Adding New Mappings
1. Add mapping function to `MappingEngine`
2. Define input/output domains
3. Implement inverse mapping (if applicable)
4. Document mathematical basis
5. Add visualization support

### Adding New Synthesis Methods
1. Create synthesizer class
2. Implement parameter validation
3. Ensure normalization
4. Support envelope application
5. Test with various inputs

## Future Enhancements

- Real-time MIDI input/output
- SuperCollider OSC integration
- Web-based interactive interface
- GPU acceleration for synthesis
- Machine learning-based generators
- Spatial audio (surround/binaural)
- Video synchronization
- Network streaming

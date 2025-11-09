# Mapping Schema

This document describes the mapping schemas used in the Math-Music-Engine for converting mathematical outputs to musical parameters.

## Frequency Mapping Schema

### Linear Mapping

```json
{
  "type": "frequency",
  "mode": "linear",
  "parameters": {
    "input_range": [0, 1],
    "output_range": [220, 880],
    "unit": "Hz"
  },
  "formula": "f(x) = f_min + x * (f_max - f_min)"
}
```

### Logarithmic Mapping

```json
{
  "type": "frequency",
  "mode": "logarithmic",
  "parameters": {
    "input_range": [0, 1],
    "output_range": [20, 20000],
    "unit": "Hz",
    "perceptual": true
  },
  "formula": "f(x) = 2^(log2(f_min) + x * (log2(f_max) - log2(f_min)))"
}
```

### Quantized Mapping (Scale)

```json
{
  "type": "frequency",
  "mode": "quantized",
  "parameters": {
    "scale": "major",
    "root_note": "C4",
    "root_frequency": 261.63,
    "octave_range": 3,
    "intervals": [0, 2, 4, 5, 7, 9, 11]
  }
}
```

## Amplitude Mapping Schema

### Linear Amplitude

```json
{
  "type": "amplitude",
  "mode": "linear",
  "parameters": {
    "input_range": [-1, 1],
    "output_range": [0, 1],
    "normalization": "peak"
  }
}
```

### Logarithmic Amplitude (dB)

```json
{
  "type": "amplitude",
  "mode": "logarithmic",
  "parameters": {
    "input_range": [0, 1],
    "output_db_range": [-80, 0],
    "reference": "full_scale"
  },
  "formula": "A_dB(x) = dB_min + x * (dB_max - dB_min)"
}
```

### Compressed Amplitude

```json
{
  "type": "amplitude",
  "mode": "compressed",
  "parameters": {
    "input_range": [0, 1],
    "output_range": [0, 1],
    "compression_ratio": 2.0,
    "threshold": 0.7
  }
}
```

## MIDI Mapping Schema

### Direct MIDI

```json
{
  "type": "midi",
  "mode": "direct",
  "parameters": {
    "input_range": [0, 1],
    "output_range": [0, 127],
    "quantization": "semitone"
  }
}
```

### Scale-Quantized MIDI

```json
{
  "type": "midi",
  "mode": "scale_quantized",
  "parameters": {
    "scale": {
      "name": "minor_pentatonic",
      "root": "A4",
      "root_midi": 69,
      "intervals": [0, 3, 5, 7, 10]
    },
    "octave_range": 2,
    "voice_leading": "nearest"
  }
}
```

## Timbre Mapping Schema

### Harmonic Distribution

```json
{
  "type": "timbre",
  "mode": "harmonic",
  "parameters": {
    "num_harmonics": 8,
    "distribution": "exponential",
    "decay_rate": 0.5,
    "odd_even_ratio": 1.0
  },
  "harmonics": {
    "1": 1.0,
    "2": 0.5,
    "3": 0.25,
    "4": 0.125
  }
}
```

### Spectral Envelope

```json
{
  "type": "timbre",
  "mode": "spectral_envelope",
  "parameters": {
    "control_points": [
      {"frequency": 100, "amplitude": 0.0},
      {"frequency": 1000, "amplitude": 1.0},
      {"frequency": 5000, "amplitude": 0.3},
      {"frequency": 10000, "amplitude": 0.0}
    ],
    "interpolation": "logarithmic"
  }
}
```

## Temporal Mapping Schema

### Rhythm Pattern

```json
{
  "type": "rhythm",
  "mode": "pattern",
  "parameters": {
    "tempo": 120,
    "time_signature": "4/4",
    "subdivision": 16,
    "pattern": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
  }
}
```

### Event Timing

```json
{
  "type": "timing",
  "mode": "events",
  "parameters": {
    "events": [
      {"time": 0.0, "type": "note_on", "data": {"note": 60}},
      {"time": 0.5, "type": "note_off", "data": {"note": 60}},
      {"time": 1.0, "type": "note_on", "data": {"note": 64}}
    ],
    "time_unit": "seconds"
  }
}
```

## Complete Example: Generator Configuration

```json
{
  "version": "0.1.0",
  "timestamp": "2025-11-08T10:00:00Z",
  "generator": {
    "name": "FourierSynth",
    "type": "additive",
    "parameters": {
      "harmonics": [1.0, 0.5, 0.25, 0.125],
      "fundamental_freq": 440.0,
      "duration": 10.0
    }
  },
  "mappings": {
    "frequency": {
      "mode": "logarithmic",
      "input_source": "function_output",
      "freq_min": 220.0,
      "freq_max": 880.0,
      "scale": "major",
      "root_note": "A4"
    },
    "amplitude": {
      "mode": "linear",
      "envelope": {
        "type": "ADSR",
        "attack": 0.01,
        "decay": 0.1,
        "sustain": 0.7,
        "release": 0.2
      }
    },
    "timbre": {
      "mode": "harmonic",
      "num_harmonics": 8,
      "distribution": "1/n"
    }
  },
  "synthesis": {
    "sample_rate": 44100,
    "bit_depth": 16,
    "channels": 1,
    "normalization": "peak",
    "dithering": false
  },
  "output": {
    "format": "wav",
    "filename": "output.wav",
    "metadata_export": true
  },
  "reproducibility": {
    "seed": 42,
    "deterministic": true,
    "version_info": {
      "python": "3.12",
      "numpy": "1.26.0",
      "scipy": "1.11.0"
    }
  }
}
```

## Scale Definitions

```json
{
  "scales": {
    "chromatic": {
      "intervals": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      "description": "All 12 semitones"
    },
    "major": {
      "intervals": [0, 2, 4, 5, 7, 9, 11],
      "description": "Major scale (Ionian mode)",
      "formula": "W-W-H-W-W-W-H"
    },
    "minor": {
      "intervals": [0, 2, 3, 5, 7, 8, 10],
      "description": "Natural minor scale (Aeolian mode)",
      "formula": "W-H-W-W-H-W-W"
    },
    "harmonic_minor": {
      "intervals": [0, 2, 3, 5, 7, 8, 11],
      "description": "Harmonic minor scale"
    },
    "pentatonic_major": {
      "intervals": [0, 2, 4, 7, 9],
      "description": "Major pentatonic scale"
    },
    "pentatonic_minor": {
      "intervals": [0, 3, 5, 7, 10],
      "description": "Minor pentatonic scale"
    },
    "whole_tone": {
      "intervals": [0, 2, 4, 6, 8, 10],
      "description": "Whole tone scale"
    },
    "dorian": {
      "intervals": [0, 2, 3, 5, 7, 9, 10],
      "description": "Dorian mode"
    },
    "phrygian": {
      "intervals": [0, 1, 3, 5, 7, 8, 10],
      "description": "Phrygian mode"
    },
    "lydian": {
      "intervals": [0, 2, 4, 6, 7, 9, 11],
      "description": "Lydian mode"
    },
    "mixolydian": {
      "intervals": [0, 2, 4, 5, 7, 9, 10],
      "description": "Mixolydian mode"
    }
  }
}
```

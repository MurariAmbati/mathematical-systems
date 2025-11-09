# Mathematical Reference

This document provides detailed mathematical foundations for the Math-Music-Engine.

## 1. Function-to-Sound Mapping

### 1.1 Frequency Mapping

#### Linear Mapping
```
f(x) = f_min + x * (f_max - f_min)
where x ∈ [0, 1]
```

#### Logarithmic Mapping
```
f(x) = 2^(log₂(f_min) + x * (log₂(f_max) - log₂(f_min)))
```
This provides perceptually linear pitch scaling.

#### MIDI Note to Frequency
```
f(n) = 440 * 2^((n - 69) / 12)
where n is MIDI note number, A4 (440 Hz) = MIDI 69
```

### 1.2 Amplitude Mapping

#### Linear Amplitude
```
A(x) = A_min + x * (A_max - A_min)
```

#### Logarithmic Amplitude (dB scale)
```
A_dB(x) = 20 * log₁₀(A_min) + x * (20 * log₁₀(A_max) - 20 * log₁₀(A_min))
A(x) = 10^(A_dB(x) / 20)
```

## 2. Mathematical Generators

### 2.1 Fourier Synthesis

**Fourier Series:**
```
f(t) = a₀/2 + Σ(n=1 to N) [aₙ·cos(nωt) + bₙ·sin(nωt)]
```

For harmonic synthesis:
```
f(t) = Σ(n=1 to N) Aₙ·sin(2πnf₀t + φₙ)
where:
  f₀ = fundamental frequency
  Aₙ = amplitude of nth harmonic
  φₙ = phase of nth harmonic
```

### 2.2 Fibonacci Sequence

**Recurrence Relation:**
```
F(0) = 0
F(1) = 1
F(n) = F(n-1) + F(n-2) for n > 1
```

**Golden Ratio:**
```
φ = (1 + √5) / 2 ≈ 1.618033988749...
lim(n→∞) F(n+1)/F(n) = φ
```

### 2.3 Chaotic Systems

#### Logistic Map
```
x(n+1) = r·x(n)·(1 - x(n))
where:
  r ∈ [0, 4] (control parameter)
  x ∈ [0, 1]
  Chaotic for r ≈ 3.57 to 4.0
```

#### Lorenz Attractor
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

where:
  σ = 10 (Prandtl number)
  ρ = 28 (Rayleigh number)
  β = 8/3
```

#### Rössler Attractor
```
dx/dt = -y - z
dy/dt = x + ay
dz/dt = b + z(x - c)

where typical values:
  a = 0.2
  b = 0.2
  c = 5.7
```

### 2.4 L-Systems (Lindenmayer Systems)

**Production Rules:**
```
G = (V, ω, P)
where:
  V = alphabet (variables and constants)
  ω = axiom (initial string)
  P = production rules
```

Example (Algae):
```
Axiom: A
Rules: A → AB, B → A
Iterations:
  0: A
  1: AB
  2: ABA
  3: ABAAB
  4: ABAABABA
```

### 2.5 Prime Numbers

**Prime Number Theorem:**
```
π(x) ≈ x / ln(x)
where π(x) is the number of primes ≤ x
```

**nth Prime Approximation:**
```
p(n) ≈ n·ln(n) for large n
```

**Prime Gaps:**
```
g(n) = p(n+1) - p(n)
Average gap ≈ ln(p(n))
```

### 2.6 Random Walks

#### Brownian Motion
```
dX(t) = μ dt + σ dW(t)
where:
  μ = drift
  σ = volatility
  W(t) = Wiener process
```

#### Geometric Brownian Motion
```
dS(t) = μS(t) dt + σS(t) dW(t)
Solution: S(t) = S(0)·exp((μ - σ²/2)t + σW(t))
```

#### Ornstein-Uhlenbeck Process
```
dX(t) = θ(μ - X(t)) dt + σ dW(t)
where:
  θ = mean reversion rate
  μ = long-term mean
```

## 3. Signal Processing

### 3.1 Sampling Theorem (Nyquist)
```
f_s ≥ 2·f_max
where:
  f_s = sampling rate
  f_max = maximum frequency in signal
```

### 3.2 ADSR Envelope

**Attack Phase:**
```
A(t) = t / t_attack for 0 ≤ t < t_attack
```

**Decay Phase:**
```
A(t) = 1 - (1 - S)·(t - t_attack) / t_decay
for t_attack ≤ t < t_attack + t_decay
where S = sustain level
```

**Sustain Phase:**
```
A(t) = S for t_attack + t_decay ≤ t < t_gate
```

**Release Phase:**
```
A(t) = S·(1 - (t - t_gate) / t_release)
for t_gate ≤ t < t_gate + t_release
```

### 3.3 Frequency Modulation

**FM Formula:**
```
s(t) = A·sin(2πf_c·t + I·sin(2πf_m·t))
where:
  f_c = carrier frequency
  f_m = modulator frequency
  I = modulation index
```

**Sidebands:**
```
Generated frequencies: f_c ± n·f_m
for n = 0, 1, 2, ..., I (approximately)
```

## 4. Musical Scales

### Scale Intervals (semitones from root)

| Scale | Intervals |
|-------|-----------|
| Chromatic | 0,1,2,3,4,5,6,7,8,9,10,11 |
| Major | 0,2,4,5,7,9,11 |
| Minor | 0,2,3,5,7,8,10 |
| Pentatonic Major | 0,2,4,7,9 |
| Pentatonic Minor | 0,3,5,7,10 |

### Frequency Ratios (Just Intonation)

| Interval | Ratio | Cents |
|----------|-------|-------|
| Unison | 1:1 | 0 |
| Minor 2nd | 16:15 | 112 |
| Major 2nd | 9:8 | 204 |
| Minor 3rd | 6:5 | 316 |
| Major 3rd | 5:4 | 386 |
| Perfect 4th | 4:3 | 498 |
| Perfect 5th | 3:2 | 702 |
| Octave | 2:1 | 1200 |

## 5. Audio Specifications

### Standard Sample Rates
- 44.1 kHz: CD quality
- 48 kHz: Professional audio/video
- 96 kHz: High-resolution audio

### Bit Depths
- 16-bit: 96 dB dynamic range
- 24-bit: 144 dB dynamic range
- 32-bit float: ~1500 dB dynamic range

### Nyquist Frequency
```
f_nyquist = f_s / 2
Maximum representable frequency without aliasing
```

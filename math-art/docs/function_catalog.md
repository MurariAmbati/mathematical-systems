# Math Art Generator - Function Catalog

## Supported Mathematical Functions

The Math Art Generator supports 30+ mathematical functions for use in expressions.

### Trigonometric Functions

| Function | Description | Example |
|----------|-------------|---------|
| `sin(x)` | Sine | `sin(theta)` |
| `cos(x)` | Cosine | `cos(x*y)` |
| `tan(x)` | Tangent | `tan(x/2)` |
| `sec(x)` | Secant | `sec(theta)` |
| `csc(x)` | Cosecant | `csc(2*x)` |
| `cot(x)` | Cotangent | `cot(x+y)` |

### Inverse Trigonometric

| Function | Description | Domain |
|----------|-------------|--------|
| `asin(x)` | Arcsine | [-1, 1] |
| `acos(x)` | Arccosine | [-1, 1] |
| `atan(x)` | Arctangent | All reals |
| `atan2(y,x)` | Two-argument arctangent | All reals |

### Hyperbolic Functions

| Function | Description | Example |
|----------|-------------|---------|
| `sinh(x)` | Hyperbolic sine | `sinh(x)` |
| `cosh(x)` | Hyperbolic cosine | `cosh(x)` |
| `tanh(x)` | Hyperbolic tangent | `tanh(x*y)` |
| `asinh(x)` | Inverse hyperbolic sine | `asinh(x)` |
| `acosh(x)` | Inverse hyperbolic cosine | `acosh(x)` (x ≥ 1) |
| `atanh(x)` | Inverse hyperbolic tangent | `atanh(x)` (-1 < x < 1) |
| `sech(x)` | Hyperbolic secant | `sech(x)` |
| `csch(x)` | Hyperbolic cosecant | `csch(x)` |
| `coth(x)` | Hyperbolic cotangent | `coth(x)` |

### Exponential and Logarithmic

| Function | Description | Example |
|----------|-------------|---------|
| `exp(x)` | Exponential (e^x) | `exp(x*y)` |
| `log(x)` | Natural logarithm | `log(x**2 + y**2)` |
| `ln(x)` | Natural logarithm (alias) | `ln(abs(x))` |
| `log10(x)` | Base-10 logarithm | `log10(x)` |
| `log2(x)` | Base-2 logarithm | `log2(x)` |

### Power and Roots

| Function | Description | Example |
|----------|-------------|---------|
| `sqrt(x)` | Square root | `sqrt(x**2 + y**2)` |
| `cbrt(x)` | Cube root | `cbrt(x**3)` |
| `pow(x, y)` | Power x^y | `pow(x, 2.5)` |
| `**` | Power operator | `x**3` |

### Rounding and Integer

| Function | Description | Example |
|----------|-------------|---------|
| `abs(x)` | Absolute value | `abs(sin(x))` |
| `sign(x)` | Sign (-1, 0, or 1) | `sign(x*y)` |
| `floor(x)` | Floor (round down) | `floor(x*10)` |
| `ceil(x)` | Ceiling (round up) | `ceil(x*10)` |
| `round(x)` | Round to nearest | `round(x*10)` |

### Other Functions

| Function | Description | Example |
|----------|-------------|---------|
| `max(x, y)` | Maximum | `max(sin(x), cos(y))` |
| `min(x, y)` | Minimum | `min(x, y)` |
| `mod(x, y)` | Modulo | `mod(x, 2*pi)` |

## Supported Variables

| Variable | Description | Common Use |
|----------|-------------|------------|
| `x` | X coordinate | Cartesian expressions |
| `y` | Y coordinate | Cartesian expressions |
| `z` | Z coordinate | 3D expressions |
| `t` | Time parameter | Animations, parametric |
| `r` | Radius | Polar expressions |
| `theta` | Angle (radians) | Polar expressions |
| `θ` | Angle (Unicode) | Polar expressions |

## Expression Examples

### Simple Patterns

```python
# Wave
"sin(x) + cos(y)"

# Ripple
"sin(sqrt(x**2 + y**2))"

# Grid
"sin(x*5) * cos(y*5)"

# Spiral
"sin(r*theta)"
```

### Interference Patterns

```python
# Two-source interference
"sin(sqrt((x-1)**2 + y**2)*5) + sin(sqrt((x+1)**2 + y**2)*5)"

# Circular waves
"cos(sqrt(x**2 + y**2)*10)"

# Beat pattern
"sin(x*10) + sin(x*11)"
```

### Polar Equations

```python
# Rose curve (5 petals)
"sin(5*theta)"

# Cardioid
"1 + cos(theta)"

# Limaçon
"1 + 0.5*cos(theta)"

# Spiral
"theta"

# Logarithmic spiral
"exp(0.2*theta)"
```

### Complex Expressions

```python
# Mandala-like
"sin(x**2 + y**2) * cos(x*y) + sin(5*atan2(y, x))"

# Interference + modulation
"sin(sqrt(x**2 + y**2)*5) * cos(x*y)"

# Wave packet
"exp(-(x**2 + y**2)/4) * sin(x*10)"
```

### Using Multiple Functions

```python
# Combination
"sin(x*3) + cos(y*3) + tan(x*y)"

# Nested
"sin(cos(x*2) + sin(y*2))"

# With absolute values
"abs(sin(x)) - abs(cos(y))"
```

## Generator-Specific Patterns

### Spirograph

```python
# Hypotrochoid
Spirograph(R=8, r=1, a=4, mode="hypo")

# Epitrochoid
Spirograph(R=8, r=3, a=5, mode="epi")

# Rose pattern
RosePattern(A=1.0, k=7)  # 7 petals
```

### Lissajous

```python
# Basic (3:2 ratio)
Lissajous(a=3, b=2, delta=pi/2)

# Complex (5:4 ratio)
Lissajous(a=5, b=4, delta=0)

# 3D
Lissajous3D(a=3, b=2, c=1)
```

### Attractors

```python
# Lorenz (classic parameters)
LorenzAttractor(sigma=10, rho=28, beta=8/3)

# Clifford (fractal)
CliffordAttractor(a=-1.4, b=1.6, c=1.0, d=0.7)

# Ikeda (laser)
IkedaAttractor(u=0.918)
```

### Custom Equations

```python
# Cartesian
CustomEquation("sin(x*y) + cos(x**2 - y**2)")

# With noise
HybridGenerator("sin(x*3) * cos(y*3)", noise_weight=0.3)

# Contour
CustomEquation("sin(x*2) + cos(y*2)", mode="contour", threshold=0)
```

## Tips for Creating Expressions

### 1. Balance Complexity
- Start simple, add complexity gradually
- Too complex may be slow to evaluate

### 2. Use Appropriate Ranges
- Match x/y ranges to function behavior
- Polar: r typically [0, max_radius]
- Angles: [0, 2π] for one rotation

### 3. Avoid Domain Errors
- Check for negative sqrt arguments
- Avoid log(0) or log(negative)
- Watch for division by zero

### 4. Experiment with Frequencies
- Integer ratios create closed curves
- Irrational ratios create dense patterns
- Higher frequencies = more detail

### 5. Combine Functions
- Addition: overlapping patterns
- Multiplication: modulation/beating
- Nesting: complex interactions

## Advanced Techniques

### Symmetry

```python
# Radial symmetry
"sin(5*atan2(y, x))"

# Mirror symmetry
"sin(abs(x)) + cos(abs(y))"
```

### Scaling

```python
# Non-uniform scaling
"sin(x*2) + cos(y*5)"

# Distance-based
"sin(sqrt(x**2 + y**2)*k)"
```

### Phase Shifts

```python
# Horizontal shift
"sin(x + pi/4)"

# Vertical shift
"sin(x) + 1"

# Time animation
"sin(x + t)"
```

### Damping/Envelope

```python
# Gaussian envelope
"exp(-(x**2 + y**2)/4) * sin(x*10)"

# Distance damping
"sin(r*10) / (r + 1)"
```

## Common Patterns

| Pattern | Expression | Visual |
|---------|-----------|--------|
| Concentric circles | `sin(sqrt(x**2 + y**2)*k)` | Ripples |
| Spiral | `sin(r*theta)` | Logarithmic |
| Checkerboard | `sin(x*k) * sin(y*k)` | Grid |
| Star | `sin(k*atan2(y,x))` | Radial |
| Hyperbolic | `sin(x*y)` | Saddle |
| Rose | `sin(k*theta)` | Petals |

## Troubleshooting

### Expression Doesn't Evaluate
- Check variable names (x, y, z, t, r, theta only)
- Verify function names (case-sensitive)
- Check parentheses balance

### Domain Errors
- Use `abs()` for sqrt of potentially negative
- Add small offset: `log(x**2 + y**2 + 0.01)`

### No Visible Pattern
- Adjust x/y ranges
- Check frequency scaling
- Try different colormap

### Too Slow
- Reduce samples
- Simplify expression
- Use pre-computed generators instead

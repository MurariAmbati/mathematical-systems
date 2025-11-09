# Math Art Generator - Architecture

## Overview

The Math Art Generator is a high-performance system for creating procedural art from mathematical equations, parametric functions, and dynamical systems.

## Core Architecture

### Layered Design

```
┌─────────────────────────────────────────────────┐
│              User Interface Layer                │
│    (Examples, Scripts, Optional Web UI)          │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│              Export & Rendering Layer            │
│  - Static Renderer (PNG, JPG)                   │
│  - Animation Renderer (MP4, GIF)                │
│  - Vector Renderer (SVG)                        │
│  - Export Utilities (JSON, CSV)                 │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│              Generator Layer                     │
│  - Spirograph (Hypo/Epi-trochoid)              │
│  - Lissajous (2D/3D parametric)                 │
│  - Attractors (Lorenz, Clifford, etc)           │
│  - Polar Patterns (r = f(θ))                    │
│  - Custom Equations (User-defined)              │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│              Core Computation Layer              │
│  - Parser (Safe expression parsing)             │
│  - Evaluator (Numeric evaluation)               │
│  - Coordinates (Transformations)                │
│  - Color Maps (Gradient mapping)                │
│  - Utils (RNG, noise, helpers)                  │
└─────────────────────────────────────────────────┘
```

## Module Details

### 1. Parser (`core/parser.py`)

**Purpose**: Safe parsing and validation of mathematical expressions

**Key Features**:
- SymPy-based expression parsing
- 30+ supported mathematical functions
- Variable validation (x, y, z, t, r, θ)
- Domain checking (sqrt, log, division)
- No unsafe `eval()` calls

**Classes**:
- `ParsedExpression`: Represents compiled expression
- Functions: `parse_equation()`, `validate_expression()`

### 2. Evaluator (`core/evaluator.py`)

**Purpose**: High-performance numeric evaluation

**Key Features**:
- NumPy vectorization for ≥10⁶ points/second
- Optional Numba JIT compilation
- Symbolic differentiation and integration
- Batch evaluation support

**Classes**:
- `Evaluator`: Main evaluation engine
- Functions: `evaluate_batch()`, `create_parametric_evaluator()`

### 3. Coordinates (`core/coordinates.py`)

**Purpose**: Coordinate system transformations

**Supported Systems**:
- Cartesian (x, y, z)
- Polar (r, θ)
- Parametric (x(t), y(t), z(t))
- Spherical (r, θ, φ)

**Key Functions**:
- `cartesian_grid()`: Create 2D grid
- `polar_to_cartesian()`: Polar → Cartesian
- `parametric_to_points()`: Generate curve points
- `normalize_points()`: Auto-scale to range

### 4. Generators (`core/generators/`)

**Base Classes**:
- `ArtGenerator`: Abstract base
- `ParametricGenerator`: For parametric curves
- `CartesianGenerator`: For f(x,y) equations
- `PolarGenerator`: For r = f(θ) patterns
- `IterativeGenerator`: For attractors/fractals

**Implementations**:

#### Spirograph (`spirograph.py`)
- Hypotrochoid: Rolling circle inside
- Epitrochoid: Rolling circle outside
- Rose patterns: r = cos(kθ)

#### Lissajous (`lissajous.py`)
- 2D: x = A sin(at+δ), y = B sin(bt)
- 3D: Add z = C sin(ct+δz)
- Bowditch: Generalized with multiple frequencies

#### Attractors (`attractors.py`)
- Lorenz: Chaotic weather model
- Clifford: Fractal-like patterns
- Ikeda: Laser dynamics
- De Jong: Parameter-sensitive art

#### Polar Patterns (`polar_patterns.py`)
- Cardioid: Heart shape
- Limaçon: Generalized cardioid
- Spirals: Archimedean, Logarithmic
- Maurer Rose: Modular arithmetic

#### Custom Equations (`custom_equation.py`)
- User expressions: f(x, y)
- Noise fields: Perlin/Simplex
- Hybrid: Expression + noise

### 5. Color Maps (`core/color_maps.py`)

**Purpose**: Map data to colors

**Classes**:
- `ColorMap`: Base colormap
- `GradientColorMap`: Custom gradients
- `CyclicColorMap`: For periodic data
- `FunctionColorMapper`: Map by value, gradient, curvature, velocity
- `DiscreteColorPalette`: Categorical colors

**Predefined Palettes**:
- Default: viridis, plasma, magma, inferno
- Custom: ocean, fire, forest, sunset, northern_lights

### 6. Renderers (`core/renderers/`)

#### Static Renderer (`static_renderer.py`)
- PNG/JPG output
- Matplotlib-based
- Styles: scatter, line, density
- Multi-panel support

#### Animation Renderer (`animation_renderer.py`)
- MP4/GIF output
- FFmpeg/Pillow writers
- Frame-by-frame generation
- Parameter sweep animations

#### Vector Renderer (`vector_renderer.py`)
- SVG output
- Scalable graphics
- Path and point rendering
- Multi-layer support

### 7. Export (`core/export.py`)

**Features**:
- Metadata JSON export
- NumPy binary (.npy)
- CSV export
- Configuration save/load
- Batch export
- HTML gallery generation

### 8. Utils (`core/utils.py`)

**Components**:
- `SeededRNG`: Deterministic randomness
- `NoiseGenerator`: Perlin/Simplex noise
- Helper functions: lerp, smoothstep, remap
- Geometry: distance, curvature, velocity
- Filters: jitter, blending

## Data Flow

### Generation Pipeline

```
User Input
    ↓
Parse Expression → Validate
    ↓
Create Generator → Configure Parameters
    ↓
Generate Points → Evaluate (vectorized)
    ↓
Post-process → Normalize, Transform
    ↓
Color Mapping → Assign colors
    ↓
Render → Static/Animation/Vector
    ↓
Export → Save with metadata
```

### Example Flow: Custom Equation

```python
# 1. Parse
expr = parse_equation("sin(x*y) + cos(x^2 - y^2)")

# 2. Create generator
gen = CustomEquation(expr, x_range=(-3,3), y_range=(-3,3))

# 3. Generate points
points = gen.generate()  # Shape: (n, 3) with (x, y, z)

# 4. Render
renderer = StaticRenderer(colormap="viridis")
renderer.render_with_function_colors(points, mode="value")

# 5. Export
renderer.save("output.png")
```

## Performance Considerations

### Optimization Strategies

1. **Vectorization**: Use NumPy operations instead of loops
2. **Caching**: Compiled functions cached in ParsedExpression
3. **Lazy Evaluation**: Points generated only when needed
4. **Batch Processing**: Process multiple points simultaneously

### Scaling

- **Points**: Handles 10⁶+ points efficiently
- **Frames**: Animation renderer streams frames
- **Memory**: Generators yield points incrementally

## Extensibility

### Adding New Generators

```python
from core.generators.base import ArtGenerator

class MyGenerator(ArtGenerator):
    id = "my_generator"
    name = "My Generator"
    
    def __init__(self, param1, param2, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
    
    def generate(self) -> NDArray:
        # Generate points
        points = ...
        return points
```

### Adding New Colormaps

```python
from core.color_maps import create_custom_colormap

my_cmap = create_custom_colormap(
    colors=['#000033', '#0077be', '#ffffff'],
    name="my_palette"
)
```

### Adding New Renderers

Subclass `StaticRenderer` or create new renderer following the interface:
- `render(points, **kwargs)`: Render points
- `save(filename)`: Save output
- `close()`: Cleanup

## Determinism

### Reproducibility

All art generation is deterministic with fixed seeds:

```python
gen = Spirograph(R=5, r=3, seed=42)
points1 = gen.generate()
points2 = gen.generate()
# points1 == points2  ✓

gen2 = Spirograph(R=5, r=3, seed=42)
points3 = gen2.generate()
# points1 == points3  ✓
```

Seeds propagate through:
- RNG in `SeededRNG`
- Noise in `NoiseGenerator`
- Random sampling in generators

## Dependencies

### Core
- **NumPy**: Numerical computing
- **SymPy**: Symbolic mathematics
- **Matplotlib**: Plotting and colormaps

### Optional
- **Numba**: JIT compilation (performance)
- **FFmpeg**: Video export (animations)
- **noise**: Perlin/Simplex noise

## Testing Strategy

See `tests/` directory for comprehensive test suite:
- Unit tests for all modules
- Integration tests for pipelines
- Determinism tests (seed reproducibility)
- Performance benchmarks

## Future Enhancements

Potential additions:
- 3D surface rendering
- Fractal generators (Mandelbrot, Julia)
- Audio-reactive generation
- Real-time parameter tweaking (web UI)
- GPU acceleration
- TikZ/LaTeX export

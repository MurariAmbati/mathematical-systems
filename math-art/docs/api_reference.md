# Math Art Generator - API Reference

Complete API documentation for all modules.

## Core Modules

### `core.parser`

#### `parse_equation(expr_str, validate=True, check_domain=True)`

Parse a mathematical expression safely.

**Parameters:**
- `expr_str` (str): Mathematical expression (e.g., "sin(x*y)")
- `validate` (bool): Whether to validate expression
- `check_domain` (bool): Check for domain issues

**Returns:**
- `ParsedExpression`: Compiled expression object

**Example:**
```python
expr = parse_equation("sin(x*y) + cos(x**2 - y**2)")
result = expr.evaluate(x=1.0, y=2.0)
```

#### `ParsedExpression`

Represents a parsed mathematical expression.

**Methods:**
- `compile(backend="numpy")`: Compile to callable function
- `evaluate(**kwargs)`: Evaluate with given variables
- `__repr__()`: String representation

**Attributes:**
- `expr_str`: Original expression string
- `sympy_expr`: SymPy expression object
- `variables`: Set of variable names

---

### `core.evaluator`

#### `Evaluator(expression, backend="numpy")`

High-performance expression evaluator.

**Parameters:**
- `expression` (str or ParsedExpression): Expression to evaluate
- `backend` (str): "numpy" or "numba"

**Methods:**

##### `evaluate(**variables)`
Evaluate expression with given values.

**Parameters:**
- `**variables`: Variable names and values

**Returns:**
- Evaluation result (scalar or array)

##### `evaluate_grid(x_range, y_range, samples=100)`
Evaluate over 2D grid.

**Parameters:**
- `x_range` (tuple): (x_min, x_max)
- `y_range` (tuple): (y_min, y_max)
- `samples` (int): Grid size

**Returns:**
- `(X, Y, Z)`: Meshgrid arrays

##### `differentiate(variable, order=1)`
Compute symbolic derivative.

##### `simplify()`
Simplify expression symbolically.

---

### `core.coordinates`

#### `cartesian_grid(x_range, y_range, samples=100)`

Create 2D Cartesian grid.

**Returns:** `(X, Y)` meshgrid arrays

#### `polar_to_cartesian(r, theta)`

Convert polar to Cartesian coordinates.

**Returns:** `(x, y)`

#### `parametric_to_points(fx, fy, t_range, samples=1000, fz=None)`

Generate points from parametric equations.

**Parameters:**
- `fx`: Function for x(t)
- `fy`: Function for y(t)
- `t_range`: (t_min, t_max)
- `samples`: Number of points
- `fz`: Optional z(t) for 3D

**Returns:** Points array (n, 2) or (n, 3)

#### `normalize_points(points, target_range=(-1, 1))`

Normalize points to target range.

---

## Generator Classes

### Base Classes

#### `ArtGenerator`

Abstract base for all generators.

**Constructor Parameters:**
- `samples` (int): Number of points
- `seed` (int): Random seed
- `**params`: Additional parameters

**Methods:**
- `generate()`: Generate points (must implement)
- `get_points(regenerate=False)`: Get cached points
- `get_config()`: Get configuration dict
- `from_config(config)`: Create from config (classmethod)

**Attributes:**
- `id`: Generator identifier
- `name`: Human-readable name
- `samples`: Number of points
- `seed`: Random seed

---

### `Spirograph`

**Module:** `core.generators.spirograph`

Generate hypotrochoid/epitrochoid curves.

**Constructor:**
```python
Spirograph(
    R=5.0,           # Radius of fixed circle
    r=3.0,           # Radius of rolling circle
    a=2.0,           # Distance to drawing point
    mode="hypo",     # "hypo" or "epi"
    samples=5000,
    seed=None
)
```

**Example:**
```python
spiro = Spirograph(R=8, r=1, a=4, mode="hypo", samples=5000)
points = spiro.generate()
```

---

### `Lissajous`

**Module:** `core.generators.lissajous`

Generate Lissajous curves.

**Constructor:**
```python
Lissajous(
    A=1.0,           # X amplitude
    B=1.0,           # Y amplitude
    a=3.0,           # X frequency
    b=2.0,           # Y frequency
    delta=pi/2,      # Phase shift
    samples=2000,
    seed=None
)
```

**Example:**
```python
lissajous = Lissajous(a=3, b=2, delta=np.pi/2, samples=5000)
points = lissajous.generate()
```

---

### `LorenzAttractor`

**Module:** `core.generators.attractors`

Generate Lorenz attractor.

**Constructor:**
```python
LorenzAttractor(
    sigma=10.0,      # Prandtl number
    rho=28.0,        # Rayleigh number
    beta=8/3,        # Geometric factor
    dt=0.01,         # Time step
    initial=(1,1,1), # Initial position
    iterations=10000,
    seed=None
)
```

---

### `PolarPattern`

**Module:** `core.generators.polar_patterns`

Generate patterns from polar equations.

**Constructor:**
```python
PolarPattern(
    expr="sin(5*theta)",  # Polar equation
    theta_range=(0, 2*pi),
    samples=2000,
    time_param=0.0,       # For animations
    seed=None
)
```

**Example:**
```python
pattern = PolarPattern("sin(5*theta)", samples=2000)
points = pattern.generate()
```

---

### `CustomEquation`

**Module:** `core.generators.custom_equation`

Generate from custom expressions.

**Constructor:**
```python
CustomEquation(
    expr="sin(x*y)",     # Expression
    x_range=(-3, 3),
    y_range=(-3, 3),
    samples=100,
    threshold=None,       # For filtering
    mode="grid",          # "grid", "contour", "scatter"
    seed=None
)
```

---

## Renderers

### `StaticRenderer`

**Module:** `core.renderers.static_renderer`

Render to static images.

**Constructor:**
```python
StaticRenderer(
    width=1024,
    height=1024,
    dpi=100,
    colormap="viridis",
    background="black",
    style="scatter"      # "scatter", "line", "density"
)
```

**Methods:**

#### `render(points, colors=None, sizes=None, alpha=0.6, **kwargs)`

Render points to figure.

**Parameters:**
- `points`: Array (n, 2) or (n, 3)
- `colors`: Optional color array
- `sizes`: Point sizes
- `alpha`: Opacity

#### `render_with_function_colors(points, mode="value", alpha=0.6, **kwargs)`

Render with function-based coloring.

**Modes:**
- `"value"`: Color by function value
- `"magnitude"`: Color by absolute value
- `"curvature"`: Color by curvature
- `"velocity"`: Color by velocity

#### `save(filename, dpi=None, transparent=False, **kwargs)`

Save rendered image.

#### `show()`

Display image interactively.

---

### `AnimationRenderer`

**Module:** `core.renderers.animation_renderer`

Render animations.

**Constructor:**
```python
AnimationRenderer(
    generator,           # Generator or callable
    frames=120,
    fps=30,
    width=1024,
    height=1024,
    dpi=100,
    colormap="viridis",
    background="black"
)
```

**Methods:**

#### `render_to_video(filename, codec="libx264", bitrate=5000, **kwargs)`

Render to video file (MP4).

#### `render_to_gif(filename, optimize=True, **kwargs)`

Render to GIF.

#### `render_to_frames(output_dir, name_pattern="frame_{:04d}.png")`

Render to individual frames.

---

### `VectorRenderer`

**Module:** `core.renderers.vector_renderer`

Render to SVG.

**Constructor:**
```python
VectorRenderer(
    width=1000,
    height=1000,
    viewbox=None,
    background="black",
    colormap="viridis"
)
```

**Methods:**

#### `render_points(points, radius=1.0, colors=None, alpha=0.8)`

Render as circles.

#### `render_path(points, stroke_width=1.0, stroke_color="white", fill="none", alpha=0.8, closed=False)`

Render as path/polyline.

#### `render_gradient_path(points, stroke_width=1.0, alpha=0.8)`

Render with gradient coloring.

#### `save(filename)`

Save SVG to file.

---

## Export Utilities

### `core.export`

#### `to_image(points, filename, width=1024, height=1024, dpi=100, **kwargs)`

Quick export to image.

#### `to_svg(points, filename, width=1000, height=1000, render_as="path", **kwargs)`

Quick export to SVG.

#### `to_video(generator, filename, frames=120, fps=30, **kwargs)`

Quick export to video.

#### `to_json(config, filename, indent=2)`

Export configuration to JSON.

#### `batch_export(generators, output_dir="output", formats=["png", "svg"], **kwargs)`

Batch export multiple generators.

**Example:**
```python
generators = [
    ("spiro1", Spirograph(R=5, r=3)),
    ("spiro2", Spirograph(R=7, r=2)),
]
batch_export(generators, formats=["png", "svg"])
```

#### `create_gallery(output_dir, title="Math Art Gallery", html_file="gallery.html")`

Create HTML gallery from images.

---

### `ArtExporter`

Exporter with metadata support.

**Constructor:**
```python
ArtExporter(output_dir="output")
```

**Methods:**

#### `export_with_metadata(points, filename, generator=None, expression=None, metadata=None)`

Export with JSON metadata.

#### `save_points_numpy(points, filename)`

Save as NumPy binary (.npy).

#### `save_points_csv(points, filename)`

Save as CSV.

#### `save_config(config, filename="config.json")`

Save configuration.

#### `load_config(filename="config.json")`

Load configuration.

---

## Color Maps

### `core.color_maps`

#### `ColorMap(name="viridis")`

Base colormap class.

**Methods:**
- `map(values, vmin=None, vmax=None, alpha=1.0)`: Map values to colors
- `get_colors(n_colors)`: Get n discrete colors

#### `get_palette(name)`

Get predefined palette.

**Available Palettes:**
- `"viridis"`, `"plasma"`, `"magma"`, `"inferno"`
- `"ocean"`, `"fire"`, `"forest"`, `"sunset"`
- `"twilight"`, `"rainbow"`, `"cool"`, `"hot"`

#### `create_custom_colormap(colors, positions=None, name="custom")`

Create custom gradient.

**Example:**
```python
cmap = create_custom_colormap(
    colors=['#000033', '#0077be', '#ffffff'],
    name="ocean_custom"
)
```

---

## Utilities

### `core.utils`

#### `SeededRNG(seed=None)`

Seeded random number generator.

**Methods:**
- `random(size=None)`: Random in [0, 1)
- `uniform(low, high, size=None)`: Uniform random
- `normal(loc, scale, size=None)`: Normal random
- `integers(low, high, size=None)`: Random integers

#### `NoiseGenerator(seed=None, noise_type="perlin")`

Perlin/Simplex noise generator.

**Methods:**

##### `noise_2d(x, y, octaves=1, persistence=0.5, lacunarity=2.0, scale=1.0)`

Generate 2D noise.

##### `noise_3d(x, y, z, ...)`

Generate 3D noise.

---

## Constants and Enums

### Coordinate Systems
- `"cartesian"`: (x, y, z)
- `"polar"`: (r, θ)
- `"parametric"`: (x(t), y(t), z(t))

### Rendering Styles
- `"scatter"`: Point cloud
- `"line"`: Connected path
- `"density"`: Density plot

### Color Modes
- `"value"`: By function value
- `"magnitude"`: By absolute value
- `"gradient"`: By gradient magnitude
- `"curvature"`: By curve curvature
- `"velocity"`: By velocity/speed

---

## Complete Examples

### Example 1: Basic Spirograph
```python
from core.generators import Spirograph
from core.renderers import StaticRenderer

spiro = Spirograph(R=8, r=1, a=4, samples=5000, seed=42)
points = spiro.generate()

renderer = StaticRenderer(width=2048, height=2048, colormap="magma")
renderer.render(points, alpha=0.7)
renderer.save("spirograph.png", dpi=300)
renderer.close()
```

### Example 2: Custom Equation
```python
from core.generators import CustomEquation
from core.export import to_image

gen = CustomEquation(
    expr="sin(x*y) + cos(x**2 - y**2)",
    x_range=(-3, 3),
    y_range=(-3, 3),
    samples=200
)
points = gen.generate()
to_image(points, "custom.png", colormap="twilight")
```

### Example 3: Animation
```python
from core.generators import PolarPattern
from core.renderers import AnimationRenderer

def time_varying_pattern(t):
    pattern = PolarPattern(f"sin(5*theta + {t*2*np.pi})")
    return pattern.generate()

anim = AnimationRenderer(
    generator=time_varying_pattern,
    frames=120,
    fps=30
)
anim.render_to_video("polar_anim.mp4")
```

### Example 4: Batch Export
```python
from core.export import batch_export
from core.generators import *

generators = [
    ("spirograph", Spirograph(R=8, r=1, a=4)),
    ("lissajous", Lissajous(a=3, b=2)),
    ("lorenz", LorenzAttractor(iterations=50000)),
    ("polar_rose", PolarPattern("sin(7*theta)")),
]

batch_export(generators, formats=["png", "svg", "json"])
```

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid expression or parameters
- `ImportError`: Missing dependencies
- `RuntimeError`: Generation/rendering errors

### Best Practices

1. Always use seeds for reproducibility
2. Validate expressions before use
3. Handle large point counts carefully
4. Check output directory exists
5. Use try/except for file operations

---

## Performance Tips

1. **Use appropriate sample counts:**
   - Spirographs: 2000-10000
   - Attractors: 10000-100000
   - Custom equations: 100-500 (grid mode)

2. **Vectorize operations:**
   - Use NumPy arrays
   - Avoid Python loops
   - Leverage evaluator caching

3. **Optimize rendering:**
   - Lower DPI for previews
   - Use appropriate point sizes
   - Consider density plots for many points

4. **Memory management:**
   - Generate points incrementally
   - Stream frames in animations
   - Clean up renderers after use

---

## Version Compatibility

- Python: 3.11+
- NumPy: 1.24+
- SymPy: 1.12+
- Matplotlib: 3.7+

---

## See Also

- [Architecture Guide](architecture.md)
- [Function Catalog](function_catalog.md)
- [README](../README.md)
- [Examples](../examples/)

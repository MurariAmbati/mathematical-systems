# Architecture Overview

## System Design

Geometry Visualizer is architected as a modular system with three main components:

### 1. Python Backend (`geometry_visualizer`)

The core geometry engine providing:

- **Primitives**: Immutable geometric objects (Point2D/3D, Vector, Polygon, Mesh)
- **Algorithms**: Computational geometry implementations
  - Convex hull (Graham scan, monotone chain)
  - Delaunay triangulation (Bowyer-Watson)
  - Voronoi diagrams
  - Boolean operations on polygons
- **Transformations**: Affine transformations for 2D and 3D
- **Scene Management**: JSON-based scene representation
- **I/O**: Import/export (OBJ, STL, SVG, JSON)

**Design Principles:**
- Immutability: All geometric objects are immutable dataclasses
- Type safety: Full type annotations with mypy checking
- Precision: Configurable numerical tolerances
- Extensibility: Plugin system for custom algorithms

### 2. Frontend Viewer (React + Three.js)

Interactive WebGL-based visualization:

- **Scene Controller**: Programmatic API for scene management
- **3D Renderer**: Three.js/React Three Fiber for WebGL rendering
- **UI Components**: React-based control panels and tools
- **Algorithm Visualization**: Step-by-step algorithm playback

**Key Technologies:**
- React 18 with TypeScript
- Three.js via @react-three/fiber
- Vite for fast development and building
- Zustand for state management (optional)

### 3. Communication Layer

**JSON Scene Schema:**
- Strict JSON schema for scene interchange
- Versioned format for compatibility
- Rich metadata support
- Camera and render configuration

## Data Flow

```
┌─────────────────┐
│  Python Backend │
│                 │
│  - Compute      │
│  - Generate     │
│  - Transform    │
└────────┬────────┘
         │
         │ JSON Scene
         ↓
┌─────────────────┐
│ Scene Controller│
│  (TypeScript)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  React + Three  │
│   (Frontend)    │
│                 │
│  - Render       │
│  - Interact     │
│  - Animate      │
└─────────────────┘
```

## Module Organization

### Backend Structure

```
geometry_visualizer/
├── primitives.py       # Core geometric types
├── transforms.py       # Affine transformations
├── utils.py            # Numerical utilities
├── scene.py            # Scene data model
├── io.py               # Import/export
└── algorithms/
    ├── convex_hull.py
    ├── delaunay.py
    ├── voronoi.py
    └── boolean_ops.py
```

### Frontend Structure

```
src/
├── types/
│   └── scene.ts        # TypeScript types
├── controllers/
│   └── SceneController.ts
├── components/
│   ├── SceneViewer.tsx
│   ├── SceneObjects.tsx
│   └── ControlPanel.tsx
└── App.tsx
```

## Extension Points

### 1. Algorithm Plugins

Implement the `AlgorithmPlugin` interface:

```python
class CustomAlgorithm:
    @staticmethod
    def run(inputs, options):
        # Compute
        return StepSequence(steps)
```

### 2. Exporter Plugins

Implement the `ExporterPlugin` interface:

```python
class CustomExporter:
    @staticmethod
    def export(geometry, filename, options):
        # Export logic
        pass
```

### 3. Custom Primitives

Extend base primitive types:

```python
@dataclass(frozen=True)
class Circle2D:
    center: Point2D
    radius: float
    
    def to_dict(self):
        return {
            "center": self.center.to_dict(),
            "radius": self.radius
        }
```

## Performance Considerations

### Backend
- NumPy arrays for bulk operations
- Optional Rust/C++ acceleration for heavy compute
- Incremental algorithms where possible

### Frontend
- Level-of-detail (LOD) for large scenes
- Instanced rendering for repeated geometry
- Web Workers for heavy client-side computation
- Lazy loading of scene objects

## Testing Strategy

- **Unit tests**: pytest for Python, Vitest for TypeScript
- **Integration tests**: Scene round-trip tests
- **Regression tests**: Golden file comparisons
- **Performance tests**: Benchmark suite with target metrics

## Deployment

### Development
```bash
# Backend
pip install -e ".[dev]"
pytest

# Frontend
npm install
npm run dev
```

### Production
```bash
# Backend package
python -m build
python -m twine upload dist/*

# Frontend build
npm run build
# Deploy dist/ to static hosting
```

## Security

- Input validation for all user-supplied geometry
- Sandboxed execution for custom algorithms
- CORS configuration for API endpoints
- Content Security Policy for frontend

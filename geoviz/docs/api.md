# API Reference

## Python API

### Core Primitives

#### Point2D

```python
from geometry_visualizer import Point2D

# Create a 2D point
p = Point2D(x=3.0, y=4.0)

# Methods
distance = p.distance_to(other_point)  # Euclidean distance
array = p.to_array()  # Convert to numpy array
dict_repr = p.to_dict()  # Serialize to dictionary
p2 = Point2D.from_dict(dict_repr)  # Deserialize
```

#### Vector2D

```python
from geometry_visualizer import Vector2D

v = Vector2D(x=3.0, y=4.0)

# Vector operations
length = v.length()  # Magnitude
unit = v.normalize()  # Unit vector
dot = v.dot(other_vector)  # Dot product
cross = v.cross(other_vector)  # 2D cross product (scalar)
rotated = v.rotate(angle)  # Rotate by angle (radians)

# Arithmetic
v3 = v1 + v2  # Addition
v3 = v1 - v2  # Subtraction
v3 = v1 * 2.0  # Scalar multiplication
```

#### Polygon2D

```python
from geometry_visualizer import Polygon2D, Point2D

vertices = (
    Point2D(0, 0),
    Point2D(4, 0),
    Point2D(4, 3),
    Point2D(0, 3),
)
poly = Polygon2D(vertices)

# With holes
holes = ((Point2D(1, 1), Point2D(2, 1), Point2D(2, 2), Point2D(1, 2)),)
poly_with_hole = Polygon2D(vertices, holes)

# Methods
area = poly.area()  # Signed area (shoelace formula)
perimeter = poly.perimeter()  # Total edge length
centroid = poly.centroid()  # Center of mass
min_pt, max_pt = poly.bounding_box()  # Axis-aligned bounding box
```

#### Mesh3D

```python
from geometry_visualizer import Mesh3D
import numpy as np

vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
], dtype=np.float64)

faces = np.array([
    [0, 1, 2],
], dtype=np.int32)

mesh = Mesh3D(vertices, faces)

# Compute vertex normals
mesh_with_normals = mesh.compute_normals()

# Bounding box
min_pt, max_pt = mesh.bounding_box()
```

### Transformations

#### Transform2D

```python
from geometry_visualizer.transforms import Transform2D
from geometry_visualizer import Point2D, Vector2D

# Create transformations
identity = Transform2D.identity()
translate = Transform2D.translation(tx=5.0, ty=3.0)
rotate = Transform2D.rotation(angle=math.pi/4)  # Radians
scale = Transform2D.scaling(sx=2.0, sy=2.0)
shear = Transform2D.shear(shx=0.5, shy=0.0)

# Compose transformations (applied right to left)
combined = translate.compose(rotate).compose(scale)

# Apply to geometry
point = Point2D(1.0, 0.0)
transformed = transform.apply_to_point(point)

vector = Vector2D(1.0, 0.0)
transformed_v = transform.apply_to_vector(vector)

# Inverse
inv_transform = transform.inverse()
```

#### Transform3D

```python
from geometry_visualizer.transforms import Transform3D
from geometry_visualizer import Point3D, Vector3D

# Rotation around axes
rx = Transform3D.rotation_x(angle=math.pi/2)
ry = Transform3D.rotation_y(angle=math.pi/2)
rz = Transform3D.rotation_z(angle=math.pi/2)

# Rotation around arbitrary axis
axis = Vector3D(1, 1, 0).normalize()
r = Transform3D.rotation_axis(axis, angle=math.pi/4)

# Translation and scaling
t = Transform3D.translation(tx=5, ty=3, tz=1)
s = Transform3D.scaling(sx=2, sy=2, sz=2)

# Apply
point3d = Point3D(1, 2, 3)
transformed = transform.apply_to_point(point3d)
```

### Algorithms

#### Convex Hull

```python
from geometry_visualizer.algorithms import convex_hull_2d, convex_hull_stepwise
from geometry_visualizer import Point2D

points = [
    Point2D(0, 0),
    Point2D(4, 0),
    Point2D(2, 3),
    Point2D(1, 1),  # Interior point
]

# Compute convex hull
hull = convex_hull_2d(points, algorithm="monotone_chain")  # or "graham_scan"

# Access results
print(f"Hull vertices: {len(hull.hull_points)}")
print(f"Area: {hull.area()}")
print(f"Perimeter: {hull.perimeter()}")

# Convert to polygon
polygon = hull.to_polygon()

# Step-by-step visualization
steps = convex_hull_stepwise(points)
for step in steps:
    print(step["description"])
```

#### Delaunay Triangulation

```python
from geometry_visualizer.algorithms import delaunay_triangulation
from geometry_visualizer import Point2D

points = [Point2D(x, y) for x, y in [(0,0), (1,0), (0.5,1), (1.5,0.5)]]

# Compute triangulation
dt = delaunay_triangulation(points)

# Access results
print(f"Triangles: {dt.triangles}")
edges = dt.get_edges()  # Unique edges

# Convert to 3D mesh (z=0)
mesh = dt.to_mesh()
```

#### Voronoi Diagram

```python
from geometry_visualizer.algorithms import voronoi_diagram
from geometry_visualizer import Point2D

points = [Point2D(x, y) for x, y in [(0,0), (1,0), (0.5,1)]]

# Compute Voronoi diagram
vd = voronoi_diagram(points)

# Access cells
for cell in vd.cells:
    print(f"Site {cell.site_index}: {len(cell.vertices)} vertices")
    if cell.is_bounded:
        poly = cell.to_polygon()
```

#### Boolean Operations

```python
from geometry_visualizer.algorithms import (
    polygon_union,
    polygon_intersection,
    polygon_difference,
    point_in_polygon,
)
from geometry_visualizer import Polygon2D, Point2D

poly1 = Polygon2D(...)
poly2 = Polygon2D(...)

# Boolean operations (return list of polygons)
union = polygon_union(poly1, poly2)
intersection = polygon_intersection(poly1, poly2)
difference = polygon_difference(poly1, poly2)

# Point-in-polygon test
point = Point2D(0.5, 0.5)
is_inside = point_in_polygon(point, poly1)
```

### Scene Management

```python
from geometry_visualizer import Scene, SceneObject

# Create scene
scene = Scene()
scene.metadata.description = "My geometry scene"
scene.metadata.units = "m"

# Add objects
scene.add_object(
    id="my_polygon",
    type="polygon2d",
    geometry=polygon.to_dict(),
    style={
        "fill": "#3b82f6",
        "stroke": "#1e40af",
        "opacity": 0.7,
    }
)

# Get/update objects
obj = scene.get_object("my_polygon")
scene.remove_object("my_polygon")

# Configure camera
scene.camera.position = [10, 10, 10]
scene.camera.target = [0, 0, 0]
scene.camera.projection = "perspective"

# Export/import
scene.export_json("my_scene.json")
loaded = Scene.import_json("my_scene.json")
```

### Import/Export

```python
from geometry_visualizer.io import (
    export_obj, import_obj,
    export_stl, export_svg,
)

# OBJ format
export_obj(mesh, "model.obj")
mesh = import_obj("model.obj")

# STL format
export_stl(mesh, "model.stl", binary=False)
export_stl(mesh, "model_binary.stl", binary=True)

# SVG format
export_svg([polygon1, polygon2], "drawing.svg", width=800, height=600)
```

---

## TypeScript/JavaScript API

### Scene Controller

```typescript
import { sceneController } from './controllers/SceneController';
import type { SceneJSON, SceneObject } from './types/scene';

// Load a scene
const result = await sceneController.loadScene(sceneData);
if (result.success) {
  console.log('Scene loaded');
} else {
  console.error(result.error.message);
}

// Add object
sceneController.addObject({
  id: 'cube1',
  type: 'mesh3d',
  geometry: {...},
  style: { color: '#3b82f6' },
  visible: true,
});

// Update object
sceneController.updateObject('cube1', {
  style: { color: '#ff0000' },
});

// Remove object
sceneController.removeObject('cube1');

// Set camera
sceneController.setCamera('perspective', {
  position: [10, 10, 10],
  fov: 50,
});

// Subscribe to changes
const unsubscribe = sceneController.subscribe((scene) => {
  console.log('Scene updated', scene);
});
```

### Scene Types

```typescript
import type {
  Point2D,
  Point3D,
  Polygon2D,
  Mesh3D,
  SceneObject,
  SceneJSON,
} from './types/scene';

const point: Point2D = { x: 3, y: 4 };

const polygon: Polygon2D = {
  vertices: [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ],
  holes: [],
};

const mesh: Mesh3D = {
  vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
  faces: [[0, 1, 2]],
};
```

### React Components

```tsx
import { SceneViewer } from './components/SceneViewer';
import { SceneController } from './controllers/SceneController';

function App() {
  const [scene, setScene] = useState<SceneJSON | null>(null);

  useEffect(() => {
    const unsubscribe = sceneController.subscribe(setScene);
    sceneController.createEmptyScene();
    return unsubscribe;
  }, []);

  return <SceneViewer scene={scene} />;
}
```

---

## Error Handling

All operations return `Result<T>` types:

```python
# Python
result = some_operation()
if result.success:
    data = result.data
else:
    print(f"Error: {result.error.message}")
```

```typescript
// TypeScript
const result = await operation();
if (result.success) {
  const data = result.data;
} else {
  console.error(result.error.code, result.error.message);
}
```

## Numerical Precision

Configure tolerance for floating-point comparisons:

```python
from geometry_visualizer.utils import EPSILON

# Default tolerance is 1e-10
# For custom tolerance, pass it to functions:
distance = point1.distance_to(point2, tolerance=1e-8)
```

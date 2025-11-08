"""
Core geometric primitives with immutable dataclasses and robust operations.

All primitives support exact arithmetic where possible and provide
tolerance-based equality checks for floating-point operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt

from geometry_visualizer.utils import EPSILON, nearly_equal


@dataclass(frozen=True)
class Point2D:
    """
    Immutable 2D point in Cartesian coordinates.
    
    Attributes:
        x: X-coordinate
        y: Y-coordinate
    
    Example:
        >>> p = Point2D(3.0, 4.0)
        >>> p.distance_to(Point2D(0.0, 0.0))
        5.0
    """
    x: float
    y: float
    
    def distance_to(self, other: Point2D, tolerance: float = EPSILON) -> float:
        """Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point2D):
            return NotImplemented
        return nearly_equal(self.x, other.x) and nearly_equal(self.y, other.y)
    
    def __hash__(self) -> int:
        return hash((round(self.x / EPSILON), round(self.y / EPSILON)))
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert to numpy array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float64)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"x": self.x, "y": self.y}
    
    @classmethod
    def from_dict(cls, data: dict) -> Point2D:
        """Deserialize from dictionary."""
        return cls(x=data["x"], y=data["y"])


@dataclass(frozen=True)
class Point3D:
    """
    Immutable 3D point in Cartesian coordinates.
    
    Attributes:
        x: X-coordinate
        y: Y-coordinate
        z: Z-coordinate
    """
    x: float
    y: float
    z: float
    
    def distance_to(self, other: Point3D, tolerance: float = EPSILON) -> float:
        """Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point3D):
            return NotImplemented
        return (
            nearly_equal(self.x, other.x)
            and nearly_equal(self.y, other.y)
            and nearly_equal(self.z, other.z)
        )
    
    def __hash__(self) -> int:
        return hash((
            round(self.x / EPSILON),
            round(self.y / EPSILON),
            round(self.z / EPSILON)
        ))
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: dict) -> Point3D:
        """Deserialize from dictionary."""
        return cls(x=data["x"], y=data["y"], z=data["z"])


@dataclass(frozen=True)
class Vector2D:
    """
    Immutable 2D vector with common vector operations.
    
    Attributes:
        x: X-component
        y: Y-component
    """
    x: float
    y: float
    
    def length(self) -> float:
        """Euclidean length (magnitude) of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self) -> Vector2D:
        """Return unit vector in same direction. Returns zero vector if length is zero."""
        length = self.length()
        if length < EPSILON:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / length, self.y / length)
    
    def dot(self, other: Vector2D) -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def cross(self, other: Vector2D) -> float:
        """2D cross product (z-component of 3D cross product)."""
        return self.x * other.y - self.y * other.x
    
    def rotate(self, angle: float) -> Vector2D:
        """
        Rotate vector by angle (in radians) counter-clockwise.
        
        Args:
            angle: Rotation angle in radians
        
        Returns:
            Rotated vector
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> Vector2D:
        return self.__mul__(scalar)
    
    def __neg__(self) -> Vector2D:
        return Vector2D(-self.x, -self.y)
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert to numpy array."""
        return np.array([self.x, self.y], dtype=np.float64)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"x": self.x, "y": self.y}
    
    @classmethod
    def from_points(cls, p1: Point2D, p2: Point2D) -> Vector2D:
        """Create vector from p1 to p2."""
        return cls(p2.x - p1.x, p2.y - p1.y)


@dataclass(frozen=True)
class Vector3D:
    """
    Immutable 3D vector with common vector operations.
    
    Attributes:
        x: X-component
        y: Y-component
        z: Z-component
    """
    x: float
    y: float
    z: float
    
    def length(self) -> float:
        """Euclidean length (magnitude) of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self) -> Vector3D:
        """Return unit vector in same direction. Returns zero vector if length is zero."""
        length = self.length()
        if length < EPSILON:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / length, self.y / length, self.z / length)
    
    def dot(self, other: Vector3D) -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: Vector3D) -> Vector3D:
        """Cross product with another vector."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> Vector3D:
        return self.__mul__(scalar)
    
    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_points(cls, p1: Point3D, p2: Point3D) -> Vector3D:
        """Create vector from p1 to p2."""
        return cls(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)


@dataclass(frozen=True)
class Line2D:
    """
    Immutable 2D line in implicit form: ax + by + c = 0.
    
    The line is normalized so that a² + b² = 1 for numerical stability.
    
    Attributes:
        a: Coefficient of x
        b: Coefficient of y
        c: Constant term
    """
    a: float
    b: float
    c: float
    
    def __post_init__(self) -> None:
        """Validate and normalize the line equation."""
        norm = math.sqrt(self.a * self.a + self.b * self.b)
        if norm < EPSILON:
            raise ValueError("Line coefficients a and b cannot both be zero")
        # Normalize (using object.__setattr__ because frozen=True)
        object.__setattr__(self, 'a', self.a / norm)
        object.__setattr__(self, 'b', self.b / norm)
        object.__setattr__(self, 'c', self.c / norm)
    
    @classmethod
    def from_points(cls, p1: Point2D, p2: Point2D) -> Line2D:
        """
        Create line passing through two points.
        
        Args:
            p1: First point
            p2: Second point
        
        Returns:
            Line passing through both points
        
        Raises:
            ValueError: If points are coincident
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx * dx + dy * dy)
        if length < EPSILON:
            raise ValueError("Cannot create line from coincident points")
        # Line perpendicular to direction vector
        a = -dy
        b = dx
        c = -(a * p1.x + b * p1.y)
        return cls(a, b, c)
    
    def distance_to_point(self, point: Point2D) -> float:
        """
        Signed distance from point to line.
        
        Positive if point is on the side of the line in the direction of (a, b).
        """
        return abs(self.a * point.x + self.b * point.y + self.c)
    
    def normal(self) -> Vector2D:
        """Unit normal vector to the line."""
        return Vector2D(self.a, self.b)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"a": self.a, "b": self.b, "c": self.c}


@dataclass(frozen=True)
class Segment2D:
    """
    Immutable 2D line segment defined by two endpoints.
    
    Attributes:
        p1: First endpoint
        p2: Second endpoint
    """
    p1: Point2D
    p2: Point2D
    
    def length(self) -> float:
        """Length of the segment."""
        return self.p1.distance_to(self.p2)
    
    def midpoint(self) -> Point2D:
        """Midpoint of the segment."""
        return Point2D(
            (self.p1.x + self.p2.x) / 2.0,
            (self.p1.y + self.p2.y) / 2.0
        )
    
    def direction(self) -> Vector2D:
        """Direction vector from p1 to p2."""
        return Vector2D.from_points(self.p1, self.p2)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "p1": self.p1.to_dict(),
            "p2": self.p2.to_dict()
        }


@dataclass(frozen=True)
class Segment3D:
    """
    Immutable 3D line segment defined by two endpoints.
    
    Attributes:
        p1: First endpoint
        p2: Second endpoint
    """
    p1: Point3D
    p2: Point3D
    
    def length(self) -> float:
        """Length of the segment."""
        return self.p1.distance_to(self.p2)
    
    def midpoint(self) -> Point3D:
        """Midpoint of the segment."""
        return Point3D(
            (self.p1.x + self.p2.x) / 2.0,
            (self.p1.y + self.p2.y) / 2.0,
            (self.p1.z + self.p2.z) / 2.0
        )
    
    def direction(self) -> Vector3D:
        """Direction vector from p1 to p2."""
        return Vector3D.from_points(self.p1, self.p2)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "p1": self.p1.to_dict(),
            "p2": self.p2.to_dict()
        }


@dataclass(frozen=True)
class Plane3D:
    """
    Immutable 3D plane in implicit form: ax + by + cz + d = 0.
    
    The plane is normalized so that a² + b² + c² = 1.
    
    Attributes:
        a: Coefficient of x
        b: Coefficient of y
        c: Coefficient of z
        d: Constant term
    """
    a: float
    b: float
    c: float
    d: float
    
    def __post_init__(self) -> None:
        """Validate and normalize the plane equation."""
        norm = math.sqrt(self.a * self.a + self.b * self.b + self.c * self.c)
        if norm < EPSILON:
            raise ValueError("Plane coefficients a, b, and c cannot all be zero")
        # Normalize
        object.__setattr__(self, 'a', self.a / norm)
        object.__setattr__(self, 'b', self.b / norm)
        object.__setattr__(self, 'c', self.c / norm)
        object.__setattr__(self, 'd', self.d / norm)
    
    @classmethod
    def from_points(cls, p1: Point3D, p2: Point3D, p3: Point3D) -> Plane3D:
        """
        Create plane passing through three non-collinear points.
        
        Args:
            p1, p2, p3: Three non-collinear points
        
        Returns:
            Plane passing through all three points
        
        Raises:
            ValueError: If points are collinear
        """
        v1 = Vector3D.from_points(p1, p2)
        v2 = Vector3D.from_points(p1, p3)
        normal = v1.cross(v2)
        
        if normal.length() < EPSILON:
            raise ValueError("Cannot create plane from collinear points")
        
        normal = normal.normalize()
        d = -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z)
        return cls(normal.x, normal.y, normal.z, d)
    
    def distance_to_point(self, point: Point3D) -> float:
        """Signed distance from point to plane."""
        return abs(self.a * point.x + self.b * point.y + self.c * point.z + self.d)
    
    def normal(self) -> Vector3D:
        """Unit normal vector to the plane."""
        return Vector3D(self.a, self.b, self.c)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}


@dataclass(frozen=True)
class Polygon2D:
    """
    Immutable 2D polygon defined by ordered vertices.
    
    Supports polygons with holes. The exterior ring is counter-clockwise,
    holes are clockwise.
    
    Attributes:
        vertices: Ordered list of vertices (exterior ring)
        holes: List of hole polygons (each with clockwise vertices)
    """
    vertices: Tuple[Point2D, ...]
    holes: Tuple[Tuple[Point2D, ...], ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        """Validate polygon."""
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
    
    def area(self) -> float:
        """
        Compute signed area using the shoelace formula.
        
        Positive area indicates counter-clockwise orientation.
        """
        area = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        area = area / 2.0
        
        # Subtract hole areas
        for hole in self.holes:
            hole_poly = Polygon2D(hole)
            area -= abs(hole_poly.area())
        
        return area
    
    def perimeter(self) -> float:
        """Compute perimeter (sum of edge lengths)."""
        perim = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            perim += self.vertices[i].distance_to(self.vertices[j])
        
        for hole in self.holes:
            hole_poly = Polygon2D(hole)
            perim += hole_poly.perimeter()
        
        return perim
    
    def centroid(self) -> Point2D:
        """Compute centroid (center of mass) for simple polygon."""
        if len(self.vertices) == 0:
            raise ValueError("Cannot compute centroid of empty polygon")
        
        cx = 0.0
        cy = 0.0
        area = 0.0
        n = len(self.vertices)
        
        for i in range(n):
            j = (i + 1) % n
            cross = (self.vertices[i].x * self.vertices[j].y - 
                    self.vertices[j].x * self.vertices[i].y)
            area += cross
            cx += (self.vertices[i].x + self.vertices[j].x) * cross
            cy += (self.vertices[i].y + self.vertices[j].y) * cross
        
        area = area / 2.0
        if abs(area) < EPSILON:
            # Degenerate polygon, return average of vertices
            cx = sum(v.x for v in self.vertices) / len(self.vertices)
            cy = sum(v.y for v in self.vertices) / len(self.vertices)
            return Point2D(cx, cy)
        
        cx = cx / (6.0 * area)
        cy = cy / (6.0 * area)
        return Point2D(cx, cy)
    
    def bounding_box(self) -> Tuple[Point2D, Point2D]:
        """Return axis-aligned bounding box as (min_point, max_point)."""
        if not self.vertices:
            raise ValueError("Cannot compute bounding box of empty polygon")
        
        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        
        return Point2D(min_x, min_y), Point2D(max_x, max_y)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "vertices": [v.to_dict() for v in self.vertices],
            "holes": [[v.to_dict() for v in hole] for hole in self.holes]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Polygon2D:
        """Deserialize from dictionary."""
        vertices = tuple(Point2D.from_dict(v) for v in data["vertices"])
        holes = tuple(
            tuple(Point2D.from_dict(v) for v in hole)
            for hole in data.get("holes", [])
        )
        return cls(vertices, holes)


@dataclass(frozen=True)
class Mesh3D:
    """
    Immutable 3D triangulated surface mesh.
    
    Attributes:
        vertices: Array of vertex positions, shape (N, 3)
        faces: Array of face indices (triangles), shape (M, 3)
        normals: Optional array of vertex normals, shape (N, 3)
    """
    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int32]
    normals: Optional[npt.NDArray[np.float64]] = None
    
    def __post_init__(self) -> None:
        """Validate mesh data."""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Faces must have shape (M, 3)")
        if np.any(self.faces < 0) or np.any(self.faces >= len(self.vertices)):
            raise ValueError("Face indices out of bounds")
        if self.normals is not None:
            if self.normals.shape != self.vertices.shape:
                raise ValueError("Normals must have same shape as vertices")
    
    def compute_normals(self) -> Mesh3D:
        """
        Compute per-vertex normals by averaging adjacent face normals.
        
        Returns:
            New mesh with computed normals
        """
        normals = np.zeros_like(self.vertices)
        
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            # Add to each vertex (will normalize later)
            for idx in face:
                normals[idx] += face_normal
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < EPSILON, 1.0, norms)  # Avoid division by zero
        normals = normals / norms
        
        return Mesh3D(self.vertices, self.faces, normals)
    
    def bounding_box(self) -> Tuple[Point3D, Point3D]:
        """Return axis-aligned bounding box as (min_point, max_point)."""
        if len(self.vertices) == 0:
            raise ValueError("Cannot compute bounding box of empty mesh")
        
        min_pt = self.vertices.min(axis=0)
        max_pt = self.vertices.max(axis=0)
        return Point3D(*min_pt), Point3D(*max_pt)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist(),
        }
        if self.normals is not None:
            result["normals"] = self.normals.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> Mesh3D:
        """Deserialize from dictionary."""
        vertices = np.array(data["vertices"], dtype=np.float64)
        faces = np.array(data["faces"], dtype=np.int32)
        normals = None
        if "normals" in data:
            normals = np.array(data["normals"], dtype=np.float64)
        return cls(vertices, faces, normals)

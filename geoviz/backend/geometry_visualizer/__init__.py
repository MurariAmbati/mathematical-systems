"""
Geometry Visualizer - Professional 2D/3D geometry engine and visualization platform.

This package provides robust computational geometry primitives, algorithms,
and utilities for geometric analysis and visualization.
"""

__version__ = "0.1.0"

from geometry_visualizer.primitives import (
    Point2D,
    Point3D,
    Vector2D,
    Vector3D,
    Line2D,
    Segment2D,
    Segment3D,
    Plane3D,
    Polygon2D,
    Mesh3D,
)
from geometry_visualizer.transforms import Transform2D, Transform3D
from geometry_visualizer.scene import Scene, SceneObject

__all__ = [
    "__version__",
    # Primitives
    "Point2D",
    "Point3D",
    "Vector2D",
    "Vector3D",
    "Line2D",
    "Segment2D",
    "Segment3D",
    "Plane3D",
    "Polygon2D",
    "Mesh3D",
    # Transforms
    "Transform2D",
    "Transform3D",
    # Scene
    "Scene",
    "SceneObject",
]

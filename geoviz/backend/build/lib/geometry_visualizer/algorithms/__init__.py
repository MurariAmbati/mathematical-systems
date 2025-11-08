"""Algorithms subpackage initialization."""

from geometry_visualizer.algorithms.convex_hull import ConvexHull2D, convex_hull_2d
from geometry_visualizer.algorithms.delaunay import DelaunayTriangulation, delaunay_triangulation
from geometry_visualizer.algorithms.voronoi import VoronoiDiagram, voronoi_diagram
from geometry_visualizer.algorithms.boolean_ops import (
    polygon_union,
    polygon_intersection,
    polygon_difference,
)

__all__ = [
    "ConvexHull2D",
    "convex_hull_2d",
    "DelaunayTriangulation",
    "delaunay_triangulation",
    "VoronoiDiagram",
    "voronoi_diagram",
    "polygon_union",
    "polygon_intersection",
    "polygon_difference",
]

"""
Voronoi diagram computation.

For production use, consider scipy.spatial.Voronoi.
This implementation derives Voronoi diagram from Delaunay triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

from geometry_visualizer.primitives import Point2D, Polygon2D
from geometry_visualizer.algorithms.delaunay import delaunay_triangulation, DelaunayTriangulation


@dataclass
class VoronoiCell:
    """
    A single Voronoi cell (region).
    
    Attributes:
        site_index: Index of the generating site point
        vertices: Vertices of the cell polygon (may be unbounded)
        is_bounded: Whether the cell is bounded
    """
    site_index: int
    vertices: List[Point2D]
    is_bounded: bool = True
    
    def to_polygon(self) -> Optional[Polygon2D]:
        """
        Convert to Polygon2D if bounded.
        
        Returns:
            Polygon2D or None if unbounded
        """
        if not self.is_bounded or len(self.vertices) < 3:
            return None
        return Polygon2D(tuple(self.vertices))


@dataclass
class VoronoiDiagram:
    """
    Voronoi diagram result.
    
    Attributes:
        points: Original input points (sites)
        cells: List of Voronoi cells
        vertices: All Voronoi vertices (circumcenters)
    """
    points: List[Point2D]
    cells: List[VoronoiCell]
    vertices: List[Point2D]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "points": [p.to_dict() for p in self.points],
            "vertices": [v.to_dict() for v in self.vertices],
            "cells": [
                {
                    "site_index": cell.site_index,
                    "vertices": [v.to_dict() for v in cell.vertices],
                    "is_bounded": cell.is_bounded,
                }
                for cell in self.cells
            ],
        }


def voronoi_diagram(points: List[Point2D]) -> VoronoiDiagram:
    """
    Compute Voronoi diagram from Delaunay triangulation.
    
    The Voronoi diagram is the dual of the Delaunay triangulation.
    Each Voronoi vertex is the circumcenter of a Delaunay triangle.
    
    Args:
        points: List of 2D points (sites)
    
    Returns:
        VoronoiDiagram object
    
    Raises:
        ValueError: If fewer than 3 points provided
    
    Note:
        This implementation only handles bounded regions properly.
        Unbounded regions are marked but vertices may be incomplete.
    
    Example:
        >>> points = [Point2D(0, 0), Point2D(1, 0), Point2D(0.5, 1)]
        >>> vd = voronoi_diagram(points)
        >>> len(vd.cells)
        3
    """
    if len(points) < 3:
        raise ValueError("Voronoi diagram requires at least 3 points")
    
    # Compute Delaunay triangulation
    dt = delaunay_triangulation(points)
    
    # Compute circumcenters (Voronoi vertices)
    circumcenters: List[Point2D] = []
    triangle_to_vertex: Dict[Tuple[int, int, int], int] = {}
    
    for tri in dt.triangles:
        center = _circumcenter(
            points[tri[0]],
            points[tri[1]],
            points[tri[2]]
        )
        idx = len(circumcenters)
        circumcenters.append(center)
        triangle_to_vertex[tri] = idx
    
    # Build Voronoi cells
    cells: List[VoronoiCell] = []
    
    for site_idx in range(len(points)):
        # Find all triangles containing this site
        adjacent_triangles = [
            tri for tri in dt.triangles
            if site_idx in tri
        ]
        
        if not adjacent_triangles:
            continue
        
        # Get circumcenters of adjacent triangles (vertices of Voronoi cell)
        cell_vertices = [
            circumcenters[triangle_to_vertex[tri]]
            for tri in adjacent_triangles
        ]
        
        # Sort vertices by angle around the site (for proper polygon order)
        site = points[site_idx]
        cell_vertices_with_angles = [
            (v, np.arctan2(v.y - site.y, v.x - site.x))
            for v in cell_vertices
        ]
        cell_vertices_with_angles.sort(key=lambda x: x[1])
        sorted_vertices = [v for v, _ in cell_vertices_with_angles]
        
        # Check if cell is bounded (very simplified check)
        # A cell is unbounded if it's on the convex hull of the point set
        is_bounded = _is_interior_point(site_idx, dt)
        
        cells.append(VoronoiCell(
            site_index=site_idx,
            vertices=sorted_vertices,
            is_bounded=is_bounded
        ))
    
    return VoronoiDiagram(
        points=points,
        cells=cells,
        vertices=circumcenters
    )


def _circumcenter(p0: Point2D, p1: Point2D, p2: Point2D) -> Point2D:
    """
    Compute circumcenter of a triangle.
    
    The circumcenter is equidistant from all three vertices.
    """
    ax = p1.x - p0.x
    ay = p1.y - p0.y
    bx = p2.x - p0.x
    by = p2.y - p0.y
    
    d = 2 * (ax * by - ay * bx)
    
    if abs(d) < 1e-10:
        # Degenerate triangle, return centroid
        return Point2D(
            (p0.x + p1.x + p2.x) / 3,
            (p0.y + p1.y + p2.y) / 3
        )
    
    ux = (ay * (ax * ax + ay * ay) - ax * (bx * bx + by * by)) / d
    uy = (ax * (bx * bx + by * by) - ay * (ax * ax + ay * ay)) / d
    
    return Point2D(p0.x + ux, p0.y + uy)


def _is_interior_point(point_idx: int, dt: DelaunayTriangulation) -> bool:
    """
    Check if a point is in the interior (not on convex hull).
    
    A point is interior if all its adjacent triangles form a closed loop.
    """
    # Get all triangles containing this point
    adjacent_triangles = [
        tri for tri in dt.triangles
        if point_idx in tri
    ]
    
    # Get all edges adjacent to this point
    edges: List[Tuple[int, int]] = []
    for tri in adjacent_triangles:
        tri_list = list(tri)
        idx = tri_list.index(point_idx)
        # Get the two other vertices
        other1 = tri_list[(idx + 1) % 3]
        other2 = tri_list[(idx + 2) % 3]
        edges.append((other1, other2))
    
    # Check if each edge appears exactly twice (once in each direction)
    edge_counts: Dict[Tuple[int, int], int] = {}
    for e in edges:
        edge_counts[e] = edge_counts.get(e, 0) + 1
        edge_counts[(e[1], e[0])] = edge_counts.get((e[1], e[0]), 0) + 1
    
    # If any edge appears only once, the point is on the boundary
    return all(count == 2 for count in edge_counts.values())

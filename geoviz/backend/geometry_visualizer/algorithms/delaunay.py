"""
Delaunay triangulation for 2D point sets.

Implements Bowyer-Watson incremental algorithm for Delaunay triangulation.
For production use, consider scipy.spatial.Delaunay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple
import numpy as np

from geometry_visualizer.primitives import Point2D, Mesh3D
from geometry_visualizer.utils import in_circle, orient2d, EPSILON


@dataclass
class Triangle:
    """
    Triangle defined by three vertex indices.
    
    Attributes:
        vertices: Tuple of three vertex indices
    """
    vertices: Tuple[int, int, int]
    
    def contains_vertex(self, vertex_idx: int) -> bool:
        """Check if triangle contains a vertex."""
        return vertex_idx in self.vertices
    
    def shares_edge(self, other: Triangle) -> bool:
        """Check if this triangle shares an edge with another."""
        shared = sum(1 for v in self.vertices if v in other.vertices)
        return shared == 2
    
    def get_shared_edge(self, other: Triangle) -> Tuple[int, int]:
        """Get the shared edge between two triangles."""
        shared_vertices = [v for v in self.vertices if v in other.vertices]
        if len(shared_vertices) != 2:
            raise ValueError("Triangles do not share an edge")
        return tuple(shared_vertices)  # type: ignore


@dataclass
class DelaunayTriangulation:
    """
    Delaunay triangulation result.
    
    Attributes:
        points: Original input points
        triangles: List of triangles (each as tuple of 3 vertex indices)
    """
    points: List[Point2D]
    triangles: List[Tuple[int, int, int]]
    
    def to_mesh(self) -> Mesh3D:
        """
        Convert triangulation to a 3D mesh (with z=0).
        
        Returns:
            Mesh3D representation
        """
        vertices = np.array([[p.x, p.y, 0.0] for p in self.points], dtype=np.float64)
        faces = np.array(self.triangles, dtype=np.int32)
        return Mesh3D(vertices, faces)
    
    def get_edges(self) -> Set[Tuple[int, int]]:
        """
        Get all unique edges in the triangulation.
        
        Returns:
            Set of edges (each edge as sorted tuple of vertex indices)
        """
        edges: Set[Tuple[int, int]] = set()
        for tri in self.triangles:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted([tri[i], tri[j]]))  # type: ignore
                edges.add(edge)
        return edges
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "points": [p.to_dict() for p in self.points],
            "triangles": [list(tri) for tri in self.triangles],
        }


def delaunay_triangulation(points: List[Point2D]) -> DelaunayTriangulation:
    """
    Compute Delaunay triangulation using Bowyer-Watson algorithm.
    
    Reference: A. Bowyer, "Computing Dirichlet tessellations", 1981.
    
    Args:
        points: List of 2D points
    
    Returns:
        DelaunayTriangulation object
    
    Raises:
        ValueError: If fewer than 3 points provided
    
    Time complexity: O(n²) average case, O(n³) worst case
    For production use with large point sets, use scipy.spatial.Delaunay
    
    Example:
        >>> points = [Point2D(0, 0), Point2D(1, 0), Point2D(0.5, 1)]
        >>> dt = delaunay_triangulation(points)
        >>> len(dt.triangles)
        1
    """
    if len(points) < 3:
        raise ValueError("Delaunay triangulation requires at least 3 points")
    
    # Create a super-triangle that contains all points
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy) * 10
    
    # Super-triangle vertices (indices len(points), len(points)+1, len(points)+2)
    super_p1 = Point2D(min_x - delta_max, min_y - 1)
    super_p2 = Point2D(min_x + 3 * delta_max, min_y - 1)
    super_p3 = Point2D(min_x + delta_max, max_y + 2 * delta_max)
    
    all_points = points + [super_p1, super_p2, super_p3]
    n = len(points)
    
    # Initialize with super-triangle
    triangles = [Triangle((n, n + 1, n + 2))]
    
    # Add points one at a time
    for i in range(n):
        point = points[i]
        bad_triangles: List[Triangle] = []
        
        # Find all triangles whose circumcircle contains the point
        for tri in triangles:
            v0, v1, v2 = tri.vertices
            p0 = all_points[v0]
            p1 = all_points[v1]
            p2 = all_points[v2]
            
            if _point_in_circumcircle(point, p0, p1, p2):
                bad_triangles.append(tri)
        
        # Find the boundary of the polygonal hole
        polygon_edges: List[Tuple[int, int]] = []
        
        for tri in bad_triangles:
            for j in range(3):
                k = (j + 1) % 3
                edge = (tri.vertices[j], tri.vertices[k])
                
                # Check if edge is shared by another bad triangle
                is_shared = False
                for other_tri in bad_triangles:
                    if other_tri is tri:
                        continue
                    if edge[0] in other_tri.vertices and edge[1] in other_tri.vertices:
                        is_shared = True
                        break
                
                if not is_shared:
                    polygon_edges.append(edge)
        
        # Remove bad triangles
        for tri in bad_triangles:
            triangles.remove(tri)
        
        # Retriangulate the polygonal hole
        for edge in polygon_edges:
            triangles.append(Triangle((edge[0], edge[1], i)))
    
    # Remove triangles that share a vertex with super-triangle
    final_triangles = [
        tri for tri in triangles
        if not any(v >= n for v in tri.vertices)
    ]
    
    return DelaunayTriangulation(
        points=points,
        triangles=[tri.vertices for tri in final_triangles]
    )


def _point_in_circumcircle(
    point: Point2D,
    p0: Point2D,
    p1: Point2D,
    p2: Point2D
) -> bool:
    """
    Check if point is inside the circumcircle of triangle (p0, p1, p2).
    
    Uses the in-circle predicate with proper orientation handling.
    """
    # Ensure triangle vertices are in counter-clockwise order
    det = orient2d((p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y))
    
    if abs(det) < EPSILON:
        # Degenerate triangle
        return False
    
    if det < 0:
        # Swap to make counter-clockwise
        p1, p2 = p2, p1
    
    result = in_circle(
        (p0.x, p0.y),
        (p1.x, p1.y),
        (p2.x, p2.y),
        (point.x, point.y)
    )
    
    return result > EPSILON

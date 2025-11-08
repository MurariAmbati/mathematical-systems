"""
Convex hull algorithms for 2D and 3D point sets.

Implements Graham scan and monotone chain algorithms for 2D convex hulls.
For production use with large point sets, consider using scipy.spatial.ConvexHull.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from geometry_visualizer.primitives import Point2D, Polygon2D
from geometry_visualizer.utils import orient2d, EPSILON


@dataclass
class ConvexHull2D:
    """
    2D convex hull result.
    
    Attributes:
        points: Original input points
        hull_indices: Indices of points on the convex hull (counter-clockwise)
        hull_points: Points on the convex hull (counter-clockwise)
    """
    points: List[Point2D]
    hull_indices: List[int]
    hull_points: List[Point2D]
    
    def to_polygon(self) -> Polygon2D:
        """Convert hull to a Polygon2D."""
        return Polygon2D(tuple(self.hull_points))
    
    def area(self) -> float:
        """Compute area of the convex hull."""
        return self.to_polygon().area()
    
    def perimeter(self) -> float:
        """Compute perimeter of the convex hull."""
        return self.to_polygon().perimeter()
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "points": [p.to_dict() for p in self.points],
            "hull_indices": self.hull_indices,
            "hull_points": [p.to_dict() for p in self.hull_points],
        }


def convex_hull_2d(
    points: List[Point2D],
    algorithm: str = "monotone_chain"
) -> ConvexHull2D:
    """
    Compute 2D convex hull of a point set.
    
    Implements Andrew's monotone chain algorithm (O(n log n)).
    Reference: A. M. Andrew, "Another Efficient Algorithm for Convex Hulls in Two Dimensions", 1979.
    
    Args:
        points: List of 2D points
        algorithm: Algorithm to use ('monotone_chain' or 'graham_scan')
    
    Returns:
        ConvexHull2D object containing hull information
    
    Raises:
        ValueError: If fewer than 3 points provided
    
    Time complexity: O(n log n)
    Space complexity: O(n)
    
    Example:
        >>> points = [Point2D(0, 0), Point2D(1, 0), Point2D(0.5, 1)]
        >>> hull = convex_hull_2d(points)
        >>> len(hull.hull_points)
        3
    """
    if len(points) < 3:
        raise ValueError("Convex hull requires at least 3 points")
    
    if algorithm == "monotone_chain":
        return _monotone_chain(points)
    elif algorithm == "graham_scan":
        return _graham_scan(points)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _monotone_chain(points: List[Point2D]) -> ConvexHull2D:
    """
    Compute convex hull using Andrew's monotone chain algorithm.
    
    This algorithm is simpler and more robust than Graham scan.
    """
    # Create list of (point, original_index) tuples
    indexed_points = [(p, i) for i, p in enumerate(points)]
    
    # Sort points lexicographically (first by x, then by y)
    indexed_points.sort(key=lambda item: (item[0].x, item[0].y))
    
    # Build lower hull
    lower: List[Tuple[Point2D, int]] = []
    for p, idx in indexed_points:
        while len(lower) >= 2:
            # Check if we make a right turn (clockwise)
            p1, p2 = lower[-2][0], lower[-1][0]
            if orient2d((p1.x, p1.y), (p2.x, p2.y), (p.x, p.y)) > EPSILON:
                break
            lower.pop()
        lower.append((p, idx))
    
    # Build upper hull
    upper: List[Tuple[Point2D, int]] = []
    for p, idx in reversed(indexed_points):
        while len(upper) >= 2:
            p1, p2 = upper[-2][0], upper[-1][0]
            if orient2d((p1.x, p1.y), (p2.x, p2.y), (p.x, p.y)) > EPSILON:
                break
            upper.pop()
        upper.append((p, idx))
    
    # Remove last point of each half because it's repeated
    hull_with_indices = lower[:-1] + upper[:-1]
    
    # Extract hull points and indices
    hull_points = [p for p, _ in hull_with_indices]
    hull_indices = [idx for _, idx in hull_with_indices]
    
    return ConvexHull2D(
        points=points,
        hull_indices=hull_indices,
        hull_points=hull_points
    )


def _graham_scan(points: List[Point2D]) -> ConvexHull2D:
    """
    Compute convex hull using Graham scan algorithm.
    
    Reference: R. L. Graham, "An Efficient Algorithm for Determining the Convex Hull of a Finite Planar Set", 1972.
    """
    # Find the point with lowest y-coordinate (and leftmost if tie)
    start_idx = min(range(len(points)), key=lambda i: (points[i].y, points[i].x))
    start_point = points[start_idx]
    
    # Create list of (point, original_index, angle) tuples
    indexed_points: List[Tuple[Point2D, int, float]] = []
    
    for i, p in enumerate(points):
        if i == start_idx:
            continue
        # Compute angle from start point
        dx = p.x - start_point.x
        dy = p.y - start_point.y
        angle = np.arctan2(dy, dx)
        indexed_points.append((p, i, angle))
    
    # Sort by angle, then by distance if angles are equal
    indexed_points.sort(key=lambda item: (
        item[2],
        (item[0].x - start_point.x)**2 + (item[0].y - start_point.y)**2
    ))
    
    # Build hull
    hull: List[Tuple[Point2D, int]] = [(start_point, start_idx)]
    
    for p, idx, _ in indexed_points:
        # Remove points that make clockwise turn
        while len(hull) >= 2:
            p1, p2 = hull[-2][0], hull[-1][0]
            if orient2d((p1.x, p1.y), (p2.x, p2.y), (p.x, p.y)) > EPSILON:
                break
            hull.pop()
        hull.append((p, idx))
    
    hull_points = [p for p, _ in hull]
    hull_indices = [idx for _, idx in hull]
    
    return ConvexHull2D(
        points=points,
        hull_indices=hull_indices,
        hull_points=hull_points
    )


def convex_hull_stepwise(points: List[Point2D], algorithm: str = "monotone_chain") -> List[dict]:
    """
    Compute convex hull with stepwise snapshots for visualization.
    
    Args:
        points: List of 2D points
        algorithm: Algorithm to use
    
    Returns:
        List of step dictionaries, each containing:
            - description: Step description
            - current_hull: Current hull points
            - active_points: Points being considered
            - completed: Whether algorithm is complete
    """
    steps: List[dict] = []
    
    if algorithm == "monotone_chain":
        # Create indexed points
        indexed_points = [(p, i) for i, p in enumerate(points)]
        indexed_points.sort(key=lambda item: (item[0].x, item[0].y))
        
        steps.append({
            "description": "Sort points lexicographically",
            "sorted_points": [p.to_dict() for p, _ in indexed_points],
            "current_hull": [],
            "completed": False,
        })
        
        # Build lower hull with steps
        lower: List[Point2D] = []
        for i, (p, idx) in enumerate(indexed_points):
            lower.append(p)
            
            # Remove points that make right turn
            removed = []
            while len(lower) >= 3:
                p1, p2, p3 = lower[-3], lower[-2], lower[-1]
                if orient2d((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)) > EPSILON:
                    break
                removed.append(lower.pop(-2))
            
            steps.append({
                "description": f"Lower hull: Process point {i+1}/{len(indexed_points)}",
                "current_hull": [pt.to_dict() for pt in lower],
                "active_point": p.to_dict(),
                "removed_points": [pt.to_dict() for pt in removed],
                "completed": False,
            })
        
        # Build upper hull
        upper: List[Point2D] = []
        for i, (p, idx) in enumerate(reversed(indexed_points)):
            upper.append(p)
            
            removed = []
            while len(upper) >= 3:
                p1, p2, p3 = upper[-3], upper[-2], upper[-1]
                if orient2d((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)) > EPSILON:
                    break
                removed.append(upper.pop(-2))
            
            steps.append({
                "description": f"Upper hull: Process point {i+1}/{len(indexed_points)}",
                "current_hull": [pt.to_dict() for pt in lower + upper],
                "active_point": p.to_dict(),
                "removed_points": [pt.to_dict() for pt in removed],
                "completed": False,
            })
        
        # Final hull
        final_hull = lower[:-1] + upper[:-1]
        steps.append({
            "description": "Convex hull complete",
            "current_hull": [p.to_dict() for p in final_hull],
            "completed": True,
        })
    
    return steps

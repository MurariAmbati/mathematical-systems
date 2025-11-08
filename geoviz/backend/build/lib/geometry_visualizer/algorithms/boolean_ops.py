"""
Boolean operations on 2D polygons.

Uses shapely library for robust polygon operations.
For production use, this provides a clean wrapper around shapely.
"""

from __future__ import annotations

from typing import List, Optional
import shapely.geometry as shp
import shapely.ops as ops

from geometry_visualizer.primitives import Point2D, Polygon2D


def _polygon_to_shapely(polygon: Polygon2D) -> shp.Polygon:
    """Convert Polygon2D to shapely Polygon."""
    exterior = [(p.x, p.y) for p in polygon.vertices]
    holes = [[(p.x, p.y) for p in hole] for hole in polygon.holes]
    return shp.Polygon(exterior, holes)


def _shapely_to_polygon(shapely_poly: shp.Polygon) -> Polygon2D:
    """Convert shapely Polygon to Polygon2D."""
    exterior_coords = list(shapely_poly.exterior.coords[:-1])  # Remove duplicate last point
    vertices = tuple(Point2D(x, y) for x, y in exterior_coords)
    
    holes = []
    for interior in shapely_poly.interiors:
        hole_coords = list(interior.coords[:-1])
        holes.append(tuple(Point2D(x, y) for x, y in hole_coords))
    
    return Polygon2D(vertices, tuple(holes))


def polygon_union(poly1: Polygon2D, poly2: Polygon2D) -> List[Polygon2D]:
    """
    Compute union of two polygons.
    
    Args:
        poly1: First polygon
        poly2: Second polygon
    
    Returns:
        List of polygons representing the union (may be multiple if disjoint)
    
    Example:
        >>> p1 = Polygon2D(tuple([Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1)]))
        >>> p2 = Polygon2D(tuple([Point2D(0.5, 0), Point2D(1.5, 0), Point2D(1.5, 1), Point2D(0.5, 1)]))
        >>> result = polygon_union(p1, p2)
        >>> len(result)
        1
    """
    shp1 = _polygon_to_shapely(poly1)
    shp2 = _polygon_to_shapely(poly2)
    
    result = shp1.union(shp2)
    
    # Handle different result types
    if isinstance(result, shp.Polygon):
        return [_shapely_to_polygon(result)]
    elif isinstance(result, shp.MultiPolygon):
        return [_shapely_to_polygon(p) for p in result.geoms]
    else:
        return []


def polygon_intersection(poly1: Polygon2D, poly2: Polygon2D) -> List[Polygon2D]:
    """
    Compute intersection of two polygons.
    
    Args:
        poly1: First polygon
        poly2: Second polygon
    
    Returns:
        List of polygons representing the intersection
    
    Example:
        >>> p1 = Polygon2D(tuple([Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2)]))
        >>> p2 = Polygon2D(tuple([Point2D(1, 1), Point2D(3, 1), Point2D(3, 3), Point2D(1, 3)]))
        >>> result = polygon_intersection(p1, p2)
        >>> len(result)
        1
    """
    shp1 = _polygon_to_shapely(poly1)
    shp2 = _polygon_to_shapely(poly2)
    
    result = shp1.intersection(shp2)
    
    if isinstance(result, shp.Polygon):
        if not result.is_empty:
            return [_shapely_to_polygon(result)]
    elif isinstance(result, shp.MultiPolygon):
        return [_shapely_to_polygon(p) for p in result.geoms if not p.is_empty]
    
    return []


def polygon_difference(poly1: Polygon2D, poly2: Polygon2D) -> List[Polygon2D]:
    """
    Compute difference of two polygons (poly1 - poly2).
    
    Args:
        poly1: First polygon (from which to subtract)
        poly2: Second polygon (to subtract)
    
    Returns:
        List of polygons representing poly1 - poly2
    
    Example:
        >>> p1 = Polygon2D(tuple([Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2)]))
        >>> p2 = Polygon2D(tuple([Point2D(1, 1), Point2D(3, 1), Point2D(3, 3), Point2D(1, 3)]))
        >>> result = polygon_difference(p1, p2)
        >>> len(result) >= 1
        True
    """
    shp1 = _polygon_to_shapely(poly1)
    shp2 = _polygon_to_shapely(poly2)
    
    result = shp1.difference(shp2)
    
    if isinstance(result, shp.Polygon):
        if not result.is_empty:
            return [_shapely_to_polygon(result)]
    elif isinstance(result, shp.MultiPolygon):
        return [_shapely_to_polygon(p) for p in result.geoms if not p.is_empty]
    
    return []


def polygon_symmetric_difference(poly1: Polygon2D, poly2: Polygon2D) -> List[Polygon2D]:
    """
    Compute symmetric difference of two polygons (XOR).
    
    Args:
        poly1: First polygon
        poly2: Second polygon
    
    Returns:
        List of polygons in poly1 XOR poly2
    """
    shp1 = _polygon_to_shapely(poly1)
    shp2 = _polygon_to_shapely(poly2)
    
    result = shp1.symmetric_difference(shp2)
    
    if isinstance(result, shp.Polygon):
        if not result.is_empty:
            return [_shapely_to_polygon(result)]
    elif isinstance(result, shp.MultiPolygon):
        return [_shapely_to_polygon(p) for p in result.geoms if not p.is_empty]
    
    return []


def point_in_polygon(point: Point2D, polygon: Polygon2D) -> bool:
    """
    Check if a point is inside a polygon using winding number algorithm.
    
    Args:
        point: Point to test
        polygon: Polygon to test against
    
    Returns:
        True if point is inside polygon (or on boundary)
    """
    shp_point = shp.Point(point.x, point.y)
    shp_poly = _polygon_to_shapely(polygon)
    
    return shp_poly.contains(shp_point) or shp_poly.touches(shp_point)

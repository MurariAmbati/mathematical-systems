"""
Utility functions for numerical operations and geometric predicates.

Provides robust floating-point comparisons and tolerance-based operations.
"""

import math
from typing import Union

# Default numerical tolerance for floating-point comparisons
EPSILON: float = 1e-10


def nearly_equal(a: float, b: float, tolerance: float = EPSILON) -> bool:
    """
    Check if two floating-point numbers are nearly equal within tolerance.
    
    Args:
        a: First number
        b: Second number
        tolerance: Maximum absolute difference for equality
    
    Returns:
        True if |a - b| <= tolerance
    """
    return abs(a - b) <= tolerance


def sign(x: float, tolerance: float = EPSILON) -> int:
    """
    Return the sign of a number with tolerance.
    
    Args:
        x: Number to check
        tolerance: Values within ±tolerance are considered zero
    
    Returns:
        -1 if x < -tolerance, 0 if |x| <= tolerance, 1 if x > tolerance
    """
    if x > tolerance:
        return 1
    elif x < -tolerance:
        return -1
    else:
        return 0


def orient2d(pa: tuple[float, float], pb: tuple[float, float], pc: tuple[float, float]) -> float:
    """
    Compute 2D orientation predicate (robust).
    
    Returns twice the signed area of triangle (pa, pb, pc).
    Positive if counter-clockwise, negative if clockwise, zero if collinear.
    
    This is a simplified version; for production use, consider Shewchuk's
    adaptive precision predicates.
    
    Args:
        pa, pb, pc: Points as (x, y) tuples
    
    Returns:
        Signed area * 2
    """
    acx = pa[0] - pc[0]
    bcx = pb[0] - pc[0]
    acy = pa[1] - pc[1]
    bcy = pb[1] - pc[1]
    return acx * bcy - acy * bcx


def in_circle(pa: tuple[float, float], pb: tuple[float, float], 
              pc: tuple[float, float], pd: tuple[float, float]) -> float:
    """
    In-circle test for Delaunay triangulation.
    
    Returns positive if pd is inside the circle defined by pa, pb, pc
    (counter-clockwise), negative if outside, zero if on the circle.
    
    This is a simplified implementation; production code should use
    exact predicates.
    
    Args:
        pa, pb, pc: Points defining the circle (counter-clockwise)
        pd: Test point
    
    Returns:
        Signed value indicating position relative to circle
    """
    ax = pa[0] - pd[0]
    ay = pa[1] - pd[1]
    bx = pb[0] - pd[0]
    by = pb[1] - pd[1]
    cx = pc[0] - pd[0]
    cy = pc[1] - pd[1]
    
    ab = ax * ax + ay * ay
    bc = bx * bx + by * by
    ca = cx * cx + cy * cy
    
    return (ax * (by * ca - bc * cy) -
            ay * (bx * ca - bc * cx) +
            ab * (bx * cy - by * cx))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to range [min_val, max_val].
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between a and b.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation parameter (typically 0-1)
    
    Returns:
        Interpolated value: a + t * (b - a)
    """
    return a + t * (b - a)


def angle_between_vectors(v1: tuple[float, ...], v2: tuple[float, ...]) -> float:
    """
    Compute angle in radians between two vectors.
    
    Args:
        v1, v2: Vectors as tuples
    
    Returns:
        Angle in radians [0, π]
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(a * a for a in v2))
    
    if mag1 < EPSILON or mag2 < EPSILON:
        return 0.0
    
    cos_angle = clamp(dot / (mag1 * mag2), -1.0, 1.0)
    return math.acos(cos_angle)

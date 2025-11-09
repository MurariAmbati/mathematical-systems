"""Utility functions for tensor analysis."""

from typing import Tuple
import numpy as np


# Default numerical tolerance
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


def validate_shape(shape: Tuple[int, ...]) -> None:
    """
    Validate that a shape tuple contains only positive integers.
    
    Args:
        shape: Tuple of dimension sizes
        
    Raises:
        ValueError: If shape is invalid
    """
    if not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValueError(f"Invalid shape: {shape}. All dimensions must be positive integers.")


def shapes_compatible(shape1: Tuple[int, ...], shape2: Tuple[int, ...], axis: int) -> bool:
    """
    Check if two shapes are compatible for contraction along an axis.
    
    Args:
        shape1: First tensor shape
        shape2: Second tensor shape
        axis: Axis index to check
        
    Returns:
        True if compatible, False otherwise
    """
    if axis < 0 or axis >= len(shape1):
        return False
    if axis >= len(shape2):
        return False
    return shape1[axis] == shape2[axis]


def allclose(a: np.ndarray, b: np.ndarray, rtol: float = DEFAULT_RTOL, 
             atol: float = DEFAULT_ATOL) -> bool:
    """
    Check if two arrays are element-wise close within tolerances.
    
    Args:
        a: First array
        b: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if arrays are close
    """
    return np.allclose(a, b, rtol=rtol, atol=atol)


def normalize_axis(axis: int, ndim: int) -> int:
    """
    Normalize a possibly negative axis index to a positive one.
    
    Args:
        axis: Axis index (can be negative)
        ndim: Number of dimensions
        
    Returns:
        Normalized positive axis index
        
    Raises:
        ValueError: If axis is out of bounds
    """
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"Axis {axis} out of bounds for tensor with {ndim} dimensions")
    
    if axis < 0:
        return ndim + axis
    return axis

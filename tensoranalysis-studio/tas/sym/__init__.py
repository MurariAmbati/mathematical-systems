"""
Optional symbolic tensor computation using SymPy.

This module provides symbolic tensor algebra capabilities.
It is optional and requires sympy to be installed.
"""

from typing import Any


def _check_sympy() -> None:
    """Check if sympy is available."""
    try:
        import sympy
    except ImportError:
        raise ImportError(
            "Symbolic features require sympy. "
            "Install with: pip install tensor-analysis-studio[symbolic]"
        )


class SymbolicTensor:
    """
    Symbolic tensor with SymPy expressions as components.
    
    Wraps SymPy symbolic arrays and preserves tensor index metadata.
    
    Note:
        This is a placeholder for future symbolic implementation.
        Full integration would require:
        - Wrapping sympy.Array or sympy.MutableDenseNDimArray
        - Symbolic index contraction
        - Symbolic differentiation
        - Pretty printing with tensor notation
    """
    
    def __init__(self, *args: Any, **kwargs: Any):
        _check_sympy()
        raise NotImplementedError(
            "Symbolic tensor support is planned for Phase 1. "
            "Use numeric Tensor for now."
        )


def symbolic_christoffel(metric: Any, coords: Any) -> Any:
    """
    Compute Christoffel symbols symbolically from metric.
    
    Args:
        metric: Symbolic metric tensor
        coords: Coordinate symbols
        
    Returns:
        Symbolic Christoffel symbols
        
    Note:
        Placeholder for future symbolic implementation.
    """
    _check_sympy()
    raise NotImplementedError(
        "Symbolic Christoffel computation planned for Phase 1"
    )


def symbolic_riemann(christoffel: Any, coords: Any) -> Any:
    """
    Compute Riemann curvature tensor symbolically.
    
    Args:
        christoffel: Symbolic Christoffel symbols
        coords: Coordinate symbols
        
    Returns:
        Symbolic Riemann tensor
        
    Note:
        Placeholder for future symbolic implementation.
    """
    _check_sympy()
    raise NotImplementedError(
        "Symbolic Riemann tensor computation planned for Phase 1"
    )


__all__ = [
    "SymbolicTensor",
    "symbolic_christoffel",
    "symbolic_riemann"
]

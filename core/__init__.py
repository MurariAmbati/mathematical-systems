"""
Math Art Generator - Core Package

High-performance procedural art generation using mathematical equations.
"""

__version__ = "0.1.0"
__author__ = "Math Art Team"

from core.parser import parse_equation
from core.coordinates import (
    cartesian_grid,
    polar_to_cartesian,
    parametric_to_points,
)

__all__ = [
    "parse_equation",
    "cartesian_grid",
    "polar_to_cartesian",
    "parametric_to_points",
]

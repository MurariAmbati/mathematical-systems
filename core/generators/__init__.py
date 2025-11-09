"""
Art Generators Package

Base classes and implementations for various mathematical art generators.
"""

from core.generators.base import ArtGenerator
from core.generators.spirograph import Spirograph
from core.generators.lissajous import Lissajous
from core.generators.attractors import (
    LorenzAttractor,
    CliffordAttractor,
    IkedaAttractor,
)
from core.generators.polar_patterns import PolarPattern
from core.generators.custom_equation import CustomEquation

__all__ = [
    "ArtGenerator",
    "Spirograph",
    "Lissajous",
    "LorenzAttractor",
    "CliffordAttractor",
    "IkedaAttractor",
    "PolarPattern",
    "CustomEquation",
]

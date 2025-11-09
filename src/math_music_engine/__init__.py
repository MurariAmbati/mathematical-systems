"""
Math-Music-Engine: A modular framework for generating music from mathematical functions.
"""

__version__ = "0.1.0"

from .parser import ExpressionParser, expression_to_function
from .core import (
    FunctionEngine,
    MappingEngine,
    MappingMode,
    Scale,
    Oscillator,
    WaveformType,
    ADSR,
    Composition,
    CompositionMode,
    OutputManager,
)

__all__ = [
    # Parser
    'ExpressionParser',
    'expression_to_function',
    
    # Core
    'FunctionEngine',
    'MappingEngine',
    'MappingMode',
    'Scale',
    'Oscillator',
    'WaveformType',
    'ADSR',
    'Composition',
    'CompositionMode',
    'OutputManager',
]

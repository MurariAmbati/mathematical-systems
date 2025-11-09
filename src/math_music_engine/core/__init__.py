"""
Core modules for the Math-Music-Engine.
"""

from .function_engine import FunctionEngine
from .mapping_engine import MappingEngine, MappingMode, Scale
from .synthesis import (
    Oscillator,
    WaveformType,
    ADSR,
    AdditiveSynthesizer,
    FrequencyModulation,
    AmplitudeModulation,
    Filter
)
from .composition import Composition, Voice, CompositionMode
from .output_manager import OutputManager, create_metadata

__all__ = [
    'FunctionEngine',
    'MappingEngine',
    'MappingMode',
    'Scale',
    'Oscillator',
    'WaveformType',
    'ADSR',
    'AdditiveSynthesizer',
    'FrequencyModulation',
    'AmplitudeModulation',
    'Filter',
    'Composition',
    'Voice',
    'CompositionMode',
    'OutputManager',
    'create_metadata'
]

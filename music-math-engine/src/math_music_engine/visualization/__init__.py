"""
Visualization module for the Math-Music-Engine.
"""

from .waveform_plot import (
    plot_waveform,
    plot_stereo_waveform,
    plot_waveform_comparison
)
from .spectrogram import (
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_frequency_spectrum
)
from .surface_plot import (
    plot_function_surface,
    plot_parametric_curve_3d,
    plot_function_time_evolution,
    plot_attractor,
    plot_phase_space
)
from .mapping_viz import (
    plot_frequency_mapping,
    plot_amplitude_mapping,
    plot_scale_quantization,
    plot_harmonic_spectrum,
    plot_mapping_overview
)

__all__ = [
    # Waveform plots
    'plot_waveform',
    'plot_stereo_waveform',
    'plot_waveform_comparison',
    
    # Spectral plots
    'plot_spectrogram',
    'plot_mel_spectrogram',
    'plot_frequency_spectrum',
    
    # 3D and surface plots
    'plot_function_surface',
    'plot_parametric_curve_3d',
    'plot_function_time_evolution',
    'plot_attractor',
    'plot_phase_space',
    
    # Mapping visualization
    'plot_frequency_mapping',
    'plot_amplitude_mapping',
    'plot_scale_quantization',
    'plot_harmonic_spectrum',
    'plot_mapping_overview'
]

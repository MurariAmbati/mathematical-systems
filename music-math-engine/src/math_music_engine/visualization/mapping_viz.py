"""
Mapping visualization for frequency and parameter mappings.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def plot_frequency_mapping(
    input_values: np.ndarray,
    frequencies: np.ndarray,
    title: str = "Frequency Mapping",
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize mapping from input values to frequencies.
    
    Args:
        input_values: Input parameter values
        frequencies: Mapped frequencies in Hz
        title: Plot title
        figsize: Figure size
        log_scale: Use logarithmic frequency scale
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(input_values, frequencies, linewidth=2)
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Frequency mapping plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_amplitude_mapping(
    input_values: np.ndarray,
    amplitudes: np.ndarray,
    title: str = "Amplitude Mapping",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize mapping from input values to amplitudes.
    
    Args:
        input_values: Input parameter values
        amplitudes: Mapped amplitudes
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(input_values, amplitudes, linewidth=2)
    ax.fill_between(input_values, 0, amplitudes, alpha=0.3)
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Amplitude mapping plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_scale_quantization(
    continuous_notes: np.ndarray,
    quantized_notes: np.ndarray,
    time: np.ndarray,
    scale_name: str = "Scale",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize scale quantization effect.
    
    Args:
        continuous_notes: Continuous note values before quantization
        quantized_notes: Quantized MIDI notes
        time: Time array
        scale_name: Name of the scale
        title: Plot title (default: auto-generated)
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    if title is None:
        title = f"Scale Quantization ({scale_name})"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot continuous values
    ax.plot(time, continuous_notes, 'b-', alpha=0.5, label='Continuous', linewidth=1)
    
    # Plot quantized values
    ax.plot(time, quantized_notes, 'r-', label='Quantized', linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MIDI Note Number')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Quantization plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_harmonic_spectrum(
    harmonics: List[float],
    fundamental_freq: float = 440.0,
    title: str = "Harmonic Spectrum",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize harmonic spectrum.
    
    Args:
        harmonics: List of harmonic amplitudes
        fundamental_freq: Fundamental frequency in Hz
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate harmonic frequencies
    harmonic_nums = np.arange(1, len(harmonics) + 1)
    frequencies = fundamental_freq * harmonic_nums
    
    # Plot as stem plot
    markerline, stemlines, baseline = ax.stem(
        frequencies, harmonics,
        basefmt=' ',
        use_line_collection=True
    )
    markerline.set_markerfacecolor('blue')
    markerline.set_markeredgecolor('blue')
    stemlines.set_color('blue')
    stemlines.set_linewidth(2)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate harmonic numbers
    for i, (freq, amp) in enumerate(zip(frequencies, harmonics)):
        if amp > 0.1:  # Only annotate significant harmonics
            ax.text(freq, amp + 0.05, f'H{i+1}', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Harmonic spectrum saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_mapping_overview(
    metadata: Dict[str, Any],
    input_signal: np.ndarray,
    output_params: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comprehensive overview of mapping process.
    
    Args:
        metadata: Mapping metadata
        input_signal: Input mathematical signal
        output_params: Dictionary of output parameters
                      (e.g., {'frequency': [...], 'amplitude': [...]})
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    n_params = len(output_params) + 1
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    
    time = np.arange(len(input_signal)) / metadata.get('sample_rate', 44100)
    
    # Plot input signal
    axes[0].plot(time, input_signal, linewidth=1)
    axes[0].set_ylabel('Input')
    axes[0].set_title('Mapping Overview')
    axes[0].grid(True, alpha=0.3)
    
    # Plot each output parameter
    for i, (param_name, param_values) in enumerate(output_params.items(), start=1):
        axes[i].plot(time, param_values, linewidth=1, color=f'C{i}')
        axes[i].set_ylabel(param_name.capitalize())
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Mapping overview saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig

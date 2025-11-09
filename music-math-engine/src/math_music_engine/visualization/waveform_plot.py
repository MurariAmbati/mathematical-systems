"""
Waveform plotting for time-domain visualization.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_waveform(
    signal: np.ndarray,
    sample_rate: int = 44100,
    duration: Optional[float] = None,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 4),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot time-domain waveform.
    
    Args:
        signal: Audio signal array
        sample_rate: Sample rate in Hz
        duration: Duration to plot (None = full signal)
        title: Plot title
        figsize: Figure size (width, height)
        output_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Handle stereo
    if signal.ndim == 2:
        return plot_stereo_waveform(
            signal, sample_rate, duration, title, figsize, output_path, show
        )
    
    # Create time array
    num_samples = len(signal)
    t = np.arange(num_samples) / sample_rate
    
    # Limit duration if specified
    if duration is not None:
        num_samples_to_plot = int(duration * sample_rate)
        if num_samples_to_plot < num_samples:
            signal = signal[:num_samples_to_plot]
            t = t[:num_samples_to_plot]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, signal, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Waveform plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_stereo_waveform(
    signal: np.ndarray,
    sample_rate: int = 44100,
    duration: Optional[float] = None,
    title: str = "Stereo Waveform",
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot stereo waveform with separate channels.
    
    Args:
        signal: Stereo audio signal (shape: [samples, 2])
        sample_rate: Sample rate in Hz
        duration: Duration to plot (None = full signal)
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create time array
    num_samples = len(signal)
    t = np.arange(num_samples) / sample_rate
    
    # Limit duration if specified
    if duration is not None:
        num_samples_to_plot = int(duration * sample_rate)
        if num_samples_to_plot < num_samples:
            signal = signal[:num_samples_to_plot]
            t = t[:num_samples_to_plot]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Left channel
    ax1.plot(t, signal[:, 0], linewidth=0.5, color='blue')
    ax1.set_ylabel('Left')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # Right channel
    ax2.plot(t, signal[:, 1], linewidth=0.5, color='red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Stereo waveform plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_waveform_comparison(
    signals: list[np.ndarray],
    labels: list[str],
    sample_rate: int = 44100,
    duration: Optional[float] = None,
    title: str = "Waveform Comparison",
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot multiple waveforms for comparison.
    
    Args:
        signals: List of audio signals
        labels: List of labels for each signal
        sample_rate: Sample rate in Hz
        duration: Duration to plot
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    num_signals = len(signals)
    fig, axes = plt.subplots(num_signals, 1, figsize=figsize, sharex=True)
    
    if num_signals == 1:
        axes = [axes]
    
    for i, (signal, label) in enumerate(zip(signals, labels)):
        # Handle stereo (use left channel)
        if signal.ndim == 2:
            signal = signal[:, 0]
        
        # Create time array
        t = np.arange(len(signal)) / sample_rate
        
        # Limit duration
        if duration is not None:
            num_samples_to_plot = int(duration * sample_rate)
            if num_samples_to_plot < len(signal):
                signal = signal[:num_samples_to_plot]
                t = t[:num_samples_to_plot]
        
        # Plot
        axes[i].plot(t, signal, linewidth=0.5)
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-1.1, 1.1)
    
    axes[0].set_title(title)
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig

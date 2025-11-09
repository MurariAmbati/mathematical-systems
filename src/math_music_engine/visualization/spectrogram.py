"""
Spectrogram visualization for frequency-domain analysis.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal


def plot_spectrogram(
    audio_signal: np.ndarray,
    sample_rate: int = 44100,
    window_size: int = 2048,
    hop_size: int = 512,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'viridis',
    vmin: float = -80,
    vmax: float = 0,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot spectrogram of audio signal.
    
    Args:
        audio_signal: Audio signal array
        sample_rate: Sample rate in Hz
        window_size: FFT window size
        hop_size: Hop size between windows
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for color scale (dB)
        vmax: Maximum value for color scale (dB)
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Handle stereo (use left channel)
    if audio_signal.ndim == 2:
        audio_signal = audio_signal[:, 0]
    
    # Compute spectrogram
    f, t, Sxx = sp_signal.spectrogram(
        audio_signal,
        fs=sample_rate,
        window='hann',
        nperseg=window_size,
        noverlap=window_size - hop_size,
        scaling='density'
    )
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectrogram
    im = ax.pcolormesh(
        t, f, Sxx_db,
        shading='gouraud',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim(0, sample_rate / 2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_mel_spectrogram(
    audio_signal: np.ndarray,
    sample_rate: int = 44100,
    n_mels: int = 128,
    window_size: int = 2048,
    hop_size: int = 512,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'magma',
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot mel-scaled spectrogram.
    
    Args:
        audio_signal: Audio signal array
        sample_rate: Sample rate in Hz
        n_mels: Number of mel bands
        window_size: FFT window size
        hop_size: Hop size between windows
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Handle stereo
    if audio_signal.ndim == 2:
        audio_signal = audio_signal[:, 0]
    
    # Compute regular spectrogram first
    f, t, Sxx = sp_signal.spectrogram(
        audio_signal,
        fs=sample_rate,
        window='hann',
        nperseg=window_size,
        noverlap=window_size - hop_size
    )
    
    # Create mel filter bank
    mel_filterbank = create_mel_filterbank(n_mels, len(f), sample_rate)
    
    # Apply mel filters
    mel_spec = np.dot(mel_filterbank, Sxx)
    
    # Convert to dB
    mel_spec_db = 10 * np.log10(mel_spec + 1e-10)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.pcolormesh(
        t, np.arange(n_mels), mel_spec_db,
        shading='gouraud',
        cmap=cmap
    )
    
    ax.set_ylabel('Mel Band')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Mel spectrogram saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def create_mel_filterbank(
    n_mels: int,
    n_fft_bins: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Create mel-scale filter bank.
    
    Args:
        n_mels: Number of mel bands
        n_fft_bins: Number of FFT bins
        sample_rate: Sample rate in Hz
        fmin: Minimum frequency
        fmax: Maximum frequency (default: Nyquist)
        
    Returns:
        Mel filter bank matrix (n_mels, n_fft_bins)
    """
    if fmax is None:
        fmax = sample_rate / 2
    
    # Helper functions for mel scale
    def hz_to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)
    
    # Create mel-spaced frequencies
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Convert to FFT bin numbers
    bin_points = np.floor((n_fft_bins + 1) * hz_points / sample_rate).astype(int)
    
    # Create filter bank
    filterbank = np.zeros((n_mels, n_fft_bins))
    
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        # Rising slope
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        
        # Falling slope
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)
    
    return filterbank


def plot_frequency_spectrum(
    audio_signal: np.ndarray,
    sample_rate: int = 44100,
    title: str = "Frequency Spectrum",
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = True,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot frequency spectrum (FFT).
    
    Args:
        audio_signal: Audio signal array
        sample_rate: Sample rate in Hz
        title: Plot title
        figsize: Figure size
        log_scale: Use logarithmic frequency scale
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Handle stereo
    if audio_signal.ndim == 2:
        audio_signal = audio_signal[:, 0]
    
    # Compute FFT
    fft = np.fft.rfft(audio_signal)
    magnitude = np.abs(fft)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Frequency bins
    freqs = np.fft.rfftfreq(len(audio_signal), 1/sample_rate)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(freqs, magnitude_db, linewidth=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlim(20, sample_rate / 2)
    else:
        ax.set_xlim(0, sample_rate / 2)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Frequency spectrum saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig

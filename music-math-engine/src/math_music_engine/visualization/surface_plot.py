"""
Surface and 3D plotting for mathematical functions.
"""

from typing import Optional, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_function_surface(
    func: Callable,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    title: str = "Function Surface",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 3D surface of a mathematical function f(x, y).
    
    Args:
        func: Function taking (x, y) and returning z
        x_range: (min, max) for x axis
        y_range: (min, max) for y axis
        resolution: Number of points per axis
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function
    Z = func(X, Y)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Surface plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_parametric_curve_3d(
    x_func: Callable,
    y_func: Callable,
    z_func: Callable,
    t_range: Tuple[float, float] = (0, 10),
    num_points: int = 1000,
    title: str = "Parametric Curve",
    figsize: Tuple[int, int] = (10, 8),
    color_by_t: bool = True,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 3D parametric curve.
    
    Args:
        x_func: Function x(t)
        y_func: Function y(t)
        z_func: Function z(t)
        t_range: (min, max) for parameter t
        num_points: Number of points to plot
        title: Plot title
        figsize: Figure size
        color_by_t: Color the curve by parameter value
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Generate parameter values
    t = np.linspace(t_range[0], t_range[1], num_points)
    
    # Evaluate functions
    x = x_func(t)
    y = y_func(t)
    z = z_func(t)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if color_by_t:
        # Color by parameter value
        ax.scatter(x, y, z, c=t, cmap='viridis', s=1)
    else:
        ax.plot(x, y, z, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Parametric curve saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_function_time_evolution(
    func: Callable,
    t_range: Tuple[float, float] = (0, 10),
    x_range: Tuple[float, float] = (-5, 5),
    num_time_steps: int = 50,
    num_x_points: int = 200,
    title: str = "Function Evolution",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'plasma',
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot evolution of function f(x, t) over time.
    
    Args:
        func: Function taking (x, t) and returning value
        t_range: (min, max) time range
        x_range: (min, max) spatial range
        num_time_steps: Number of time steps
        num_x_points: Number of spatial points
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create grids
    t = np.linspace(t_range[0], t_range[1], num_time_steps)
    x = np.linspace(x_range[0], x_range[1], num_x_points)
    T, X = np.meshgrid(t, x)
    
    # Evaluate function
    Z = func(X, T)
    
    # Create 2D plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.pcolormesh(T, X, Z, cmap=cmap, shading='gouraud')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('X')
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Value')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Evolution plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_attractor(
    trajectory: np.ndarray,
    title: str = "Attractor",
    figsize: Tuple[int, int] = (10, 8),
    color_by_time: bool = True,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 3D attractor from trajectory data.
    
    Args:
        trajectory: Array of shape (n_points, 3) with [x, y, z] coordinates
        title: Plot title
        figsize: Figure size
        color_by_time: Color the trajectory by time
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    if trajectory.shape[1] != 3:
        raise ValueError("Trajectory must have 3 dimensions")
    
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if color_by_time:
        # Color by time progression
        time_colors = np.arange(len(x))
        scatter = ax.scatter(x, y, z, c=time_colors, cmap='viridis', s=1, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Time')
    else:
        ax.plot(x, y, z, linewidth=0.5, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Attractor plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_phase_space(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Phase Space",
    figsize: Tuple[int, int] = (8, 8),
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 2D phase space diagram.
    
    Args:
        x: First coordinate array
        y: Second coordinate array
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trajectory
    ax.plot(x, y, linewidth=0.5, alpha=0.7)
    
    # Mark start and end
    ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Phase space plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig

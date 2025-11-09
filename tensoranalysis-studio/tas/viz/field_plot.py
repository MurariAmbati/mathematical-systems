"""
Field visualization for tensor fields on grids.

Supports visualizing:
- Vector fields (arrows, streamlines)
- Symmetric rank-2 tensors (ellipses/glyphs)
- Scalar fields (heatmaps)
"""

from typing import Optional, Tuple, Callable
import numpy as np

from tas.core.tensor import Tensor


def plot_vector_field_2d(
    vector_field: Callable[[np.ndarray], Tensor],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    grid_points: int = 20,
    style: str = "arrows",
    title: Optional[str] = None
) -> None:
    """
    Plot a 2D vector field.
    
    Args:
        vector_field: Function that takes (x, y) and returns 2D vector Tensor
        x_range: Range of x coordinates
        y_range: Range of y coordinates
        grid_points: Number of grid points in each direction
        style: 'arrows' or 'streamlines'
        title: Optional plot title
        
    Note:
        Requires matplotlib and optionally plotly for interactive plots.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Field plotting requires matplotlib. "
            "Install with: pip install tensor-analysis-studio[viz]"
        )
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], grid_points)
    y = np.linspace(y_range[0], y_range[1], grid_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate vector field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(grid_points):
        for j in range(grid_points):
            point = np.array([X[i, j], Y[i, j]])
            vec = vector_field(point)
            U[i, j] = vec.data[0]
            V[i, j] = vec.data[1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if style == "arrows":
        ax.quiver(X, Y, U, V, alpha=0.8)
    elif style == "streamlines":
        ax.streamplot(X, Y, U, V, density=1.5, linewidth=1, arrowsize=1.5)
    else:
        raise ValueError(f"Unknown style: {style}")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Vector Field")
    
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_tensor_field_2d(
    tensor_field: Callable[[np.ndarray], Tensor],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    grid_points: int = 15,
    title: Optional[str] = None
) -> None:
    """
    Plot a 2D rank-2 symmetric tensor field using ellipse glyphs.
    
    Each tensor is visualized as an ellipse whose axes are the eigenvectors
    and whose radii are proportional to the eigenvalues.
    
    Args:
        tensor_field: Function that takes (x, y) and returns 2x2 tensor
        x_range: Range of x coordinates
        y_range: Range of y coordinates
        grid_points: Number of grid points in each direction
        title: Optional plot title
        
    Note:
        Requires matplotlib for plotting.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError:
        raise ImportError(
            "Tensor field plotting requires matplotlib. "
            "Install with: pip install tensor-analysis-studio[viz]"
        )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], grid_points)
    y = np.linspace(y_range[0], y_range[1], grid_points)
    
    for xi in x:
        for yi in y:
            point = np.array([xi, yi])
            T = tensor_field(point)
            
            if T.ndim != 2 or T.shape != (2, 2):
                raise ValueError("Tensor field must return 2x2 tensors")
            
            # Compute eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(T.data)
            
            # Ellipse parameters
            width = abs(eigvals[0]) * 0.3
            height = abs(eigvals[1]) * 0.3
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            
            # Determine color based on eigenvalues
            color = "blue" if eigvals[0] > 0 and eigvals[1] > 0 else "red"
            
            # Draw ellipse
            ellipse = Ellipse(
                (xi, yi), width, height, angle=angle,
                edgecolor=color, facecolor=color, alpha=0.3, linewidth=1
            )
            ax.add_patch(ellipse)
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Rank-2 Tensor Field (Ellipse Glyphs)")
    
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_scalar_field_2d(
    scalar_field: Callable[[np.ndarray], float],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    grid_points: int = 100,
    title: Optional[str] = None,
    colormap: str = "viridis"
) -> None:
    """
    Plot a 2D scalar field as a heatmap.
    
    Args:
        scalar_field: Function that takes (x, y) and returns scalar
        x_range: Range of x coordinates
        y_range: Range of y coordinates
        grid_points: Number of grid points in each direction
        title: Optional plot title
        colormap: Matplotlib colormap name
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Scalar field plotting requires matplotlib. "
            "Install with: pip install tensor-analysis-studio[viz]"
        )
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], grid_points)
    y = np.linspace(y_range[0], y_range[1], grid_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate scalar field
    Z = np.zeros_like(X)
    for i in range(grid_points):
        for j in range(grid_points):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = scalar_field(point)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(X, Y, Z, levels=50, cmap=colormap)
    ax.contour(X, Y, Z, levels=10, colors="black", alpha=0.3, linewidths=0.5)
    
    plt.colorbar(im, ax=ax, label="Value")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Scalar Field")
    
    plt.show()

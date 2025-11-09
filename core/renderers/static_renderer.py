"""
Static Image Renderer

Renders mathematical art to static images (PNG, JPG).
"""

from typing import Optional, Union, Tuple, Literal
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.cm as cm
from core.color_maps import ColorMap, FunctionColorMapper, get_palette
from core.coordinates import normalize_points


class StaticRenderer:
    """
    Render mathematical art as static images.
    
    Supports 2D and 3D point clouds with various rendering styles.
    """
    
    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        dpi: int = 100,
        colormap: Union[str, ColorMap] = "viridis",
        background: str = "black",
        style: str = "scatter",
    ):
        """
        Initialize static renderer.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            dpi: Dots per inch
            colormap: Color mapping (name or ColorMap object)
            background: Background color
            style: Rendering style ("scatter", "line", "density")
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.background = background
        self.style = style
        
        # Setup colormap
        if isinstance(colormap, str):
            self.colormap = get_palette(colormap)
        else:
            self.colormap = colormap
        
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self._setup_figure()
    
    def _setup_figure(self):
        """Setup matplotlib figure and axes."""
        fig_width = self.width / self.dpi
        fig_height = self.height / self.dpi
        
        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)
        
        # Set background
        self.fig.patch.set_facecolor(self.background)
        self.ax.set_facecolor(self.background)
        
        # Remove axes
        self.ax.axis('off')
        
        # Set aspect ratio
        self.ax.set_aspect('equal')
    
    def render(
        self,
        points: NDArray,
        colors: Optional[NDArray] = None,
        sizes: Optional[Union[float, NDArray]] = None,
        alpha: float = 0.6,
        **kwargs
    ):
        """
        Render points to the figure.
        
        Args:
            points: Array of shape (n, 2) or (n, 3) with point coordinates
            colors: Optional color array (RGBA or color values)
            sizes: Point sizes (scalar or array)
            alpha: Global alpha/opacity
            **kwargs: Additional rendering parameters
        """
        if points.shape[1] == 2:
            self._render_2d(points, colors, sizes, alpha, **kwargs)
        elif points.shape[1] == 3:
            self._render_3d(points, colors, sizes, alpha, **kwargs)
        else:
            raise ValueError(f"Points must have 2 or 3 dimensions, got {points.shape[1]}")
    
    def _render_2d(
        self,
        points: NDArray,
        colors: Optional[NDArray],
        sizes: Optional[Union[float, NDArray]],
        alpha: float,
        **kwargs
    ):
        """Render 2D points."""
        x, y = points[:, 0], points[:, 1]
        
        # Determine colors
        if colors is None:
            # Map by position or index
            color_values = np.linspace(0, 1, len(x))
            colors = self.colormap.map(color_values, alpha=alpha)
        elif colors.ndim == 1:
            # Scalar values - map to colors
            colors = self.colormap.map(colors, alpha=alpha)
        
        # Determine sizes
        if sizes is None:
            sizes = 1.0
        
        # Render based on style
        if self.style == "scatter":
            self.ax.scatter(x, y, c=colors, s=sizes, **kwargs)
        elif self.style == "line":
            if colors.ndim == 2:
                # Use first color
                color = colors[0]
            else:
                color = 'white'
            self.ax.plot(x, y, c=color, linewidth=sizes if np.isscalar(sizes) else 1.0, alpha=alpha, **kwargs)
        elif self.style == "density":
            # Hexbin for density plot
            self.ax.hexbin(x, y, gridsize=100, cmap=self.colormap._cmap, mincnt=1, **kwargs)
        else:
            raise ValueError(f"Unknown style: {self.style}")
        
        # Auto-scale
        margin = 0.1
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        self.ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
        self.ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    
    def _render_3d(
        self,
        points: NDArray,
        colors: Optional[NDArray],
        sizes: Optional[Union[float, NDArray]],
        alpha: float,
        **kwargs
    ):
        """Render 3D points (projected to 2D)."""
        # For now, use simple orthographic projection
        # Project onto x-y plane
        x, y = points[:, 0], points[:, 1]
        z = points[:, 2]
        
        # Color by z-coordinate if no colors provided
        if colors is None:
            colors = self.colormap.map(z, alpha=alpha)
        elif colors.ndim == 1:
            colors = self.colormap.map(colors, alpha=alpha)
        
        # Determine sizes
        if sizes is None:
            sizes = 1.0
        
        # Render as 2D
        self.ax.scatter(x, y, c=colors, s=sizes, **kwargs)
        
        # Auto-scale
        margin = 0.1
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        self.ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
        self.ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    
    def render_with_function_colors(
        self,
        points: NDArray,
        mode: Literal["value", "magnitude", "gradient", "curvature", "velocity"] = "value",
        alpha: float = 0.6,
        **kwargs
    ):
        """
        Render with colors based on function properties.
        
        Args:
            points: Array of shape (n, 2) or (n, 3)
            mode: Coloring mode
            alpha: Opacity
            **kwargs: Additional rendering parameters
        """
        color_mapper = FunctionColorMapper(self.colormap)
        
        if mode == "value" and points.shape[1] == 3:
            # Use third column as value
            colors = color_mapper.map_by_value(points[:, 2], alpha=alpha)
            self.render(points[:, :2], colors=colors, alpha=alpha, **kwargs)
        elif mode == "magnitude" and points.shape[1] == 3:
            colors = color_mapper.map_by_magnitude(points[:, 2], alpha=alpha)
            self.render(points[:, :2], colors=colors, alpha=alpha, **kwargs)
        elif mode == "curvature":
            colors = color_mapper.map_by_curvature(points[:, :2], alpha=alpha)
            self.render(points, colors=colors, alpha=alpha, **kwargs)
        elif mode == "velocity":
            colors = color_mapper.map_by_velocity(points, alpha=alpha)
            self.render(points, colors=colors, alpha=alpha, **kwargs)
        else:
            # Default to regular rendering
            self.render(points, alpha=alpha, **kwargs)
    
    def clear(self):
        """Clear the current figure."""
        self.ax.clear()
        self.ax.set_facecolor(self.background)
        self.ax.axis('off')
        self.ax.set_aspect('equal')
    
    def save(
        self,
        filename: str,
        dpi: Optional[int] = None,
        transparent: bool = False,
        **kwargs
    ):
        """
        Save the rendered image.
        
        Args:
            filename: Output filename (PNG, JPG, etc.)
            dpi: Override DPI for output
            transparent: Transparent background
            **kwargs: Additional savefig parameters
        """
        if dpi is None:
            dpi = self.dpi
        
        self.fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor=self.fig.get_facecolor() if not transparent else 'none',
            transparent=transparent,
            **kwargs
        )
        
        print(f"Saved image to: {filename}")
    
    def show(self):
        """Display the rendered image."""
        plt.show()
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MultiPanelRenderer:
    """
    Render multiple art pieces in a grid layout.
    """
    
    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        width: int = 2048,
        height: int = 2048,
        dpi: int = 100,
        background: str = "black",
    ):
        """
        Initialize multi-panel renderer.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            width: Total width
            height: Total height
            dpi: DPI
            background: Background color
        """
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.dpi = dpi
        self.background = background
        
        fig_width = width / dpi
        fig_height = height / dpi
        
        self.fig, self.axes = plt.subplots(
            rows, cols,
            figsize=(fig_width, fig_height),
            dpi=dpi
        )
        
        self.fig.patch.set_facecolor(background)
        
        # Flatten axes array for easy indexing
        if rows == 1 and cols == 1:
            self.axes = np.array([self.axes])
        else:
            self.axes = self.axes.ravel()
        
        # Setup each axis
        for ax in self.axes:
            ax.set_facecolor(background)
            ax.axis('off')
            ax.set_aspect('equal')
        
        self.current_panel = 0
    
    def render_panel(
        self,
        panel_index: int,
        points: NDArray,
        colormap: Union[str, ColorMap] = "viridis",
        **kwargs
    ):
        """
        Render points to a specific panel.
        
        Args:
            panel_index: Panel index (0 to rows*cols-1)
            points: Points to render
            colormap: Colormap to use
            **kwargs: Additional rendering parameters
        """
        ax = self.axes[panel_index]
        
        if isinstance(colormap, str):
            cmap = get_palette(colormap)
        else:
            cmap = colormap
        
        x, y = points[:, 0], points[:, 1]
        
        # Simple scatter plot
        colors = cmap.map(np.linspace(0, 1, len(x)))
        ax.scatter(x, y, c=colors, s=1, **kwargs)
        
        # Auto-scale
        margin = 0.1
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
        ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    
    def save(self, filename: str, **kwargs):
        """Save the multi-panel figure."""
        self.fig.savefig(
            filename,
            dpi=self.dpi,
            bbox_inches='tight',
            facecolor=self.fig.get_facecolor(),
            **kwargs
        )
        print(f"Saved multi-panel image to: {filename}")
    
    def close(self):
        """Close the figure."""
        plt.close(self.fig)

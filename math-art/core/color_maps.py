"""
Color Mapping System

Gradient and color mapping utilities for mathematical art.
"""

from typing import Optional, Tuple, Union, List, Callable
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap


class ColorMap:
    """Base class for color mapping strategies."""
    
    def __init__(self, name: str = "viridis"):
        """
        Initialize color map.
        
        Args:
            name: Name of matplotlib colormap or custom palette
        """
        self.name = name
        self._cmap = None
        self._load_colormap()
    
    def _load_colormap(self):
        """Load the colormap from matplotlib."""
        try:
            self._cmap = get_cmap(self.name)
        except ValueError:
            # Default to viridis if not found
            self._cmap = get_cmap("viridis")
    
    def map(
        self,
        values: NDArray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: float = 1.0
    ) -> NDArray:
        """
        Map values to colors.
        
        Args:
            values: Array of values to map
            vmin: Minimum value for mapping (auto if None)
            vmax: Maximum value for mapping (auto if None)
            alpha: Alpha/opacity value
            
        Returns:
            RGBA array of shape (*values.shape, 4)
        """
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        
        # Normalize to [0, 1]
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(values)
        
        # Clip to valid range
        normalized = np.clip(normalized, 0, 1)
        
        # Apply colormap
        colors = self._cmap(normalized)
        
        # Set alpha
        if alpha != 1.0:
            colors[..., 3] = alpha
        
        return colors
    
    def get_colors(self, n_colors: int) -> NDArray:
        """
        Get n discrete colors from the colormap.
        
        Args:
            n_colors: Number of colors to sample
            
        Returns:
            Array of shape (n_colors, 4) with RGBA values
        """
        values = np.linspace(0, 1, n_colors)
        return self._cmap(values)


class GradientColorMap(ColorMap):
    """Custom gradient colormap from list of colors."""
    
    def __init__(self, colors: List[str], name: str = "custom_gradient"):
        """
        Initialize gradient from list of colors.
        
        Args:
            colors: List of color strings (hex or named colors)
            name: Name for this colormap
        """
        self.colors = colors
        self.name = name
        self._create_gradient()
    
    def _create_gradient(self):
        """Create linear gradient from color list."""
        n_colors = len(self.colors)
        positions = np.linspace(0, 1, n_colors)
        
        # Convert colors to RGB
        rgb_colors = [mcolors.to_rgb(c) for c in self.colors]
        
        # Create color map
        cdict = {
            'red': [],
            'green': [],
            'blue': []
        }
        
        for pos, rgb in zip(positions, rgb_colors):
            cdict['red'].append((pos, rgb[0], rgb[0]))
            cdict['green'].append((pos, rgb[1], rgb[1]))
            cdict['blue'].append((pos, rgb[2], rgb[2]))
        
        self._cmap = mcolors.LinearSegmentedColormap(self.name, cdict)


class CyclicColorMap(ColorMap):
    """Colormap that cycles smoothly (for periodic data)."""
    
    def __init__(self, name: str = "twilight"):
        """
        Initialize cyclic colormap.
        
        Args:
            name: Name of cyclic colormap (twilight, twilight_shifted, hsv)
        """
        super().__init__(name)


class FunctionColorMapper:
    """Map colors based on function properties (value, derivative, curvature)."""
    
    def __init__(self, colormap: Union[str, ColorMap] = "viridis"):
        """
        Initialize function-based color mapper.
        
        Args:
            colormap: Colormap to use
        """
        if isinstance(colormap, str):
            self.colormap = ColorMap(colormap)
        else:
            self.colormap = colormap
    
    def map_by_value(
        self,
        values: NDArray,
        alpha: float = 1.0
    ) -> NDArray:
        """
        Map colors directly from function values.
        
        Args:
            values: Function values
            alpha: Opacity
            
        Returns:
            RGBA colors
        """
        return self.colormap.map(values, alpha=alpha)
    
    def map_by_magnitude(
        self,
        values: NDArray,
        alpha: float = 1.0
    ) -> NDArray:
        """
        Map colors from absolute magnitude of values.
        
        Args:
            values: Function values
            alpha: Opacity
            
        Returns:
            RGBA colors
        """
        magnitude = np.abs(values)
        return self.colormap.map(magnitude, alpha=alpha)
    
    def map_by_gradient(
        self,
        values: NDArray,
        alpha: float = 1.0
    ) -> NDArray:
        """
        Map colors based on gradient magnitude.
        
        Args:
            values: Function values (2D array)
            alpha: Opacity
            
        Returns:
            RGBA colors
        """
        # Compute gradient magnitude
        gy, gx = np.gradient(values)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        return self.colormap.map(gradient_mag, alpha=alpha)
    
    def map_by_curvature(
        self,
        points: NDArray,
        alpha: float = 1.0
    ) -> NDArray:
        """
        Map colors based on curvature of curve.
        
        Args:
            points: Curve points of shape (n, 2)
            alpha: Opacity
            
        Returns:
            RGBA colors
        """
        from core.utils import compute_curvature_2d
        curvature = compute_curvature_2d(points)
        return self.colormap.map(curvature, alpha=alpha)
    
    def map_by_velocity(
        self,
        points: NDArray,
        alpha: float = 1.0
    ) -> NDArray:
        """
        Map colors based on velocity along curve.
        
        Args:
            points: Curve points of shape (n, d)
            alpha: Opacity
            
        Returns:
            RGBA colors
        """
        from core.utils import compute_velocity
        velocity = compute_velocity(points)
        return self.colormap.map(velocity, alpha=alpha)


class DiscreteColorPalette:
    """Discrete color palette for categorical or indexed coloring."""
    
    def __init__(self, colors: Optional[List[str]] = None):
        """
        Initialize discrete palette.
        
        Args:
            colors: List of colors (uses default palette if None)
        """
        if colors is None:
            # Default palette (inspired by Tableau)
            self.colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf'
            ]
        else:
            self.colors = colors
        
        # Convert to RGB arrays
        self.rgb_colors = np.array([mcolors.to_rgba(c) for c in self.colors])
    
    def get_color(self, index: int) -> NDArray:
        """Get color by index (cycles if index exceeds palette size)."""
        return self.rgb_colors[index % len(self.rgb_colors)]
    
    def map_indices(self, indices: NDArray) -> NDArray:
        """
        Map array of indices to colors.
        
        Args:
            indices: Integer array
            
        Returns:
            RGBA colors array
        """
        indices_mod = indices % len(self.rgb_colors)
        return self.rgb_colors[indices_mod]


# Predefined color palettes
PALETTES = {
    "default": ColorMap("viridis"),
    "heat": ColorMap("hot"),
    "cool": ColorMap("cool"),
    "rainbow": ColorMap("rainbow"),
    "plasma": ColorMap("plasma"),
    "magma": ColorMap("magma"),
    "inferno": ColorMap("inferno"),
    "cividis": ColorMap("cividis"),
    "twilight": CyclicColorMap("twilight"),
    "ocean": GradientColorMap(['#000033', '#0077be', '#00c9ff', '#ffffff'], "ocean"),
    "fire": GradientColorMap(['#000000', '#8B0000', '#FF4500', '#FFD700', '#FFFFFF'], "fire"),
    "forest": GradientColorMap(['#004d00', '#228B22', '#90EE90', '#FFFF00'], "forest"),
    "sunset": GradientColorMap(['#4A148C', '#E91E63', '#FF5722', '#FFC107'], "sunset"),
    "northern_lights": GradientColorMap(['#001F3F', '#00A86B', '#7FDBFF', '#39FF14'], "northern_lights"),
}


def get_palette(name: str) -> ColorMap:
    """
    Get predefined color palette by name.
    
    Args:
        name: Palette name
        
    Returns:
        ColorMap instance
        
    Raises:
        ValueError: If palette not found
    """
    if name in PALETTES:
        return PALETTES[name]
    else:
        # Try as matplotlib colormap
        try:
            return ColorMap(name)
        except:
            raise ValueError(
                f"Unknown palette: {name}. "
                f"Available: {list(PALETTES.keys())}"
            )


def blend_colors(
    color1: Union[str, Tuple],
    color2: Union[str, Tuple],
    weight: float
) -> NDArray:
    """
    Blend two colors.
    
    Args:
        color1: First color
        color2: Second color
        weight: Blend weight (0=all color1, 1=all color2)
        
    Returns:
        RGBA array
    """
    c1 = np.array(mcolors.to_rgba(color1))
    c2 = np.array(mcolors.to_rgba(color2))
    return (1 - weight) * c1 + weight * c2


def create_custom_colormap(
    colors: List[str],
    positions: Optional[List[float]] = None,
    name: str = "custom"
) -> ColorMap:
    """
    Create custom colormap from colors and positions.
    
    Args:
        colors: List of colors
        positions: List of positions in [0, 1] (evenly spaced if None)
        name: Name for colormap
        
    Returns:
        ColorMap instance
    """
    if positions is None:
        positions = np.linspace(0, 1, len(colors))
    
    # Ensure positions are sorted and normalized
    positions = np.array(positions)
    positions = (positions - positions.min()) / (positions.max() - positions.min())
    
    # Convert colors to RGB
    rgb_colors = [mcolors.to_rgb(c) for c in colors]
    
    # Create colormap dictionary
    cdict = {'red': [], 'green': [], 'blue': []}
    
    for pos, rgb in zip(positions, rgb_colors):
        cdict['red'].append((pos, rgb[0], rgb[0]))
        cdict['green'].append((pos, rgb[1], rgb[1]))
        cdict['blue'].append((pos, rgb[2], rgb[2]))
    
    cmap_obj = mcolors.LinearSegmentedColormap(name, cdict)
    
    colormap = ColorMap.__new__(ColorMap)
    colormap.name = name
    colormap._cmap = cmap_obj
    return colormap

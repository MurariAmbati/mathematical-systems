"""
Vector Renderer

Renders mathematical art as vector graphics (SVG).
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from core.color_maps import ColorMap, get_palette
import matplotlib.colors as mcolors


class VectorRenderer:
    """
    Render mathematical art as vector graphics (SVG).
    
    Creates scalable vector graphics from point data.
    """
    
    def __init__(
        self,
        width: float = 1000,
        height: float = 1000,
        viewbox: Optional[Tuple[float, float, float, float]] = None,
        background: str = "black",
        colormap: Union[str, ColorMap] = "viridis",
    ):
        """
        Initialize vector renderer.
        
        Args:
            width: SVG width
            height: SVG height
            viewbox: Custom viewbox (x, y, width, height) or None for auto
            background: Background color
            colormap: Color mapping
        """
        self.width = width
        self.height = height
        self.viewbox = viewbox
        self.background = background
        
        if isinstance(colormap, str):
            self.colormap = get_palette(colormap)
        else:
            self.colormap = colormap
        
        self.elements: List[str] = []
    
    def _normalize_points(
        self,
        points: NDArray,
        margin: float = 0.1
    ) -> NDArray:
        """
        Normalize points to fit within SVG viewport.
        
        Args:
            points: Input points (n, 2)
            margin: Margin as fraction of dimension
            
        Returns:
            Normalized points
        """
        x, y = points[:, 0], points[:, 1]
        
        # Get data bounds
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Add margin
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_min -= margin * x_range
        x_max += margin * x_range
        y_min -= margin * y_range
        y_max += margin * y_range
        
        # Normalize to [0, 1]
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        
        # Scale to SVG coordinates
        x_svg = x_norm * self.width
        y_svg = (1 - y_norm) * self.height  # Flip Y axis (SVG y increases downward)
        
        return np.column_stack([x_svg, y_svg])
    
    def render_points(
        self,
        points: NDArray,
        radius: float = 1.0,
        colors: Optional[NDArray] = None,
        alpha: float = 0.8,
    ):
        """
        Render points as circles.
        
        Args:
            points: Array of shape (n, 2)
            radius: Circle radius
            colors: Optional color array
            alpha: Opacity
        """
        # Normalize points
        points_svg = self._normalize_points(points)
        
        # Determine colors
        if colors is None:
            color_values = np.linspace(0, 1, len(points))
            colors = self.colormap.map(color_values)
        elif colors.ndim == 1:
            colors = self.colormap.map(colors)
        
        # Generate SVG circles
        for i, (x, y) in enumerate(points_svg):
            color = colors[i]
            r, g, b = color[:3]
            a = color[3] * alpha if len(color) > 3 else alpha
            
            # Convert to hex color
            hex_color = mcolors.to_hex([r, g, b])
            
            circle = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" ' \
                    f'fill="{hex_color}" fill-opacity="{a:.2f}" />'
            self.elements.append(circle)
    
    def render_path(
        self,
        points: NDArray,
        stroke_width: float = 1.0,
        stroke_color: str = "white",
        fill: str = "none",
        alpha: float = 0.8,
        closed: bool = False,
    ):
        """
        Render points as a path (polyline).
        
        Args:
            points: Array of shape (n, 2)
            stroke_width: Line width
            stroke_color: Line color
            fill: Fill color
            alpha: Opacity
            closed: Whether to close the path
        """
        # Normalize points
        points_svg = self._normalize_points(points)
        
        # Build path data
        path_data = f"M {points_svg[0, 0]:.2f},{points_svg[0, 1]:.2f}"
        
        for x, y in points_svg[1:]:
            path_data += f" L {x:.2f},{y:.2f}"
        
        if closed:
            path_data += " Z"
        
        # Create path element
        path = f'<path d="{path_data}" ' \
               f'stroke="{stroke_color}" stroke-width="{stroke_width}" ' \
               f'fill="{fill}" stroke-opacity="{alpha:.2f}" />'
        self.elements.append(path)
    
    def render_gradient_path(
        self,
        points: NDArray,
        stroke_width: float = 1.0,
        alpha: float = 0.8,
    ):
        """
        Render path with gradient coloring along the curve.
        
        Args:
            points: Array of shape (n, 2)
            stroke_width: Line width
            alpha: Opacity
        """
        # Normalize points
        points_svg = self._normalize_points(points)
        
        # Generate segments with colors
        color_values = np.linspace(0, 1, len(points))
        colors = self.colormap.map(color_values)
        
        for i in range(len(points_svg) - 1):
            x1, y1 = points_svg[i]
            x2, y2 = points_svg[i + 1]
            
            color = colors[i]
            hex_color = mcolors.to_hex(color[:3])
            
            line = f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" ' \
                   f'stroke="{hex_color}" stroke-width="{stroke_width}" ' \
                   f'stroke-opacity="{alpha:.2f}" />'
            self.elements.append(line)
    
    def clear(self):
        """Clear all elements."""
        self.elements = []
    
    def save(self, filename: str):
        """
        Save SVG to file.
        
        Args:
            filename: Output filename
        """
        # Ensure .svg extension
        if not filename.endswith('.svg'):
            filename += '.svg'
        
        # Determine viewbox
        if self.viewbox is not None:
            vb = f"{self.viewbox[0]} {self.viewbox[1]} {self.viewbox[2]} {self.viewbox[3]}"
        else:
            vb = f"0 0 {self.width} {self.height}"
        
        # Build SVG content
        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{self.width}" height="{self.height}" viewBox="{vb}" 
     xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="{self.background}"/>
'''
        
        svg_footer = '</svg>'
        
        # Write to file
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(svg_header)
            for element in self.elements:
                f.write('  ' + element + '\n')
            f.write(svg_footer)
        
        print(f"Saved SVG to: {filename}")
    
    def to_string(self) -> str:
        """
        Get SVG as string.
        
        Returns:
            SVG content as string
        """
        if self.viewbox is not None:
            vb = f"{self.viewbox[0]} {self.viewbox[1]} {self.viewbox[2]} {self.viewbox[3]}"
        else:
            vb = f"0 0 {self.width} {self.height}"
        
        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg width="{self.width}" height="{self.height}" viewBox="{vb}" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'  <rect width="100%" height="100%" fill="{self.background}"/>',
        ]
        
        svg_parts.extend(['  ' + elem for elem in self.elements])
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)


class SVGMultiLayer:
    """
    Create SVG with multiple layers.
    """
    
    def __init__(
        self,
        width: float = 1000,
        height: float = 1000,
        background: str = "black",
    ):
        """
        Initialize multi-layer SVG renderer.
        
        Args:
            width: SVG width
            height: SVG height
            background: Background color
        """
        self.width = width
        self.height = height
        self.background = background
        self.layers: List[VectorRenderer] = []
    
    def add_layer(self) -> VectorRenderer:
        """
        Add a new layer.
        
        Returns:
            VectorRenderer for the new layer
        """
        layer = VectorRenderer(
            width=self.width,
            height=self.height,
            background="none",  # Transparent
        )
        self.layers.append(layer)
        return layer
    
    def save(self, filename: str):
        """Save multi-layer SVG."""
        if not filename.endswith('.svg'):
            filename += '.svg'
        
        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'  <rect width="100%" height="100%" fill="{self.background}"/>',
        ]
        
        # Add each layer as a group
        for i, layer in enumerate(self.layers):
            svg_parts.append(f'  <g id="layer{i}">')
            for elem in layer.elements:
                svg_parts.append('    ' + elem)
            svg_parts.append('  </g>')
        
        svg_parts.append('</svg>')
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write('\n'.join(svg_parts))
        
        print(f"Saved multi-layer SVG to: {filename}")

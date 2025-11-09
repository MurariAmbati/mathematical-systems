"""
Export Utilities

Save/export art in various formats with metadata.
"""

import json
from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from core.generators.base import ArtGenerator


class ArtExporter:
    """
    Export mathematical art with metadata.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize art exporter.
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_with_metadata(
        self,
        points: NDArray,
        filename: str,
        generator: Optional[ArtGenerator] = None,
        expression: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Export art with accompanying metadata JSON.
        
        Args:
            points: Point data
            filename: Base filename (without extension)
            generator: Optional generator object
            expression: Optional expression string
            metadata: Optional additional metadata
        """
        base_path = self.output_dir / filename
        
        # Save metadata
        meta = self._build_metadata(points, generator, expression, metadata)
        meta_path = base_path.with_suffix('.json')
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Saved metadata to: {meta_path}")
        
        return meta
    
    def _build_metadata(
        self,
        points: NDArray,
        generator: Optional[ArtGenerator],
        expression: Optional[str],
        additional: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build metadata dictionary."""
        meta = {
            "timestamp": datetime.now().isoformat(),
            "points_shape": list(points.shape),
            "points_dtype": str(points.dtype),
            "points_bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
                "mean": points.mean(axis=0).tolist(),
                "std": points.std(axis=0).tolist(),
            }
        }
        
        if generator is not None:
            meta["generator"] = generator.get_config()
        
        if expression is not None:
            meta["expression"] = expression
        
        if additional is not None:
            meta.update(additional)
        
        return meta
    
    def save_points_numpy(self, points: NDArray, filename: str):
        """
        Save points as NumPy binary file.
        
        Args:
            points: Point data
            filename: Output filename
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix('.npy')
        
        np.save(filepath, points)
        print(f"Saved points to: {filepath}")
    
    def save_points_csv(self, points: NDArray, filename: str):
        """
        Save points as CSV file.
        
        Args:
            points: Point data
            filename: Output filename
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix('.csv')
        
        # Determine column names
        if points.shape[1] == 2:
            header = "x,y"
        elif points.shape[1] == 3:
            header = "x,y,z"
        else:
            header = ",".join([f"col{i}" for i in range(points.shape[1])])
        
        np.savetxt(filepath, points, delimiter=',', header=header, comments='')
        print(f"Saved points to: {filepath}")
    
    def save_config(
        self,
        config: Dict[str, Any],
        filename: str = "config.json"
    ):
        """
        Save configuration to JSON.
        
        Args:
            config: Configuration dictionary
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved config to: {filepath}")
    
    def load_config(self, filename: str = "config.json") -> Dict[str, Any]:
        """
        Load configuration from JSON.
        
        Args:
            filename: Input filename
            
        Returns:
            Configuration dictionary
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return config
    
    def load_points_numpy(self, filename: str) -> NDArray:
        """Load points from NumPy file."""
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix('.npy')
        
        return np.load(filepath)
    
    def load_points_csv(self, filename: str) -> NDArray:
        """Load points from CSV file."""
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix('.csv')
        
        return np.loadtxt(filepath, delimiter=',', skiprows=1)


def to_image(
    points: NDArray,
    filename: str,
    width: int = 1024,
    height: int = 1024,
    dpi: int = 100,
    **kwargs
):
    """
    Quick export points to image.
    
    Args:
        points: Point data
        filename: Output filename
        width: Image width
        height: Image height
        dpi: DPI
        **kwargs: Additional rendering parameters
    """
    from core.renderers.static_renderer import StaticRenderer
    
    renderer = StaticRenderer(width=width, height=height, dpi=dpi, **kwargs)
    renderer.render(points)
    renderer.save(filename)
    renderer.close()


def to_svg(
    points: NDArray,
    filename: str,
    width: float = 1000,
    height: float = 1000,
    render_as: str = "path",
    **kwargs
):
    """
    Quick export points to SVG.
    
    Args:
        points: Point data
        filename: Output filename
        width: SVG width
        height: SVG height
        render_as: "path" or "points"
        **kwargs: Additional rendering parameters
    """
    from core.renderers.vector_renderer import VectorRenderer
    
    renderer = VectorRenderer(width=width, height=height, **kwargs)
    
    if render_as == "path":
        renderer.render_path(points[:, :2])
    else:
        renderer.render_points(points[:, :2])
    
    renderer.save(filename)


def to_video(
    generator: Union[ArtGenerator, 'Callable'],
    filename: str,
    frames: int = 120,
    fps: int = 30,
    width: int = 1024,
    height: int = 1024,
    **kwargs
):
    """
    Quick export animation to video.
    
    Args:
        generator: Generator or callable
        filename: Output filename
        frames: Number of frames
        fps: Frames per second
        width: Frame width
        height: Frame height
        **kwargs: Additional rendering parameters
    """
    from core.renderers.animation_renderer import AnimationRenderer
    
    anim_renderer = AnimationRenderer(
        generator=generator,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        **kwargs
    )
    
    anim_renderer.render_to_video(filename)


def to_json(
    config: Dict[str, Any],
    filename: str,
    indent: int = 2
):
    """
    Export configuration to JSON.
    
    Args:
        config: Configuration dictionary
        filename: Output filename
        indent: JSON indentation
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=indent)
    
    print(f"Saved JSON to: {filename}")


def batch_export(
    generators: list,
    output_dir: str = "output",
    formats: list = ["png", "svg"],
    **kwargs
):
    """
    Batch export multiple generators.
    
    Args:
        generators: List of (name, generator) tuples
        output_dir: Output directory
        formats: List of formats to export
        **kwargs: Additional rendering parameters
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exporter = ArtExporter(output_dir)
    
    for name, generator in generators:
        print(f"\nExporting {name}...")
        
        # Generate points
        points = generator.generate()
        
        # Export in each format
        for fmt in formats:
            if fmt == "png":
                to_image(points, str(output_path / f"{name}.png"), **kwargs)
            elif fmt == "svg":
                to_svg(points, str(output_path / f"{name}.svg"))
            elif fmt == "npy":
                exporter.save_points_numpy(points, f"{name}.npy")
            elif fmt == "csv":
                exporter.save_points_csv(points, f"{name}.csv")
        
        # Export metadata
        exporter.export_with_metadata(
            points,
            name,
            generator=generator if isinstance(generator, ArtGenerator) else None
        )


def create_gallery(
    output_dir: str,
    title: str = "Math Art Gallery",
    html_file: str = "gallery.html"
):
    """
    Create HTML gallery from exported images.
    
    Args:
        output_dir: Directory containing images
        title: Gallery title
        html_file: Output HTML filename
    """
    output_path = Path(output_dir)
    
    # Find all PNG images
    images = sorted(output_path.glob("*.png"))
    
    if not images:
        print("No PNG images found in output directory")
        return
    
    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"  <title>{title}</title>",
        "  <style>",
        "    body { background: #000; color: #fff; font-family: Arial; }",
        "    .gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px; }",
        "    .item { text-align: center; }",
        "    .item img { width: 100%; border-radius: 8px; }",
        "    h1 { text-align: center; padding: 20px; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
        "  <div class='gallery'>",
    ]
    
    for img_path in images:
        name = img_path.stem
        html_parts.append(f"    <div class='item'>")
        html_parts.append(f"      <img src='{img_path.name}' alt='{name}'>")
        html_parts.append(f"      <p>{name}</p>")
        html_parts.append(f"    </div>")
    
    html_parts.extend([
        "  </div>",
        "</body>",
        "</html>",
    ])
    
    # Write HTML file
    html_path = output_path / html_file
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_parts))
    
    print(f"Created gallery: {html_path}")

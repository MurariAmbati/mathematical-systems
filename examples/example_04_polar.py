"""
Example 4: Polar Patterns

Generate beautiful polar coordinate art.
"""

from core.generators.polar_patterns import (
    PolarPattern,
    Cardioid,
    Limacon,
    ArchimedeanSpiral,
    LogarithmicSpiral,
    MaunderRose
)
from core.renderers.static_renderer import StaticRenderer
from core.export import to_svg
import numpy as np


def example_polar_rose():
    """Polar rose pattern."""
    print("Generating polar rose...")
    
    # r = sin(5*theta)
    pattern = PolarPattern(
        expr="sin(5*theta)",
        samples=2000,
        seed=42
    )
    points = pattern.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="fire",
        background="black",
        style="line"
    )
    renderer.render(points, sizes=2.0, alpha=0.8)
    renderer.save("output/polar_rose.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/polar_rose.png")


def example_cardioid():
    """Cardioid (heart shape)."""
    print("Generating cardioid...")
    
    cardioid = Cardioid(a=1.0, samples=2000, seed=42)
    points = cardioid.generate()
    
    # Render as SVG
    to_svg(points, "output/cardioid.svg", render_as="path", colormap="sunset")
    
    # Also as raster
    renderer = StaticRenderer(width=2048, height=2048, colormap="sunset", background="black")
    renderer.render(points, alpha=0.9)
    renderer.save("output/cardioid.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/cardioid.png and output/cardioid.svg")


def example_limacon():
    """Limaçon curve."""
    print("Generating limaçon...")
    
    limacon = Limacon(a=1.0, b=0.7, samples=2000, seed=42)
    points = limacon.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="northern_lights",
        background="black"
    )
    renderer.render(points, alpha=0.85)
    renderer.save("output/limacon.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/limacon.png")


def example_spirals():
    """Archimedean and logarithmic spirals."""
    print("Generating spirals...")
    
    from core.renderers.static_renderer import MultiPanelRenderer
    
    # Create two types of spirals
    arch_spiral = ArchimedeanSpiral(a=0, b=0.5, n_turns=5, samples=3000, seed=42)
    log_spiral = LogarithmicSpiral(a=1.0, b=0.2, n_turns=3, samples=3000, seed=42)
    
    multi = MultiPanelRenderer(rows=1, cols=2, width=4096, height=2048, background="black")
    
    # Archimedean spiral
    points1 = arch_spiral.generate()
    multi.render_panel(0, points1, colormap="viridis")
    
    # Logarithmic spiral
    points2 = log_spiral.generate()
    multi.render_panel(1, points2, colormap="plasma")
    
    multi.save("output/spirals.png")
    multi.close()
    
    print("✓ Saved: output/spirals.png")


def example_maurer_rose():
    """Maurer rose with modular arithmetic."""
    print("Generating Maurer rose...")
    
    maurer = MaunderRose(n=7, d=29, samples=360, seed=42)
    points = maurer.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="rainbow",
        background="black",
        style="line"
    )
    renderer.render(points, sizes=2.0, alpha=0.9)
    renderer.save("output/maurer_rose.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/maurer_rose.png")


def example_complex_polar():
    """Complex polar equation."""
    print("Generating complex polar pattern...")
    
    # r = sin(3*theta) * cos(5*theta)
    pattern = PolarPattern(
        expr="sin(3*theta) * cos(5*theta)",
        samples=3000,
        seed=42
    )
    points = pattern.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="twilight",
        background="black"
    )
    renderer.render(points, alpha=0.8)
    renderer.save("output/complex_polar.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/complex_polar.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Polar Pattern Examples")
    print("=" * 60)
    print()
    
    example_polar_rose()
    print()
    
    example_cardioid()
    print()
    
    example_limacon()
    print()
    
    example_spirals()
    print()
    
    example_maurer_rose()
    print()
    
    example_complex_polar()
    print()
    
    print("=" * 60)
    print("All polar pattern examples completed!")
    print("=" * 60)

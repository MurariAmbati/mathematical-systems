"""
Example 3: Lissajous Curves

Generate beautiful Lissajous figures and parametric curves.
"""

from core.generators.lissajous import Lissajous, Lissajous3D, BowditchCurve
from core.renderers.static_renderer import StaticRenderer
from core.renderers.vector_renderer import VectorRenderer
from core.export import to_svg
import numpy as np


def example_basic_lissajous():
    """Basic Lissajous curve."""
    print("Generating basic Lissajous curve...")
    
    lissajous = Lissajous(
        A=1.0, B=1.0,
        a=3, b=2,
        delta=np.pi/2,
        samples=5000,
        seed=42
    )
    points = lissajous.generate()
    
    # Render as line
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="cool",
        background="black",
        style="line"
    )
    renderer.render(points, sizes=2.0, alpha=0.8)
    renderer.save("output/lissajous_basic.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/lissajous_basic.png")


def example_lissajous_svg():
    """Lissajous curve as vector graphic."""
    print("Generating Lissajous SVG...")
    
    lissajous = Lissajous(A=1.0, B=1.0, a=5, b=4, delta=0, samples=3000, seed=42)
    points = lissajous.generate()
    
    # Render as SVG path
    svg_renderer = VectorRenderer(width=1000, height=1000, background="white", colormap="rainbow")
    svg_renderer.render_gradient_path(points, stroke_width=2.0, alpha=0.9)
    svg_renderer.save("output/lissajous_vector.svg")
    
    print("✓ Saved: output/lissajous_vector.svg")


def example_lissajous_3d():
    """3D Lissajous curve."""
    print("Generating 3D Lissajous curve...")
    
    lissajous_3d = Lissajous3D(
        A=1.0, B=1.0, C=1.0,
        a=3, b=2, c=1,
        delta_x=0,
        delta_y=np.pi/2,
        delta_z=np.pi/4,
        samples=5000,
        seed=42
    )
    points = lissajous_3d.generate()
    
    # Render 3D projection
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="twilight",
        background="black"
    )
    renderer.render(points, sizes=1.0, alpha=0.6)
    renderer.save("output/lissajous_3d.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/lissajous_3d.png")


def example_lissajous_variations():
    """Multiple Lissajous variations."""
    print("Generating Lissajous variations...")
    
    from core.renderers.static_renderer import MultiPanelRenderer
    
    # Different frequency ratios
    params = [
        (3, 2, 0),
        (5, 4, np.pi/2),
        (7, 5, np.pi/4),
        (9, 8, 0),
    ]
    
    multi = MultiPanelRenderer(rows=2, cols=2, width=4096, height=4096, background="black")
    
    for idx, (a, b, delta) in enumerate(params):
        lissajous = Lissajous(A=1, B=1, a=a, b=b, delta=delta, samples=5000, seed=42)
        points = lissajous.generate()
        multi.render_panel(idx, points, colormap=["viridis", "plasma", "inferno", "magma"][idx])
    
    multi.save("output/lissajous_variations.png")
    multi.close()
    
    print("✓ Saved: output/lissajous_variations.png")


def example_bowditch_curve():
    """Bowditch curve (generalized Lissajous)."""
    print("Generating Bowditch curve...")
    
    bowditch = BowditchCurve(
        A=1.0, B=1.0,
        A2=0.5, B2=0.5,
        a=3, b=2,
        a2=5, b2=4,
        delta=np.pi/4,
        samples=8000,
        seed=42
    )
    points = bowditch.generate()
    
    # Render with velocity-based coloring
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="ocean",
        background="black"
    )
    renderer.render_with_function_colors(points, mode="velocity", alpha=0.7)
    renderer.save("output/bowditch_curve.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/bowditch_curve.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Lissajous Curve Examples")
    print("=" * 60)
    print()
    
    example_basic_lissajous()
    print()
    
    example_lissajous_svg()
    print()
    
    example_lissajous_3d()
    print()
    
    example_lissajous_variations()
    print()
    
    example_bowditch_curve()
    print()
    
    print("=" * 60)
    print("All Lissajous examples completed!")
    print("=" * 60)

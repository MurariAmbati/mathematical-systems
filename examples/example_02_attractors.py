"""
Example 2: Attractors

Generate strange attractors and chaotic systems.
"""

from core.generators.attractors import (
    LorenzAttractor,
    CliffordAttractor,
    IkedaAttractor,
    DeJongAttractor
)
from core.renderers.static_renderer import StaticRenderer
from core.export import to_image

def example_lorenz_attractor():
    """Classic Lorenz attractor."""
    print("Generating Lorenz attractor...")
    
    lorenz = LorenzAttractor(
        sigma=10.0,
        rho=28.0,
        beta=8/3,
        dt=0.01,
        iterations=50000,
        seed=42
    )
    points = lorenz.generate()
    
    # Render with plasma colormap
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="plasma",
        background="black",
        style="scatter"
    )
    renderer.render(points, sizes=0.5, alpha=0.4)
    renderer.save("output/lorenz_attractor.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/lorenz_attractor.png")


def example_clifford_attractor():
    """Clifford attractor - intricate fractal pattern."""
    print("Generating Clifford attractor...")
    
    clifford = CliffordAttractor(
        a=-1.4,
        b=1.6,
        c=1.0,
        d=0.7,
        iterations=100000,
        seed=42
    )
    points = clifford.generate()
    
    # Render as density plot
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="viridis",
        background="black",
        style="density"
    )
    renderer.render(points, alpha=0.6)
    renderer.save("output/clifford_attractor.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/clifford_attractor.png")


def example_ikeda_attractor():
    """Ikeda attractor - laser dynamics."""
    print("Generating Ikeda attractor...")
    
    ikeda = IkedaAttractor(u=0.918, iterations=30000, seed=42)
    points = ikeda.generate()
    
    # Render with custom colors
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="inferno",
        background="black"
    )
    renderer.render(points, sizes=1.0, alpha=0.5)
    renderer.save("output/ikeda_attractor.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/ikeda_attractor.png")


def example_dejong_attractor():
    """De Jong attractor variations."""
    print("Generating De Jong attractor...")
    
    dejong = DeJongAttractor(
        a=1.4,
        b=-2.3,
        c=2.4,
        d=-2.1,
        iterations=100000,
        seed=42
    )
    points = dejong.generate()
    
    # Render with magma colormap
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="magma",
        background="black",
        style="scatter"
    )
    renderer.render(points, sizes=0.3, alpha=0.3)
    renderer.save("output/dejong_attractor.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/dejong_attractor.png")


def example_attractor_gallery():
    """Create a gallery of attractors."""
    print("Generating attractor gallery...")
    
    from core.renderers.static_renderer import MultiPanelRenderer
    
    attractors = [
        LorenzAttractor(iterations=20000, seed=42),
        CliffordAttractor(iterations=50000, seed=42),
        IkedaAttractor(iterations=15000, seed=42),
        DeJongAttractor(iterations=50000, seed=42),
    ]
    
    names = ["Lorenz", "Clifford", "Ikeda", "De Jong"]
    colormaps = ["plasma", "viridis", "inferno", "magma"]
    
    multi = MultiPanelRenderer(rows=2, cols=2, width=4096, height=4096, background="black")
    
    for idx, (attractor, cmap) in enumerate(zip(attractors, colormaps)):
        points = attractor.generate()
        multi.render_panel(idx, points, colormap=cmap)
    
    multi.save("output/attractor_gallery.png")
    multi.close()
    
    print("✓ Saved: output/attractor_gallery.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Strange Attractor Examples")
    print("=" * 60)
    print()
    
    example_lorenz_attractor()
    print()
    
    example_clifford_attractor()
    print()
    
    example_ikeda_attractor()
    print()
    
    example_dejong_attractor()
    print()
    
    example_attractor_gallery()
    print()
    
    print("=" * 60)
    print("All attractor examples completed!")
    print("=" * 60)

"""
Example 1: Spirograph Patterns

Generate beautiful spirograph art with various parameters.
"""

from core.generators.spirograph import Spirograph, RosePattern
from core.renderers.static_renderer import StaticRenderer
from core.export import ArtExporter, to_image, to_svg

def example_basic_spirograph():
    """Basic spirograph example."""
    print("Generating basic spirograph...")
    
    # Create spirograph with specific parameters
    spiro = Spirograph(R=8, r=1, a=4, mode="hypo", samples=5000, seed=42)
    points = spiro.generate()
    
    # Render with high quality
    renderer = StaticRenderer(width=2048, height=2048, colormap="magma", background="black")
    renderer.render(points, alpha=0.7)
    renderer.save("output/spirograph_basic.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/spirograph_basic.png")


def example_rose_pattern():
    """Rose pattern example."""
    print("Generating rose pattern...")
    
    # Create a 7-petaled rose
    rose = RosePattern(A=1.0, k=7, samples=3000, seed=42)
    points = rose.generate()
    
    # Render with cool colormap
    renderer = StaticRenderer(width=2048, height=2048, colormap="cool", background="black")
    renderer.render(points, alpha=0.8)
    renderer.save("output/rose_pattern.png", dpi=300)
    renderer.close()
    
    # Also export as SVG
    to_svg(points, "output/rose_pattern.svg", render_as="path")
    
    print("✓ Saved: output/rose_pattern.png and output/rose_pattern.svg")


def example_spirograph_variations():
    """Create multiple spirograph variations."""
    print("Generating spirograph variations...")
    
    from core.renderers.static_renderer import MultiPanelRenderer
    
    # Parameters for different spirographs
    configs = [
        {"R": 5, "r": 3, "a": 2, "mode": "hypo"},
        {"R": 7, "r": 2, "a": 3, "mode": "hypo"},
        {"R": 10, "r": 3, "a": 5, "mode": "epi"},
        {"R": 6, "r": 4, "a": 2, "mode": "epi"},
    ]
    
    colormaps = ["viridis", "plasma", "inferno", "magma"]
    
    # Create multi-panel renderer
    multi = MultiPanelRenderer(rows=2, cols=2, width=4096, height=4096, background="black")
    
    for idx, (config, cmap) in enumerate(zip(configs, colormaps)):
        spiro = Spirograph(**config, samples=5000, seed=42)
        points = spiro.generate()
        multi.render_panel(idx, points, colormap=cmap)
    
    multi.save("output/spirograph_variations.png")
    multi.close()
    
    print("✓ Saved: output/spirograph_variations.png")


def example_with_metadata():
    """Example with metadata export."""
    print("Generating spirograph with metadata...")
    
    spiro = Spirograph(R=12, r=5, a=3, mode="hypo", samples=10000, seed=123)
    points = spiro.generate()
    
    # Export with metadata
    exporter = ArtExporter("output")
    to_image(points, "output/spirograph_metadata.png", width=2048, height=2048, colormap="twilight")
    exporter.export_with_metadata(
        points,
        "spirograph_metadata",
        generator=spiro,
        metadata={"description": "Hypotrochoid pattern with R=12, r=5, a=3"}
    )
    
    print("✓ Saved: output/spirograph_metadata.png and metadata JSON")


if __name__ == "__main__":
    print("=" * 60)
    print("Spirograph Examples")
    print("=" * 60)
    print()
    
    example_basic_spirograph()
    print()
    
    example_rose_pattern()
    print()
    
    example_spirograph_variations()
    print()
    
    example_with_metadata()
    print()
    
    print("=" * 60)
    print("All spirograph examples completed!")
    print("=" * 60)

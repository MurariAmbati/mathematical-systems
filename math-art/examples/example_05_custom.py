"""
Example 5: Custom Equations

Generate art from user-defined mathematical expressions.
"""

from core.generators.custom_equation import CustomEquation, NoiseField, HybridGenerator
from core.renderers.static_renderer import StaticRenderer
from core.parser import parse_equation
import numpy as np


def example_simple_expression():
    """Simple custom expression."""
    print("Generating art from simple expression...")
    
    # z = sin(x*y) + cos(x^2 - y^2)
    gen = CustomEquation(
        expr="sin(x*y) + cos(x**2 - y**2)",
        x_range=(-3, 3),
        y_range=(-3, 3),
        samples=200,
        mode="grid",
        seed=42
    )
    points = gen.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="viridis",
        background="black"
    )
    renderer.render_with_function_colors(points, mode="value", alpha=0.7)
    renderer.save("output/custom_simple.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/custom_simple.png")


def example_wave_interference():
    """Wave interference pattern."""
    print("Generating wave interference...")
    
    # Interference of circular waves
    gen = CustomEquation(
        expr="sin(sqrt((x-1)**2 + y**2)*5) + sin(sqrt((x+1)**2 + y**2)*5)",
        x_range=(-3, 3),
        y_range=(-3, 3),
        samples=300,
        mode="grid",
        seed=42
    )
    points = gen.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="twilight",
        background="black"
    )
    renderer.render_with_function_colors(points, mode="value", alpha=0.8)
    renderer.save("output/wave_interference.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/wave_interference.png")


def example_mandala_pattern():
    """Mandala-like pattern using trigonometry."""
    print("Generating mandala pattern...")
    
    # Complex trigonometric expression
    gen = CustomEquation(
        expr="sin(x**2 + y**2) * cos(x*y) + sin(5*atan2(y, x))",
        x_range=(-2, 2),
        y_range=(-2, 2),
        samples=300,
        mode="grid",
        seed=42
    )
    points = gen.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="rainbow",
        background="black"
    )
    renderer.render_with_function_colors(points, mode="value", alpha=0.75)
    renderer.save("output/mandala.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/mandala.png")


def example_noise_field():
    """Pure Perlin noise field."""
    print("Generating noise field...")
    
    noise_gen = NoiseField(
        x_range=(-5, 5),
        y_range=(-5, 5),
        samples=150,
        noise_type="perlin",
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        scale=0.5,
        seed=42
    )
    points = noise_gen.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="forest",
        background="black"
    )
    renderer.render_with_function_colors(points, mode="value", alpha=0.8)
    renderer.save("output/noise_field.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/noise_field.png")


def example_hybrid_art():
    """Hybrid expression + noise."""
    print("Generating hybrid art...")
    
    hybrid = HybridGenerator(
        expr="sin(x*3) * cos(y*3)",
        x_range=(-2, 2),
        y_range=(-2, 2),
        samples=200,
        noise_weight=0.3,
        noise_scale=0.8,
        seed=42
    )
    points = hybrid.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="ocean",
        background="black"
    )
    renderer.render_with_function_colors(points, mode="value", alpha=0.8)
    renderer.save("output/hybrid.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/hybrid.png")


def example_contour_art():
    """Contour line art."""
    print("Generating contour art...")
    
    gen = CustomEquation(
        expr="sin(x*2) + cos(y*2)",
        x_range=(-3, 3),
        y_range=(-3, 3),
        samples=300,
        mode="contour",
        threshold=0.0,
        seed=42
    )
    points = gen.generate()
    
    renderer = StaticRenderer(
        width=2048,
        height=2048,
        colormap="cool",
        background="white"
    )
    renderer.render(points, sizes=0.5, alpha=0.8)
    renderer.save("output/contour_art.png", dpi=300)
    renderer.close()
    
    print("✓ Saved: output/contour_art.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Custom Equation Examples")
    print("=" * 60)
    print()
    
    example_simple_expression()
    print()
    
    example_wave_interference()
    print()
    
    example_mandala_pattern()
    print()
    
    example_noise_field()
    print()
    
    example_hybrid_art()
    print()
    
    example_contour_art()
    print()
    
    print("=" * 60)
    print("All custom equation examples completed!")
    print("=" * 60)

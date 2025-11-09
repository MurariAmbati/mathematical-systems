"""
Quick Start Guide - Math Art Generator

Run this to test the installation and generate your first art!
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from core import parse_equation
        from core.generators import Spirograph, Lissajous, LorenzAttractor
        from core.renderers import StaticRenderer
        from core.export import to_image
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def generate_first_art():
    """Generate your first piece of mathematical art!"""
    print("\nGenerating your first mathematical art...")
    
    try:
        from core.generators.spirograph import Spirograph
        from core.renderers.static_renderer import StaticRenderer
        
        # Create output directory
        Path("output").mkdir(exist_ok=True)
        
        # Generate a beautiful spirograph
        print("  Creating spirograph...")
        spiro = Spirograph(R=8, r=1, a=4, samples=5000, seed=42)
        points = spiro.generate()
        
        # Render it
        print("  Rendering...")
        renderer = StaticRenderer(
            width=1024,
            height=1024,
            colormap="magma",
            background="black"
        )
        renderer.render(points, alpha=0.7)
        renderer.save("output/my_first_art.png")
        renderer.close()
        
        print("✓ Success! Saved to: output/my_first_art.png")
        return True
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_examples():
    """Show available examples."""
    print("\n" + "="*60)
    print("Available Examples")
    print("="*60)
    
    examples = [
        ("examples/example_01_spirograph.py", "Spirograph patterns"),
        ("examples/example_02_attractors.py", "Strange attractors"),
        ("examples/example_03_lissajous.py", "Lissajous curves"),
        ("examples/example_04_polar.py", "Polar patterns"),
        ("examples/example_05_custom.py", "Custom equations"),
        ("examples/run_all_examples.py", "Run all examples"),
    ]
    
    for path, description in examples:
        print(f"  • {path:<40} - {description}")
    
    print("\nTo run an example:")
    print("  python examples/example_01_spirograph.py")
    print("\nTo run all examples:")
    print("  python examples/run_all_examples.py")


def show_quick_usage():
    """Show quick usage examples."""
    print("\n" + "="*60)
    print("Quick Usage Examples")
    print("="*60)
    
    print("""
1. Spirograph:
    from core.generators import Spirograph
    from core.export import to_image
    
    spiro = Spirograph(R=8, r=1, a=4, samples=5000)
    points = spiro.generate()
    to_image(points, "spirograph.png")

2. Lissajous Curve:
    from core.generators import Lissajous
    from core.export import to_svg
    
    lissajous = Lissajous(a=3, b=2, samples=3000)
    points = lissajous.generate()
    to_svg(points, "lissajous.svg")

3. Custom Equation:
    from core.generators import CustomEquation
    from core.export import to_image
    
    gen = CustomEquation("sin(x*y) + cos(x**2 - y**2)")
    points = gen.generate()
    to_image(points, "custom.png")

4. Lorenz Attractor:
    from core.generators import LorenzAttractor
    from core.export import to_image
    
    lorenz = LorenzAttractor(iterations=50000)
    points = lorenz.generate()
    to_image(points, "lorenz.png", colormap="plasma")
    """)


def main():
    """Main quick start routine."""
    print("="*60)
    print(" "*15 + "MATH ART GENERATOR")
    print(" "*18 + "Quick Start")
    print("="*60)
    
    # Test imports
    if not test_imports():
        print("\n⚠ Please install dependencies:")
        print("  pip install -e .")
        return
    
    # Generate first art
    print()
    if generate_first_art():
        print("\n🎨 Congratulations! You've created your first mathematical art!")
    
    # Show examples
    show_examples()
    
    # Show usage
    show_quick_usage()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Open output/my_first_art.png to see your art")
    print("2. Try running the example scripts")
    print("3. Experiment with parameters")
    print("4. Read docs/architecture.md for details")
    print("="*60)


if __name__ == "__main__":
    main()

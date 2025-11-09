"""
Master Example Script

Run all examples and create a comprehensive gallery.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

# Import all example modules
import examples.example_01_spirograph as ex1
import examples.example_02_attractors as ex2
import examples.example_03_lissajous as ex3
import examples.example_04_polar as ex4
import examples.example_05_custom as ex5


def create_output_directory():
    """Ensure output directory exists."""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()


def run_all_examples():
    """Run all example scripts."""
    print("=" * 80)
    print(" " * 25 + "MATH ART GENERATOR")
    print(" " * 20 + "Comprehensive Example Suite")
    print("=" * 80)
    print()
    
    create_output_directory()
    
    # Example 1: Spirographs
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Spirograph Patterns")
    print("=" * 80)
    try:
        ex1.example_basic_spirograph()
        ex1.example_rose_pattern()
        ex1.example_spirograph_variations()
        ex1.example_with_metadata()
        print("✓ Spirograph examples completed successfully")
    except Exception as e:
        print(f"✗ Error in spirograph examples: {e}")
    
    # Example 2: Attractors
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Strange Attractors")
    print("=" * 80)
    try:
        ex2.example_lorenz_attractor()
        ex2.example_clifford_attractor()
        ex2.example_ikeda_attractor()
        ex2.example_dejong_attractor()
        ex2.example_attractor_gallery()
        print("✓ Attractor examples completed successfully")
    except Exception as e:
        print(f"✗ Error in attractor examples: {e}")
    
    # Example 3: Lissajous
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Lissajous Curves")
    print("=" * 80)
    try:
        ex3.example_basic_lissajous()
        ex3.example_lissajous_svg()
        ex3.example_lissajous_3d()
        ex3.example_lissajous_variations()
        ex3.example_bowditch_curve()
        print("✓ Lissajous examples completed successfully")
    except Exception as e:
        print(f"✗ Error in Lissajous examples: {e}")
    
    # Example 4: Polar Patterns
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Polar Patterns")
    print("=" * 80)
    try:
        ex4.example_polar_rose()
        ex4.example_cardioid()
        ex4.example_limacon()
        ex4.example_spirals()
        ex4.example_maurer_rose()
        ex4.example_complex_polar()
        print("✓ Polar pattern examples completed successfully")
    except Exception as e:
        print(f"✗ Error in polar pattern examples: {e}")
    
    # Example 5: Custom Equations
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Custom Equations")
    print("=" * 80)
    try:
        ex5.example_simple_expression()
        ex5.example_wave_interference()
        ex5.example_mandala_pattern()
        ex5.example_noise_field()
        ex5.example_hybrid_art()
        ex5.example_contour_art()
        print("✓ Custom equation examples completed successfully")
    except Exception as e:
        print(f"✗ Error in custom equation examples: {e}")


def create_gallery():
    """Create HTML gallery of all generated art."""
    print("\n" + "=" * 80)
    print("Creating Gallery")
    print("=" * 80)
    
    from core.export import create_gallery
    
    try:
        create_gallery(
            output_dir="output",
            title="Math Art Generator - Gallery",
            html_file="gallery.html"
        )
        print("✓ Gallery created successfully")
        print("\nOpen output/gallery.html in your browser to view all artworks!")
    except Exception as e:
        print(f"✗ Error creating gallery: {e}")


def print_summary():
    """Print summary of generated files."""
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    output_dir = Path("output")
    
    if output_dir.exists():
        png_files = list(output_dir.glob("*.png"))
        svg_files = list(output_dir.glob("*.svg"))
        json_files = list(output_dir.glob("*.json"))
        
        print(f"\nGenerated files:")
        print(f"  - PNG images: {len(png_files)}")
        print(f"  - SVG vectors: {len(svg_files)}")
        print(f"  - JSON metadata: {len(json_files)}")
        print(f"\nTotal: {len(png_files) + len(svg_files) + len(json_files)} files")
        print(f"\nOutput directory: {output_dir.absolute()}")
    else:
        print("\nNo output directory found.")
    
    print("\n" + "=" * 80)
    print(" " * 25 + "ALL EXAMPLES COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_all_examples()
        create_gallery()
        print_summary()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

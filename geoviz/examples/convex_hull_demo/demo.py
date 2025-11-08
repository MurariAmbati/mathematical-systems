"""
Convex Hull Demo

This example demonstrates computing and visualizing a 2D convex hull.
"""

import random
from geometry_visualizer.primitives import Point2D
from geometry_visualizer.scene import Scene
from geometry_visualizer.algorithms.convex_hull import convex_hull_2d

def main():
    # Generate random points
    random.seed(42)
    num_points = 20
    points = [
        Point2D(random.uniform(-10, 10), random.uniform(-10, 10))
        for _ in range(num_points)
    ]
    
    # Compute convex hull
    hull = convex_hull_2d(points)
    hull_polygon = hull.to_polygon()
    
    print(f"Generated {len(points)} random points")
    print(f"Convex hull has {len(hull.hull_points)} vertices")
    print(f"Hull area: {hull.area():.2f}")
    print(f"Hull perimeter: {hull.perimeter():.2f}")
    
    # Create scene
    scene = Scene()
    scene.metadata.description = "Convex Hull Demonstration"
    
    # Add all points
    for i, point in enumerate(points):
        scene.add_object(
            id=f"point_{i}",
            type="point2d",
            geometry=point.to_dict(),
            style={
                "color": "#3b82f6" if point in hull.hull_points else "#9ca3af",
                "pointSize": 0.2,
            }
        )
    
    # Add convex hull polygon
    scene.add_object(
        id="convex_hull",
        type="polygon2d",
        geometry=hull_polygon.to_dict(),
        style={
            "fill": "#3b82f6",
            "stroke": "#1e40af",
            "strokeWidth": 2,
            "opacity": 0.3,
        }
    )
    
    # Export scene
    scene.export_json("convex_hull_scene.json")
    print("\nScene exported to convex_hull_scene.json")
    
    # Also export as SVG
    from geometry_visualizer.io import export_svg
    export_svg([hull_polygon], "convex_hull.svg")
    print("SVG exported to convex_hull.svg")


if __name__ == "__main__":
    main()

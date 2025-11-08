"""Tests for convex hull algorithms."""

import pytest
from geometry_visualizer.primitives import Point2D
from geometry_visualizer.algorithms.convex_hull import convex_hull_2d, convex_hull_stepwise


class TestConvexHull2D:
    def test_triangle(self):
        """Test convex hull of a triangle (should be the triangle itself)."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(4.0, 0.0),
            Point2D(2.0, 3.0),
        ]
        hull = convex_hull_2d(points)
        assert len(hull.hull_points) == 3
    
    def test_square(self):
        """Test convex hull of a square."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(1.0, 0.0),
            Point2D(1.0, 1.0),
            Point2D(0.0, 1.0),
        ]
        hull = convex_hull_2d(points)
        assert len(hull.hull_points) == 4
    
    def test_interior_points(self):
        """Test convex hull with interior points."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(4.0, 0.0),
            Point2D(4.0, 4.0),
            Point2D(0.0, 4.0),
            Point2D(2.0, 2.0),  # Interior point
            Point2D(1.0, 1.0),  # Interior point
        ]
        hull = convex_hull_2d(points)
        assert len(hull.hull_points) == 4
        # Check that interior points are not in hull
        interior = Point2D(2.0, 2.0)
        assert interior not in hull.hull_points
    
    def test_collinear_points(self):
        """Test convex hull with collinear points."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(1.0, 1.0),
            Point2D(2.0, 2.0),
            Point2D(3.0, 3.0),
        ]
        hull = convex_hull_2d(points)
        # Hull of collinear points should have 2 endpoints
        assert len(hull.hull_points) == 2
    
    def test_area(self):
        """Test area calculation of convex hull."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(2.0, 0.0),
            Point2D(2.0, 2.0),
            Point2D(0.0, 2.0),
        ]
        hull = convex_hull_2d(points)
        assert abs(hull.area() - 4.0) < 1e-10
    
    def test_algorithm_choice(self):
        """Test that both algorithms produce same result."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(4.0, 0.0),
            Point2D(4.0, 3.0),
            Point2D(2.0, 4.0),
            Point2D(0.0, 3.0),
            Point2D(2.0, 1.0),  # Interior
        ]
        
        hull_monotone = convex_hull_2d(points, algorithm="monotone_chain")
        hull_graham = convex_hull_2d(points, algorithm="graham_scan")
        
        # Both should have same number of points
        assert len(hull_monotone.hull_points) == len(hull_graham.hull_points)
    
    def test_stepwise_visualization(self):
        """Test stepwise algorithm for visualization."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(2.0, 0.0),
            Point2D(2.0, 2.0),
            Point2D(0.0, 2.0),
            Point2D(1.0, 1.0),  # Interior
        ]
        
        steps = convex_hull_stepwise(points)
        assert len(steps) > 0
        assert steps[-1]["completed"] is True
        assert "description" in steps[0]


class TestConvexHullEdgeCases:
    def test_insufficient_points(self):
        """Test error handling for too few points."""
        with pytest.raises(ValueError):
            convex_hull_2d([Point2D(0, 0), Point2D(1, 1)])
    
    def test_duplicate_points(self):
        """Test handling of duplicate points."""
        points = [
            Point2D(0.0, 0.0),
            Point2D(1.0, 0.0),
            Point2D(1.0, 1.0),
            Point2D(0.0, 1.0),
            Point2D(0.0, 0.0),  # Duplicate
        ]
        hull = convex_hull_2d(points)
        # Should still work
        assert len(hull.hull_points) >= 3

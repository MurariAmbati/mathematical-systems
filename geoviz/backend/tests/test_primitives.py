"""Tests for geometric primitives."""

import pytest
import numpy as np
from geometry_visualizer.primitives import (
    Point2D, Point3D, Vector2D, Vector3D,
    Line2D, Segment2D, Polygon2D, Mesh3D
)


class TestPoint2D:
    def test_creation(self):
        p = Point2D(3.0, 4.0)
        assert p.x == 3.0
        assert p.y == 4.0
    
    def test_distance(self):
        p1 = Point2D(0.0, 0.0)
        p2 = Point2D(3.0, 4.0)
        assert p1.distance_to(p2) == 5.0
    
    def test_equality(self):
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.0, 2.0)
        p3 = Point2D(1.0, 3.0)
        assert p1 == p2
        assert p1 != p3
    
    def test_serialization(self):
        p = Point2D(1.5, 2.5)
        d = p.to_dict()
        assert d == {"x": 1.5, "y": 2.5}
        p2 = Point2D.from_dict(d)
        assert p == p2


class TestVector2D:
    def test_length(self):
        v = Vector2D(3.0, 4.0)
        assert v.length() == 5.0
    
    def test_normalize(self):
        v = Vector2D(3.0, 4.0)
        vn = v.normalize()
        assert abs(vn.length() - 1.0) < 1e-10
    
    def test_dot_product(self):
        v1 = Vector2D(1.0, 0.0)
        v2 = Vector2D(0.0, 1.0)
        assert v1.dot(v2) == 0.0
        
        v3 = Vector2D(3.0, 4.0)
        v4 = Vector2D(1.0, 0.0)
        assert v3.dot(v4) == 3.0
    
    def test_cross_product(self):
        v1 = Vector2D(1.0, 0.0)
        v2 = Vector2D(0.0, 1.0)
        # 2D cross product returns scalar (z-component)
        assert v1.cross(v2) == 1.0
    
    def test_arithmetic(self):
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(3.0, 4.0)
        
        v3 = v1 + v2
        assert v3.x == 4.0 and v3.y == 6.0
        
        v4 = v2 - v1
        assert v4.x == 2.0 and v4.y == 2.0
        
        v5 = v1 * 2.0
        assert v5.x == 2.0 and v5.y == 4.0


class TestLine2D:
    def test_from_points(self):
        p1 = Point2D(0.0, 0.0)
        p2 = Point2D(1.0, 1.0)
        line = Line2D.from_points(p1, p2)
        
        # Line passing through (0,0) and (1,1) is x - y = 0 (normalized)
        assert abs(line.distance_to_point(p1)) < 1e-10
        assert abs(line.distance_to_point(p2)) < 1e-10
    
    def test_distance_to_point(self):
        # Horizontal line y = 0
        line = Line2D.from_points(Point2D(0, 0), Point2D(1, 0))
        p = Point2D(0.5, 3.0)
        assert abs(line.distance_to_point(p) - 3.0) < 1e-10


class TestPolygon2D:
    def test_triangle_area(self):
        triangle = Polygon2D(tuple([
            Point2D(0.0, 0.0),
            Point2D(4.0, 0.0),
            Point2D(0.0, 3.0),
        ]))
        assert abs(triangle.area() - 6.0) < 1e-10
    
    def test_square_perimeter(self):
        square = Polygon2D(tuple([
            Point2D(0.0, 0.0),
            Point2D(1.0, 0.0),
            Point2D(1.0, 1.0),
            Point2D(0.0, 1.0),
        ]))
        assert abs(square.perimeter() - 4.0) < 1e-10
    
    def test_centroid(self):
        square = Polygon2D(tuple([
            Point2D(0.0, 0.0),
            Point2D(2.0, 0.0),
            Point2D(2.0, 2.0),
            Point2D(0.0, 2.0),
        ]))
        centroid = square.centroid()
        assert abs(centroid.x - 1.0) < 1e-10
        assert abs(centroid.y - 1.0) < 1e-10
    
    def test_bounding_box(self):
        poly = Polygon2D(tuple([
            Point2D(1.0, 1.0),
            Point2D(5.0, 2.0),
            Point2D(3.0, 6.0),
        ]))
        min_pt, max_pt = poly.bounding_box()
        assert min_pt.x == 1.0 and min_pt.y == 1.0
        assert max_pt.x == 5.0 and max_pt.y == 6.0


class TestMesh3D:
    def test_triangle_mesh(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        mesh = Mesh3D(vertices, faces)
        assert len(mesh.vertices) == 3
        assert len(mesh.faces) == 1
    
    def test_compute_normals(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        mesh = Mesh3D(vertices, faces)
        mesh_with_normals = mesh.compute_normals()
        
        assert mesh_with_normals.normals is not None
        assert len(mesh_with_normals.normals) == 3
        # Check that normals are unit vectors
        for normal in mesh_with_normals.normals:
            length = np.linalg.norm(normal)
            assert abs(length - 1.0) < 1e-10

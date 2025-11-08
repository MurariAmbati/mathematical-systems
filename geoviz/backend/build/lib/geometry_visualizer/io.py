"""
Import/export functionality for various geometry file formats.

Supports: JSON (scene format), OBJ, STL, SVG
"""

from __future__ import annotations

from typing import List, TextIO
import json
import numpy as np

from geometry_visualizer.primitives import Mesh3D, Polygon2D, Point2D, Point3D
from geometry_visualizer.scene import Scene


def export_obj(mesh: Mesh3D, filename: str) -> None:
    """
    Export mesh to Wavefront OBJ format.
    
    Args:
        mesh: Mesh3D object to export
        filename: Output file path
    
    Example:
        >>> mesh = Mesh3D(vertices, faces)
        >>> export_obj(mesh, "model.obj")
    """
    with open(filename, 'w') as f:
        f.write("# Exported from geometry-visualizer\n")
        f.write(f"# {len(mesh.vertices)} vertices, {len(mesh.faces)} faces\n\n")
        
        # Write vertices
        for v in mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        f.write("\n")
        
        # Write vertex normals if available
        if mesh.normals is not None:
            for n in mesh.normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
            f.write("\n")
        
        # Write faces (OBJ indices start at 1)
        for face in mesh.faces:
            if mesh.normals is not None:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def import_obj(filename: str) -> Mesh3D:
    """
    Import mesh from Wavefront OBJ format.
    
    Args:
        filename: Input file path
    
    Returns:
        Mesh3D object
    
    Note:
        Only supports triangulated meshes. Quads and n-gons will need triangulation.
    """
    vertices: List[List[float]] = []
    normals: List[List[float]] = []
    faces: List[List[int]] = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                # Vertex
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            elif parts[0] == 'vn':
                # Vertex normal
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            elif parts[0] == 'f':
                # Face (indices start at 1 in OBJ)
                face_vertices = []
                for i in range(1, len(parts)):
                    # Handle f v, f v/vt, f v/vt/vn, f v//vn formats
                    vertex_data = parts[i].split('/')
                    vertex_idx = int(vertex_data[0]) - 1
                    face_vertices.append(vertex_idx)
                
                if len(face_vertices) == 3:
                    faces.append(face_vertices)
                elif len(face_vertices) == 4:
                    # Triangulate quad
                    faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
                    faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])
    
    vertices_array = np.array(vertices, dtype=np.float64)
    faces_array = np.array(faces, dtype=np.int32)
    normals_array = np.array(normals, dtype=np.float64) if normals else None
    
    return Mesh3D(vertices_array, faces_array, normals_array)


def export_stl(mesh: Mesh3D, filename: str, binary: bool = False) -> None:
    """
    Export mesh to STL format.
    
    Args:
        mesh: Mesh3D object to export
        filename: Output file path
        binary: If True, write binary STL; otherwise ASCII STL
    """
    if binary:
        _export_stl_binary(mesh, filename)
    else:
        _export_stl_ascii(mesh, filename)


def _export_stl_ascii(mesh: Mesh3D, filename: str) -> None:
    """Export mesh to ASCII STL format."""
    with open(filename, 'w') as f:
        f.write("solid geometry\n")
        
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            
            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid geometry\n")


def _export_stl_binary(mesh: Mesh3D, filename: str) -> None:
    """Export mesh to binary STL format."""
    import struct
    
    with open(filename, 'wb') as f:
        # Header (80 bytes)
        header = b'Binary STL from geometry-visualizer'
        header += b' ' * (80 - len(header))
        f.write(header)
        
        # Number of triangles
        f.write(struct.pack('<I', len(mesh.faces)))
        
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            
            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            
            # Write normal
            f.write(struct.pack('<fff', normal[0], normal[1], normal[2]))
            
            # Write vertices
            f.write(struct.pack('<fff', v0[0], v0[1], v0[2]))
            f.write(struct.pack('<fff', v1[0], v1[1], v1[2]))
            f.write(struct.pack('<fff', v2[0], v2[1], v2[2]))
            
            # Attribute byte count (unused)
            f.write(struct.pack('<H', 0))


def export_svg(polygons: List[Polygon2D], filename: str, 
               width: int = 800, height: int = 600,
               padding: float = 20.0) -> None:
    """
    Export 2D polygons to SVG format.
    
    Args:
        polygons: List of Polygon2D objects
        filename: Output file path
        width: SVG viewport width
        height: SVG viewport height
        padding: Padding around geometry
    """
    if not polygons:
        raise ValueError("No polygons to export")
    
    # Compute bounding box
    all_points = []
    for poly in polygons:
        all_points.extend(poly.vertices)
    
    min_x = min(p.x for p in all_points)
    max_x = max(p.x for p in all_points)
    min_y = min(p.y for p in all_points)
    max_y = max(p.y for p in all_points)
    
    # Compute scale to fit in viewport with padding
    data_width = max_x - min_x
    data_height = max_y - min_y
    
    scale_x = (width - 2 * padding) / data_width if data_width > 0 else 1
    scale_y = (height - 2 * padding) / data_height if data_height > 0 else 1
    scale = min(scale_x, scale_y)
    
    def transform_point(p: Point2D) -> tuple[float, float]:
        # Transform to SVG coordinates (y-axis flipped)
        x = (p.x - min_x) * scale + padding
        y = height - ((p.y - min_y) * scale + padding)
        return x, y
    
    with open(filename, 'w') as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" ')
        f.write(f'width="{width}" height="{height}" ')
        f.write(f'viewBox="0 0 {width} {height}">\n')
        f.write(f'  <rect width="100%" height="100%" fill="white"/>\n')
        
        # Draw each polygon
        for i, poly in enumerate(polygons):
            # Draw exterior
            points_str = ' '.join(f'{x},{y}' for x, y in 
                                (transform_point(p) for p in poly.vertices))
            f.write(f'  <polygon points="{points_str}" ')
            f.write(f'fill="lightblue" stroke="black" stroke-width="1.5" ')
            f.write(f'fill-opacity="0.7"/>\n')
            
            # Draw holes
            for hole in poly.holes:
                hole_points_str = ' '.join(f'{x},{y}' for x, y in 
                                          (transform_point(p) for p in hole))
                f.write(f'  <polygon points="{hole_points_str}" ')
                f.write(f'fill="white" stroke="black" stroke-width="1"/>\n')
        
        f.write('</svg>\n')


def export_scene_json(scene: Scene, filename: str) -> None:
    """
    Export scene to JSON format.
    
    This is a convenience wrapper around Scene.export_json().
    
    Args:
        scene: Scene object
        filename: Output file path
    """
    scene.export_json(filename)


def import_scene_json(filename: str) -> Scene:
    """
    Import scene from JSON format.
    
    This is a convenience wrapper around Scene.import_json().
    
    Args:
        filename: Input file path
    
    Returns:
        Scene object
    """
    return Scene.import_json(filename)

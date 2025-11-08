"""
Affine transformations for 2D and 3D geometry.

All transformation objects are immutable and can be composed via multiplication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union
import numpy as np
import numpy.typing as npt

from geometry_visualizer.primitives import Point2D, Point3D, Vector2D, Vector3D


@dataclass(frozen=True)
class Transform2D:
    """
    Immutable 2D affine transformation represented as 3x3 matrix.
    
    Matrix form:
        | a  b  tx |
        | c  d  ty |
        | 0  0  1  |
    
    Attributes:
        matrix: 3x3 transformation matrix
    """
    matrix: npt.NDArray[np.float64]
    
    def __post_init__(self) -> None:
        """Validate transformation matrix."""
        if self.matrix.shape != (3, 3):
            raise ValueError("Transform2D matrix must be 3x3")
        # Ensure last row is [0, 0, 1]
        if not np.allclose(self.matrix[2, :], [0, 0, 1]):
            mat = self.matrix.copy()
            mat[2, :] = [0, 0, 1]
            object.__setattr__(self, 'matrix', mat)
    
    @classmethod
    def identity(cls) -> Transform2D:
        """Create identity transformation."""
        return cls(np.eye(3, dtype=np.float64))
    
    @classmethod
    def translation(cls, tx: float, ty: float) -> Transform2D:
        """
        Create translation transformation.
        
        Args:
            tx: Translation in x direction
            ty: Translation in y direction
        """
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 2] = tx
        matrix[1, 2] = ty
        return cls(matrix)
    
    @classmethod
    def rotation(cls, angle: float, center: Point2D = Point2D(0, 0)) -> Transform2D:
        """
        Create rotation transformation.
        
        Args:
            angle: Rotation angle in radians (counter-clockwise)
            center: Center of rotation (default: origin)
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Rotate around center: translate to origin, rotate, translate back
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        
        if center.x != 0 or center.y != 0:
            # T * R * T^-1
            t1 = cls.translation(-center.x, -center.y)
            r = cls(matrix)
            t2 = cls.translation(center.x, center.y)
            return t2.compose(r).compose(t1)
        
        return cls(matrix)
    
    @classmethod
    def scaling(cls, sx: float, sy: float, center: Point2D = Point2D(0, 0)) -> Transform2D:
        """
        Create scaling transformation.
        
        Args:
            sx: Scale factor in x direction
            sy: Scale factor in y direction
            center: Center of scaling (default: origin)
        """
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        
        if center.x != 0 or center.y != 0:
            # T * S * T^-1
            t1 = cls.translation(-center.x, -center.y)
            s = cls(matrix)
            t2 = cls.translation(center.x, center.y)
            return t2.compose(s).compose(t1)
        
        return cls(matrix)
    
    @classmethod
    def shear(cls, shx: float, shy: float) -> Transform2D:
        """
        Create shear transformation.
        
        Args:
            shx: Shear factor in x direction
            shy: Shear factor in y direction
        """
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 1] = shx
        matrix[1, 0] = shy
        return cls(matrix)
    
    def compose(self, other: Transform2D) -> Transform2D:
        """
        Compose with another transformation (self applied after other).
        
        Args:
            other: Transformation to apply first
        
        Returns:
            Combined transformation
        """
        return Transform2D(self.matrix @ other.matrix)
    
    def inverse(self) -> Transform2D:
        """
        Compute inverse transformation.
        
        Returns:
            Inverse transformation
        
        Raises:
            ValueError: If transformation is not invertible
        """
        try:
            inv_matrix = np.linalg.inv(self.matrix)
            return Transform2D(inv_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Transformation is not invertible")
    
    def apply_to_point(self, point: Point2D) -> Point2D:
        """Apply transformation to a point."""
        vec = np.array([point.x, point.y, 1.0])
        result = self.matrix @ vec
        return Point2D(result[0], result[1])
    
    def apply_to_vector(self, vector: Vector2D) -> Vector2D:
        """Apply transformation to a vector (ignoring translation)."""
        vec = np.array([vector.x, vector.y, 0.0])
        result = self.matrix @ vec
        return Vector2D(result[0], result[1])
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"matrix": self.matrix.tolist()}
    
    @classmethod
    def from_dict(cls, data: dict) -> Transform2D:
        """Deserialize from dictionary."""
        matrix = np.array(data["matrix"], dtype=np.float64)
        return cls(matrix)


@dataclass(frozen=True)
class Transform3D:
    """
    Immutable 3D affine transformation represented as 4x4 matrix.
    
    Matrix form:
        | r00 r01 r02 tx |
        | r10 r11 r12 ty |
        | r20 r21 r22 tz |
        | 0   0   0   1  |
    
    Attributes:
        matrix: 4x4 transformation matrix
    """
    matrix: npt.NDArray[np.float64]
    
    def __post_init__(self) -> None:
        """Validate transformation matrix."""
        if self.matrix.shape != (4, 4):
            raise ValueError("Transform3D matrix must be 4x4")
        # Ensure last row is [0, 0, 0, 1]
        if not np.allclose(self.matrix[3, :], [0, 0, 0, 1]):
            mat = self.matrix.copy()
            mat[3, :] = [0, 0, 0, 1]
            object.__setattr__(self, 'matrix', mat)
    
    @classmethod
    def identity(cls) -> Transform3D:
        """Create identity transformation."""
        return cls(np.eye(4, dtype=np.float64))
    
    @classmethod
    def translation(cls, tx: float, ty: float, tz: float) -> Transform3D:
        """
        Create translation transformation.
        
        Args:
            tx, ty, tz: Translation in x, y, z directions
        """
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = tx
        matrix[1, 3] = ty
        matrix[2, 3] = tz
        return cls(matrix)
    
    @classmethod
    def rotation_x(cls, angle: float) -> Transform3D:
        """
        Create rotation around X axis.
        
        Args:
            angle: Rotation angle in radians
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix = np.eye(4, dtype=np.float64)
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_y(cls, angle: float) -> Transform3D:
        """
        Create rotation around Y axis.
        
        Args:
            angle: Rotation angle in radians
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_z(cls, angle: float) -> Transform3D:
        """
        Create rotation around Z axis.
        
        Args:
            angle: Rotation angle in radians
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_axis(cls, axis: Vector3D, angle: float) -> Transform3D:
        """
        Create rotation around arbitrary axis (Rodrigues' formula).
        
        Args:
            axis: Rotation axis (will be normalized)
            angle: Rotation angle in radians
        """
        axis = axis.normalize()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        one_minus_cos = 1.0 - cos_a
        
        x, y, z = axis.x, axis.y, axis.z
        
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = cos_a + x * x * one_minus_cos
        matrix[0, 1] = x * y * one_minus_cos - z * sin_a
        matrix[0, 2] = x * z * one_minus_cos + y * sin_a
        matrix[1, 0] = y * x * one_minus_cos + z * sin_a
        matrix[1, 1] = cos_a + y * y * one_minus_cos
        matrix[1, 2] = y * z * one_minus_cos - x * sin_a
        matrix[2, 0] = z * x * one_minus_cos - y * sin_a
        matrix[2, 1] = z * y * one_minus_cos + x * sin_a
        matrix[2, 2] = cos_a + z * z * one_minus_cos
        
        return cls(matrix)
    
    @classmethod
    def scaling(cls, sx: float, sy: float, sz: float) -> Transform3D:
        """
        Create scaling transformation.
        
        Args:
            sx, sy, sz: Scale factors in x, y, z directions
        """
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        matrix[2, 2] = sz
        return cls(matrix)
    
    def compose(self, other: Transform3D) -> Transform3D:
        """
        Compose with another transformation (self applied after other).
        
        Args:
            other: Transformation to apply first
        
        Returns:
            Combined transformation
        """
        return Transform3D(self.matrix @ other.matrix)
    
    def inverse(self) -> Transform3D:
        """
        Compute inverse transformation.
        
        Returns:
            Inverse transformation
        
        Raises:
            ValueError: If transformation is not invertible
        """
        try:
            inv_matrix = np.linalg.inv(self.matrix)
            return Transform3D(inv_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Transformation is not invertible")
    
    def apply_to_point(self, point: Point3D) -> Point3D:
        """Apply transformation to a point."""
        vec = np.array([point.x, point.y, point.z, 1.0])
        result = self.matrix @ vec
        return Point3D(result[0], result[1], result[2])
    
    def apply_to_vector(self, vector: Vector3D) -> Vector3D:
        """Apply transformation to a vector (ignoring translation)."""
        vec = np.array([vector.x, vector.y, vector.z, 0.0])
        result = self.matrix @ vec
        return Vector3D(result[0], result[1], result[2])
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"matrix": self.matrix.tolist()}
    
    @classmethod
    def from_dict(cls, data: dict) -> Transform3D:
        """Deserialize from dictionary."""
        matrix = np.array(data["matrix"], dtype=np.float64)
        return cls(matrix)

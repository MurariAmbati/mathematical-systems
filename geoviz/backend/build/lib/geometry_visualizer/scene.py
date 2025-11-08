"""
Scene data model for geometric visualization.

Provides a structured scene representation with objects, camera, and render options,
serializable to/from JSON according to the scene schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

from geometry_visualizer.primitives import (
    Point2D, Point3D, Polygon2D, Mesh3D
)
from geometry_visualizer.transforms import Transform2D, Transform3D


@dataclass
class SceneObject:
    """
    A single geometric object in the scene.
    
    Attributes:
        id: Unique identifier for this object
        type: Object type (e.g., 'polygon', 'mesh', 'point', 'line')
        geometry: The geometry data (dict or primitive object)
        style: Visual styling properties (colors, line width, etc.)
        transform: Optional transformation applied to the object
        properties: Additional custom properties
        visible: Whether object should be rendered
    """
    id: str
    type: str
    geometry: Union[Dict[str, Any], Point2D, Point3D, Polygon2D, Mesh3D]
    style: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[Union[Transform2D, Transform3D, Dict[str, Any]]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "style": self.style,
            "properties": self.properties,
            "visible": self.visible,
        }
        
        # Serialize geometry
        if hasattr(self.geometry, 'to_dict'):
            result["geometry"] = self.geometry.to_dict()
        else:
            result["geometry"] = self.geometry
        
        # Serialize transform
        if self.transform is not None:
            if hasattr(self.transform, 'to_dict'):
                result["transform"] = self.transform.to_dict()
            else:
                result["transform"] = self.transform
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneObject:
        """Deserialize from dictionary."""
        # For now, keep geometry and transform as dicts
        # In production, you'd deserialize to proper types based on 'type' field
        return cls(
            id=data["id"],
            type=data["type"],
            geometry=data["geometry"],
            style=data.get("style", {}),
            transform=data.get("transform"),
            properties=data.get("properties", {}),
            visible=data.get("visible", True),
        )


@dataclass
class Camera:
    """
    Camera configuration for scene viewing.
    
    Attributes:
        position: Camera position in 3D space
        target: Point the camera is looking at
        up: Up vector
        projection: Projection type ('perspective' or 'orthographic')
        fov: Field of view in degrees (for perspective)
        near: Near clipping plane
        far: Far clipping plane
        zoom: Zoom factor (for orthographic)
    """
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 10.0])
    target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    up: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    projection: str = "perspective"
    fov: float = 50.0
    near: float = 0.1
    far: float = 1000.0
    zoom: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "position": self.position,
            "target": self.target,
            "up": self.up,
            "projection": self.projection,
            "fov": self.fov,
            "near": self.near,
            "far": self.far,
            "zoom": self.zoom,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Camera:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class RenderOptions:
    """
    Rendering configuration for the scene.
    
    Attributes:
        grid: Whether to show grid
        axes: Whether to show coordinate axes
        lighting: Lighting preset ('default', 'ambient', 'studio')
        background: Background color
        antialias: Enable antialiasing
        shadows: Enable shadows
    """
    grid: bool = True
    axes: bool = True
    lighting: str = "default"
    background: str = "#f0f0f0"
    antialias: bool = True
    shadows: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "grid": self.grid,
            "axes": self.axes,
            "lighting": self.lighting,
            "background": self.background,
            "antialias": self.antialias,
            "shadows": self.shadows,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RenderOptions:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class SceneMetadata:
    """
    Metadata for the scene.
    
    Attributes:
        units: Units of measurement ('m', 'cm', 'mm', 'in', 'ft')
        coordinate_system: Coordinate system identifier
        timestamp: ISO timestamp of scene creation
        source: Source application or script
        version: Schema version
        description: Human-readable description
    """
    units: str = "m"
    coordinate_system: str = "cartesian"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "geometry-visualizer"
    version: str = "1.0"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "units": self.units,
            "coordinate_system": self.coordinate_system,
            "timestamp": self.timestamp,
            "source": self.source,
            "version": self.version,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneMetadata:
        """Deserialize from dictionary."""
        return cls(**data)


class Scene:
    """
    Complete scene containing geometric objects, camera, and render settings.
    
    The Scene class is the top-level container for all visualization data
    and can be serialized to/from JSON for interchange.
    
    Example:
        >>> scene = Scene()
        >>> scene.add_object(SceneObject(
        ...     id="poly1",
        ...     type="polygon",
        ...     geometry=Polygon2D([...]).to_dict()
        ... ))
        >>> scene.export_json("scene.json")
    """
    
    def __init__(
        self,
        metadata: Optional[SceneMetadata] = None,
        camera: Optional[Camera] = None,
        render_options: Optional[RenderOptions] = None,
    ):
        """
        Initialize a new scene.
        
        Args:
            metadata: Scene metadata (created with defaults if None)
            camera: Camera configuration (created with defaults if None)
            render_options: Render options (created with defaults if None)
        """
        self.metadata = metadata or SceneMetadata()
        self.camera = camera or Camera()
        self.render_options = render_options or RenderOptions()
        self.objects: List[SceneObject] = []
        self.animations: List[Dict[str, Any]] = []
    
    def add_object(
        self,
        obj: Optional[SceneObject] = None,
        *,
        id: Optional[str] = None,
        type: Optional[str] = None,
        geometry: Optional[Any] = None,
        **kwargs: Any
    ) -> SceneObject:
        """
        Add an object to the scene.
        
        Args:
            obj: SceneObject to add (if provided, other args ignored)
            id: Object ID (required if obj not provided)
            type: Object type (required if obj not provided)
            geometry: Geometry data (required if obj not provided)
            **kwargs: Additional SceneObject parameters (style, transform, etc.)
        
        Returns:
            The added SceneObject
        """
        if obj is None:
            if id is None or type is None or geometry is None:
                raise ValueError("Must provide either obj or (id, type, geometry)")
            obj = SceneObject(id=id, type=type, geometry=geometry, **kwargs)
        
        self.objects.append(obj)
        return obj
    
    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None
    
    def remove_object(self, object_id: str) -> bool:
        """
        Remove object by ID.
        
        Returns:
            True if object was found and removed, False otherwise
        """
        for i, obj in enumerate(self.objects):
            if obj.id == object_id:
                del self.objects[i]
                return True
        return False
    
    def clear_objects(self) -> None:
        """Remove all objects from the scene."""
        self.objects.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize scene to dictionary.
        
        Returns:
            Dictionary representation conforming to scene schema
        """
        return {
            "metadata": self.metadata.to_dict(),
            "camera": self.camera.to_dict(),
            "renderOptions": self.render_options.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
            "animations": self.animations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scene:
        """
        Deserialize scene from dictionary.
        
        Args:
            data: Dictionary representation of scene
        
        Returns:
            Scene object
        """
        scene = cls(
            metadata=SceneMetadata.from_dict(data.get("metadata", {})),
            camera=Camera.from_dict(data.get("camera", {})),
            render_options=RenderOptions.from_dict(data.get("renderOptions", {})),
        )
        
        for obj_data in data.get("objects", []):
            scene.add_object(SceneObject.from_dict(obj_data))
        
        scene.animations = data.get("animations", [])
        
        return scene
    
    def export_json(self, filename: str, indent: int = 2) -> None:
        """
        Export scene to JSON file.
        
        Args:
            filename: Output file path
            indent: JSON indentation level
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    @classmethod
    def import_json(cls, filename: str) -> Scene:
        """
        Import scene from JSON file.
        
        Args:
            filename: Input file path
        
        Returns:
            Scene object
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json_string(self, indent: int = 2) -> str:
        """
        Serialize scene to JSON string.
        
        Args:
            indent: JSON indentation level
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json_string(cls, json_str: str) -> Scene:
        """
        Deserialize scene from JSON string.
        
        Args:
            json_str: JSON string representation
        
        Returns:
            Scene object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

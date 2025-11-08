/**
 * TypeScript types for scene data matching the JSON schema.
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface Polygon2D {
  vertices: Point2D[];
  holes?: Point2D[][];
}

export interface Mesh3D {
  vertices: [number, number, number][];
  faces: [number, number, number][];
  normals?: [number, number, number][];
}

export type GeometryData = Point2D | Point3D | Polygon2D | Mesh3D | Record<string, any>;

export interface Transform2D {
  matrix: [[number, number, number], [number, number, number], [number, number, number]];
}

export interface Transform3D {
  matrix: [
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number]
  ];
}

export type Transform = Transform2D | Transform3D;

export interface SceneObjectStyle {
  color?: string;
  stroke?: string;
  strokeWidth?: number;
  fill?: string;
  opacity?: number;
  wireframe?: boolean;
  pointSize?: number;
}

export interface SceneObject {
  id: string;
  type: string;
  geometry: GeometryData;
  style?: SceneObjectStyle;
  transform?: Transform;
  properties?: Record<string, any>;
  visible?: boolean;
}

export interface Camera {
  position?: [number, number, number];
  target?: [number, number, number];
  up?: [number, number, number];
  projection?: 'perspective' | 'orthographic';
  fov?: number;
  near?: number;
  far?: number;
  zoom?: number;
}

export interface RenderOptions {
  grid?: boolean;
  axes?: boolean;
  lighting?: 'default' | 'ambient' | 'studio' | 'night';
  background?: string;
  antialias?: boolean;
  shadows?: boolean;
}

export interface SceneMetadata {
  version: string;
  units?: string;
  coordinate_system?: string;
  timestamp?: string;
  source?: string;
  description?: string;
}

export interface Animation {
  objectId: string;
  property: string;
  keyframes: any[];
  duration: number;
  easing?: string;
}

export interface SceneJSON {
  metadata: SceneMetadata;
  camera?: Camera;
  renderOptions?: RenderOptions;
  objects: SceneObject[];
  animations?: Animation[];
}

/**
 * Algorithm visualization step
 */
export interface AlgorithmStep {
  description: string;
  current_hull?: Point2D[];
  active_point?: Point2D;
  removed_points?: Point2D[];
  sorted_points?: Point2D[];
  completed: boolean;
  [key: string]: any;
}

/**
 * Result type for async operations
 */
export interface Result<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

/**
 * SceneObjects - Renders all geometric objects in the scene
 */

import * as THREE from 'three'
import type { SceneObject, Mesh3D, Polygon2D, Point3D } from '@/types/scene'

interface SceneObjectsProps {
  objects: SceneObject[]
}

export function SceneObjects({ objects }: SceneObjectsProps) {
  return (
    <>
      {objects.map((obj) => {
        // Only hide if explicitly set to false
        if (obj.visible === false) return null
        return <GeometricObject key={obj.id} object={obj} />
      })}
    </>
  )
}

interface GeometricObjectProps {
  object: SceneObject
}

function GeometricObject({ object }: GeometricObjectProps) {
  const style = object.style || {}

  switch (object.type) {
    case 'mesh3d':
      return <Mesh3DObject geometry={object.geometry as Mesh3D} style={style} />

    case 'polygon2d':
      return <Polygon2DObject geometry={object.geometry as Polygon2D} style={style} />

    case 'point3d':
      return <Point3DObject geometry={object.geometry as Point3D} style={style} />

    case 'point2d': {
      const point = object.geometry as { x: number; y: number }
      return <Point3DObject geometry={{ x: point.x, y: point.y, z: 0 }} style={style} />
    }

    default:
      console.warn(`Unknown object type: ${object.type}`)
      return null
  }
}

function Mesh3DObject({ geometry, style }: { geometry: Mesh3D; style: any }) {
  const positions = new Float32Array(geometry.vertices.flat())
  const indices = new Uint32Array(geometry.faces.flat())

  const color = style.color || style.fill || '#3b82f6'
  const wireframe = style.wireframe || false
  const opacity = style.opacity ?? 1

  return (
    <mesh>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={geometry.vertices.length}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute attach="index" count={indices.length} array={indices} itemSize={1} />
      </bufferGeometry>
      <meshStandardMaterial
        color={color}
        wireframe={wireframe}
        transparent={opacity < 1}
        opacity={opacity}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

function Polygon2DObject({ geometry, style }: { geometry: Polygon2D; style: any }) {
  // Convert 2D polygon to 3D mesh (extrude slightly or keep flat)
  const shape = new THREE.Shape()

  if (geometry.vertices.length > 0) {
    const first = geometry.vertices[0]
    shape.moveTo(first.x, first.y)

    for (let i = 1; i < geometry.vertices.length; i++) {
      const v = geometry.vertices[i]
      shape.lineTo(v.x, v.y)
    }
    shape.closePath()
  }

  // Add holes
  if (geometry.holes) {
    for (const hole of geometry.holes) {
      const holePath = new THREE.Path()
      if (hole.length > 0) {
        holePath.moveTo(hole[0].x, hole[0].y)
        for (let i = 1; i < hole.length; i++) {
          holePath.lineTo(hole[i].x, hole[i].y)
        }
        holePath.closePath()
      }
      shape.holes.push(holePath)
    }
  }

  const color = style.fill || style.color || '#3b82f6'
  const stroke = style.stroke || '#000000'
  const opacity = style.opacity ?? 0.7

  return (
    <group>
      {/* Filled polygon */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <shapeGeometry args={[shape]} />
        <meshStandardMaterial
          color={color}
          transparent={opacity < 1}
          opacity={opacity}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Outline */}
      {geometry.vertices.length > 0 && (
        <lineSegments>
          <edgesGeometry
            attach="geometry"
            args={[new THREE.ShapeGeometry(shape).rotateX(-Math.PI / 2)]}
          />
          <lineBasicMaterial color={stroke} linewidth={style.strokeWidth || 2} />
        </lineSegments>
      )}
    </group>
  )
}

function Point3DObject({ geometry, style }: { geometry: Point3D; style: any }) {
  const color = style.color || '#ff0000'
  const size = style.pointSize || 0.1

  return (
    <mesh position={[geometry.x, geometry.y, geometry.z]}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial color={color} />
    </mesh>
  )
}

/**
 * ControlPanel - UI controls for scene manipulation
 */

import { useState } from 'react'
import type { SceneJSON } from '@/types/scene'
import { sceneController } from '@/controllers/SceneController'

interface ControlPanelProps {
  scene: SceneJSON | null
}

export function ControlPanel({ scene }: ControlPanelProps) {
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null)

  // Generate a torus (donut shape)
  const generateTorus = (majorRadius = 2, minorRadius = 0.5, majorSegments = 24, minorSegments = 12) => {
    const vertices: [number, number, number][] = []
    const faces: [number, number, number][] = []
    
    for (let i = 0; i < majorSegments; i++) {
      const theta = (i / majorSegments) * Math.PI * 2
      for (let j = 0; j < minorSegments; j++) {
        const phi = (j / minorSegments) * Math.PI * 2
        const x = (majorRadius + minorRadius * Math.cos(phi)) * Math.cos(theta)
        const y = minorRadius * Math.sin(phi)
        const z = (majorRadius + minorRadius * Math.cos(phi)) * Math.sin(theta)
        vertices.push([x, y, z])
      }
    }
    
    for (let i = 0; i < majorSegments; i++) {
      for (let j = 0; j < minorSegments; j++) {
        const a = i * minorSegments + j
        const b = i * minorSegments + ((j + 1) % minorSegments)
        const c = ((i + 1) % majorSegments) * minorSegments + j
        const d = ((i + 1) % majorSegments) * minorSegments + ((j + 1) % minorSegments)
        faces.push([a, b, d])
        faces.push([a, d, c])
      }
    }
    
    return { vertices, faces }
  }

  // Generate a sphere using UV sphere algorithm
  const generateSphere = (radius = 1, segments = 16, rings = 16) => {
    const vertices: [number, number, number][] = []
    const faces: [number, number, number][] = []
    
    for (let lat = 0; lat <= rings; lat++) {
      const theta = (lat * Math.PI) / rings
      const sinTheta = Math.sin(theta)
      const cosTheta = Math.cos(theta)
      
      for (let lon = 0; lon <= segments; lon++) {
        const phi = (lon * 2 * Math.PI) / segments
        const x = Math.cos(phi) * sinTheta
        const y = cosTheta
        const z = Math.sin(phi) * sinTheta
        vertices.push([radius * x, radius * y, radius * z])
      }
    }
    
    for (let lat = 0; lat < rings; lat++) {
      for (let lon = 0; lon < segments; lon++) {
        const first = lat * (segments + 1) + lon
        const second = first + segments + 1
        faces.push([first, second, first + 1])
        faces.push([second, second + 1, first + 1])
      }
    }
    
    return { vertices, faces }
  }

  // Generate a helix/spiral
  const generateHelix = (radius = 1, height = 4, turns = 3, segments = 100, thickness = 0.15) => {
    const vertices: [number, number, number][] = []
    const faces: [number, number, number][] = []
    const sides = 8
    
    for (let i = 0; i <= segments; i++) {
      const t = i / segments
      const angle = t * turns * Math.PI * 2
      const centerX = radius * Math.cos(angle)
      const centerY = t * height - height / 2
      const centerZ = radius * Math.sin(angle)
      
      for (let j = 0; j < sides; j++) {
        const sideAngle = (j / sides) * Math.PI * 2
        const offsetX = thickness * Math.cos(sideAngle) * Math.sin(angle)
        const offsetY = thickness * Math.sin(sideAngle)
        const offsetZ = -thickness * Math.cos(sideAngle) * Math.cos(angle)
        vertices.push([centerX + offsetX, centerY + offsetY, centerZ + offsetZ])
      }
    }
    
    for (let i = 0; i < segments; i++) {
      for (let j = 0; j < sides; j++) {
        const a = i * sides + j
        const b = i * sides + ((j + 1) % sides)
        const c = (i + 1) * sides + j
        const d = (i + 1) * sides + ((j + 1) % sides)
        faces.push([a, b, d])
        faces.push([a, d, c])
      }
    }
    
    return { vertices, faces }
  }

  // Generate a geodesic polyhedron pattern
  const generateGeodesicSphere = () => {
    const phi = (1 + Math.sqrt(5)) / 2 // golden ratio
    const vertices: [number, number, number][] = [
      [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
      [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
      [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ].map(([x, y, z]) => {
      const len = Math.sqrt(x * x + y * y + z * z)
      return [x / len * 1.5, y / len * 1.5, z / len * 1.5] as [number, number, number]
    })
    
    const faces: [number, number, number][] = [
      [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    
    return { vertices, faces }
  }

  const handleLoadCube = async () => {
    const exampleScene: SceneJSON = {
      metadata: {
        version: '1.0',
        description: 'Simple cube',
      },
      objects: [
        {
          id: 'cube1',
          type: 'mesh3d',
          geometry: {
            vertices: [
              [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ],
            faces: [
              [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
              [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
              [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
            ],
          },
          style: {
            color: '#3b82f6',
            wireframe: false,
          },
        },
      ],
    }
    await sceneController.loadScene(exampleScene)
  }

  const handleLoadTorus = async () => {
    const torusGeometry = generateTorus(2, 0.6, 32, 16)
    const scene: SceneJSON = {
      metadata: {
        version: '1.0',
        description: 'Torus (donut shape)',
      },
      objects: [
        {
          id: 'torus1',
          type: 'mesh3d',
          geometry: torusGeometry,
          style: {
            color: '#8b5cf6',
            wireframe: false,
          },
        },
      ],
    }
    await sceneController.loadScene(scene)
  }

  const handleLoadSpheres = async () => {
    const sphere1 = generateSphere(1, 20, 20)
    const sphere2 = generateSphere(0.7, 16, 16)
    const sphere3 = generateSphere(0.5, 16, 16)
    
    // Position spheres
    const sphere2Positioned = {
      vertices: sphere2.vertices.map(
        ([x, y, z]: [number, number, number]) => [x + 2, y + 0.5, z + 1] as [number, number, number]
      ),
      faces: sphere2.faces,
    }
    
    const sphere3Positioned = {
      vertices: sphere3.vertices.map(
        ([x, y, z]: [number, number, number]) => [x - 1.8, y - 0.3, z + 1.5] as [number, number, number]
      ),
      faces: sphere3.faces,
    }
    
    const scene: SceneJSON = {
      metadata: {
        version: '1.0',
        description: 'Multiple spheres pattern',
      },
      objects: [
        {
          id: 'sphere1',
          type: 'mesh3d',
          geometry: sphere1,
          style: { color: '#ef4444', wireframe: false },
        },
        {
          id: 'sphere2',
          type: 'mesh3d',
          geometry: sphere2Positioned,
          style: { color: '#10b981', wireframe: false },
        },
        {
          id: 'sphere3',
          type: 'mesh3d',
          geometry: sphere3Positioned,
          style: { color: '#f59e0b', wireframe: false },
        },
      ],
    }
    await sceneController.loadScene(scene)
  }

  const handleLoadHelix = async () => {
    const helixGeometry = generateHelix(1.5, 5, 4, 120, 0.2)
    const scene: SceneJSON = {
      metadata: {
        version: '1.0',
        description: 'DNA-like helix spiral',
      },
      objects: [
        {
          id: 'helix1',
          type: 'mesh3d',
          geometry: helixGeometry,
          style: {
            color: '#06b6d4',
            wireframe: false,
          },
        },
      ],
    }
    await sceneController.loadScene(scene)
  }

  const handleLoadGeodesic = async () => {
    const geodesicGeometry = generateGeodesicSphere()
    const scene: SceneJSON = {
      metadata: {
        version: '1.0',
        description: 'Geodesic polyhedron',
      },
      objects: [
        {
          id: 'geodesic1',
          type: 'mesh3d',
          geometry: geodesicGeometry,
          style: {
            color: '#ec4899',
            wireframe: true,
          },
        },
      ],
    }
    await sceneController.loadScene(scene)
  }

  const handleLoadComplex = async () => {
    const geodesic = generateGeodesicSphere()
    const geodesicScaled = {
      vertices: geodesic.vertices.map(
        ([x, y, z]: [number, number, number]) => [x * 2, y * 2, z * 2] as [number, number, number]
      ),
      faces: geodesic.faces,
    }
    
    const scene: SceneJSON = {
      metadata: {
        version: '1.0',
        description: 'Complex multi-shape scene',
      },
      objects: [
        {
          id: 'base_torus',
          type: 'mesh3d',
          geometry: generateTorus(2.5, 0.4, 36, 18),
          style: { color: '#6366f1', wireframe: false },
        },
        {
          id: 'inner_sphere',
          type: 'mesh3d',
          geometry: generateSphere(1.3, 24, 24),
          style: { color: '#f59e0b', wireframe: false, opacity: 0.8 },
        },
        {
          id: 'geodesic_wire',
          type: 'mesh3d',
          geometry: geodesicScaled,
          style: { color: '#10b981', wireframe: true },
        },
      ],
    }
    await sceneController.loadScene(scene)
  }

  return (
    <div
      style={{
        width: '300px',
        background: '#ffffff',
        borderLeft: '1px solid #e5e7eb',
        display: 'flex',
        flexDirection: 'column',
        padding: '1rem',
        overflow: 'auto',
      }}
    >
      <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem', fontWeight: 'bold' }}>
        Controls
      </h2>

      {/* Scene info */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
          Scene Info
        </h3>
        <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
          <p>Objects: {scene?.objects.length || 0}</p>
          <p>Version: {scene?.metadata.version || '1.0'}</p>
          {scene?.metadata.description && <p>Description: {scene.metadata.description}</p>}
        </div>
      </div>

      {/* Actions */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
          Load Examples
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <button
            onClick={handleLoadCube}
            style={{
              width: '100%',
              padding: '0.5rem',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Cube
          </button>
          <button
            onClick={handleLoadTorus}
            style={{
              width: '100%',
              padding: '0.5rem',
              background: '#8b5cf6',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Torus
          </button>
          <button
            onClick={handleLoadSpheres}
            style={{
              width: '100%',
              padding: '0.5rem',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Spheres
          </button>
          <button
            onClick={handleLoadHelix}
            style={{
              width: '100%',
              padding: '0.5rem',
              background: '#06b6d4',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Helix
          </button>
          <button
            onClick={handleLoadGeodesic}
            style={{
              width: '100%',
              padding: '0.5rem',
              background: '#ec4899',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Geodesic
          </button>
          <button
            onClick={handleLoadComplex}
            style={{
              width: '100%',
              padding: '0.5rem',
              background: '#6366f1',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Complex Scene
          </button>
          <button
            onClick={() => sceneController.createEmptyScene()}
            style={{
              width: '100%',
              padding: '0.5rem',
              background: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
              marginTop: '0.5rem',
            }}
          >
            Clear Scene
          </button>
        </div>
      </div>

      {/* Objects list */}
      <div>
        <h3 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
          Objects
        </h3>
        {scene && scene.objects && scene.objects.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {scene.objects.map((obj) => (
              <div
                key={obj.id}
                onClick={() => setSelectedObjectId(obj.id)}
                style={{
                  padding: '0.5rem',
                  background: selectedObjectId === obj.id ? '#dbeafe' : '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.25rem',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                }}
              >
                <div style={{ fontWeight: '500' }}>{obj.id}</div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                  Type: {obj.type}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: '0.75rem', color: '#6b7280' }}>No objects in scene</p>
        )}
      </div>
    </div>
  )
}

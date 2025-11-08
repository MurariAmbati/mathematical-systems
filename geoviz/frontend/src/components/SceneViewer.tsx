/**
 * SceneViewer - 3D visualization component using Three.js via React Three Fiber
 */

import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, PerspectiveCamera, OrthographicCamera } from '@react-three/drei'
import { SceneObjects } from './SceneObjects'
import type { SceneJSON } from '@/types/scene'

interface SceneViewerProps {
  scene: SceneJSON | null
}

export function SceneViewer({ scene }: SceneViewerProps) {
  // Provide default values if scene is null
  const camera = scene?.camera || {
    position: [0, 0, 10] as [number, number, number],
    target: [0, 0, 0] as [number, number, number],
    up: [0, 1, 0] as [number, number, number],
    projection: 'perspective' as const,
  }
  
  const renderOptions = scene?.renderOptions || {
    grid: true,
    axes: true,
    lighting: 'default' as const,
    background: '#f0f0f0',
    antialias: true,
  }
  
  const objects = scene?.objects || []
  const isPerspective = camera.projection !== 'orthographic'

  return (
    <div style={{ flex: 1, position: 'relative' }}>
      <Canvas
        style={{ background: renderOptions.background || '#f0f0f0' }}
        gl={{ antialias: renderOptions.antialias !== false }}
        shadows={renderOptions.shadows || false}
      >
        {/* Camera */}
        {isPerspective ? (
          <PerspectiveCamera
            makeDefault
            position={camera.position || [0, 0, 10]}
            fov={camera.fov || 50}
            near={camera.near || 0.1}
            far={camera.far || 1000}
          />
        ) : (
          <OrthographicCamera
            makeDefault
            position={camera.position || [0, 0, 10]}
            zoom={camera.zoom || 1}
            near={camera.near || 0.1}
            far={camera.far || 1000}
          />
        )}

        {/* Lighting */}
        {renderOptions.lighting === 'ambient' ? (
          <ambientLight intensity={1} />
        ) : renderOptions.lighting === 'studio' ? (
          <>
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={0.5} />
            <directionalLight position={[-10, -10, -5]} intensity={0.3} />
          </>
        ) : (
          <>
            <ambientLight intensity={0.5} />
            <directionalLight position={[5, 5, 5]} intensity={0.8} />
          </>
        )}

        {/* Grid and axes */}
        {renderOptions.grid !== false && (
          <Grid
            args={[20, 20]}
            cellSize={1}
            cellThickness={0.5}
            cellColor="#666"
            sectionSize={5}
            sectionThickness={1}
            sectionColor="#888"
            fadeDistance={30}
            fadeStrength={1}
            followCamera={false}
          />
        )}

        {/* Axes helper */}
        {renderOptions.axes !== false && <axesHelper args={[5]} />}

        {/* Scene objects */}
        <SceneObjects objects={objects} />

        {/* Controls */}
        <OrbitControls
          target={camera.target || [0, 0, 0]}
          enableDamping
          dampingFactor={0.05}
        />
      </Canvas>
    </div>
  )
}

import { useState, useEffect } from 'react'
import { SceneViewer } from './components/SceneViewer'
import { ControlPanel } from './components/ControlPanel'
import { sceneController } from './controllers/SceneController'
import type { SceneJSON } from './types/scene'

function App() {
  const [scene, setScene] = useState<SceneJSON | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Subscribe to scene changes
    const unsubscribe = sceneController.subscribe((newScene) => {
      setScene(newScene)
      setIsLoading(false)
    })

    // Create initial empty scene
    sceneController.createEmptyScene()
    
    // Fallback timeout in case scene doesn't load
    const timeout = setTimeout(() => {
      setIsLoading(false)
    }, 100)

    return () => {
      unsubscribe()
      clearTimeout(timeout)
    }
  }, [])

  if (isLoading) {
    return (
      <div style={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: '#f0f0f0'
      }}>
        <p>Loading Geometry Visualizer...</p>
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex' }}>
      <SceneViewer scene={scene} />
      <ControlPanel scene={scene} />
    </div>
  )
}

export default App

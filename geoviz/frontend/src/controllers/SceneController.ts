/**
 * SceneController - Main API for managing geometric scenes
 * 
 * Provides programmatic control over scene loading, object management,
 * camera control, and rendering.
 */

import type { SceneJSON, SceneObject, Camera, Result } from '@/types/scene';

export class SceneController {
  private scene: SceneJSON | null = null;
  private listeners: Set<(scene: SceneJSON) => void> = new Set();

  /**
   * Load a complete scene from JSON
   */
  async loadScene(json: SceneJSON): Promise<Result<void>> {
    try {
      // Validate scene structure
      if (!json.metadata || !json.objects) {
        return {
          success: false,
          error: {
            code: 'INVALID_SCENE',
            message: 'Scene must contain metadata and objects',
          },
        };
      }

      this.scene = json;
      this.notifyListeners();

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'LOAD_ERROR',
          message: error instanceof Error ? error.message : 'Unknown error',
        },
      };
    }
  }

  /**
   * Add an object to the current scene
   */
  addObject(objectSpec: SceneObject): Result<void> {
    if (!this.scene) {
      return {
        success: false,
        error: {
          code: 'NO_SCENE',
          message: 'No scene loaded',
        },
      };
    }

    // Check for duplicate ID
    if (this.scene.objects.some((obj) => obj.id === objectSpec.id)) {
      return {
        success: false,
        error: {
          code: 'DUPLICATE_ID',
          message: `Object with ID '${objectSpec.id}' already exists`,
        },
      };
    }

    this.scene.objects.push(objectSpec);
    this.notifyListeners();

    return { success: true };
  }

  /**
   * Remove an object from the scene
   */
  removeObject(objectId: string): Result<void> {
    if (!this.scene) {
      return {
        success: false,
        error: {
          code: 'NO_SCENE',
          message: 'No scene loaded',
        },
      };
    }

    const index = this.scene.objects.findIndex((obj) => obj.id === objectId);
    if (index === -1) {
      return {
        success: false,
        error: {
          code: 'NOT_FOUND',
          message: `Object with ID '${objectId}' not found`,
        },
      };
    }

    this.scene.objects.splice(index, 1);
    this.notifyListeners();

    return { success: true };
  }

  /**
   * Update an object's properties
   */
  updateObject(objectId: string, updates: Partial<SceneObject>): Result<void> {
    if (!this.scene) {
      return {
        success: false,
        error: {
          code: 'NO_SCENE',
          message: 'No scene loaded',
        },
      };
    }

    const object = this.scene.objects.find((obj) => obj.id === objectId);
    if (!object) {
      return {
        success: false,
        error: {
          code: 'NOT_FOUND',
          message: `Object with ID '${objectId}' not found`,
        },
      };
    }

    Object.assign(object, updates);
    this.notifyListeners();

    return { success: true };
  }

  /**
   * Get an object by ID
   */
  getObject(objectId: string): SceneObject | null {
    if (!this.scene) return null;
    return this.scene.objects.find((obj) => obj.id === objectId) || null;
  }

  /**
   * Set camera configuration
   */
  setCamera(
    preset: 'orthographic' | 'perspective',
    options?: Partial<Camera>
  ): Result<void> {
    if (!this.scene) {
      return {
        success: false,
        error: {
          code: 'NO_SCENE',
          message: 'No scene loaded',
        },
      };
    }

    this.scene.camera = {
      ...this.scene.camera,
      projection: preset,
      ...options,
    };
    this.notifyListeners();

    return { success: true };
  }

  /**
   * Get the current scene
   */
  getScene(): SceneJSON | null {
    return this.scene;
  }

  /**
   * Clear the entire scene
   */
  clear(): void {
    this.scene = null;
    this.notifyListeners();
  }

  /**
   * Subscribe to scene changes
   */
  subscribe(listener: (scene: SceneJSON) => void): () => void {
    this.listeners.add(listener);
    // Immediately call listener if scene exists
    if (this.scene) {
      listener(this.scene);
    }
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Notify all listeners of scene changes
   */
  private notifyListeners(): void {
    if (this.scene) {
      this.listeners.forEach((listener) => listener(this.scene!));
    }
  }

  /**
   * Create a new empty scene
   */
  createEmptyScene(): void {
    this.scene = {
      metadata: {
        version: '1.0',
        source: 'geometry-visualizer-frontend',
        timestamp: new Date().toISOString(),
      },
      objects: [],
      camera: {
        position: [0, 0, 10],
        target: [0, 0, 0],
        up: [0, 1, 0],
        projection: 'perspective',
      },
      renderOptions: {
        grid: true,
        axes: true,
        lighting: 'default',
        background: '#f0f0f0',
      },
    };
    this.notifyListeners();
  }
}

// Singleton instance
export const sceneController = new SceneController();

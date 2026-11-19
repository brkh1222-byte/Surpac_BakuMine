/**
 * `window.__bakumine`: a small handle for end-to-end tests and console debugging.
 * Installed in dev builds and in mock mode only.
 */
import type { BufferGeometry, Object3D, Scene, WebGLRenderer } from 'three'
import { useViewer } from '../state/store'
import { buildMockData } from './mockData'
import { getOrigin, type Vec3 } from './origin'
import { view, type CameraState, type StandardView } from './view'

export interface BufferReport {
  /** Object name (or type) owning the buffer. */
  name: string
  vertices: number
  /** Largest |coordinate| stored in the float32 position buffer. */
  maxAbs: number
}

export interface DebugHandle {
  store: typeof useViewer
  origin: () => Vec3
  camera: () => CameraState | null
  setView: (name: StandardView) => void
  zoomToExtents: () => void
  lookAt: (world: Vec3, distance: number) => void
  /** Frames rendered so far. */
  frames: () => number
  /** Renders one frame now. */
  render: () => void
  /** Position buffers of every object in the scene graph (layers and reference grid). */
  buffers: () => BufferReport[]
  /** Largest |coordinate| in any position buffer; must stay far below 1e5. */
  maxAbsBufferCoord: () => number
  /** Renders, reads the framebuffer back and counts distinct colours. */
  canvasStats: () => { width: number; height: number; distinctColors: number }
  /** Centres and sizes of the mock boxes, world coordinates. */
  mockBoxes: () => { centre: [number, number, number]; size: [number, number, number] }[]
}

declare global {
  interface Window {
    __bakumine?: DebugHandle
  }
}

export function installDebugHandle(gl: WebGLRenderer, scene: Scene): () => void {
  const render = () => {
    if (!view.camera) return
    gl.render(scene, view.camera)
    view.frames++
  }

  const buffers = (): BufferReport[] => {
    const out: BufferReport[] = []
    scene.traverse((object: Object3D) => {
      const geometry = (object as Object3D & { geometry?: BufferGeometry }).geometry
      const position = geometry?.getAttribute('position')
      if (!position) return
      let maxAbs = 0
      const array = position.array
      for (let i = 0; i < array.length; i++) {
        const a = Math.abs(array[i])
        if (a > maxAbs) maxAbs = a
      }
      out.push({ name: object.name || object.type, vertices: position.count, maxAbs })
    })
    return out
  }

  const handle: DebugHandle = {
    store: useViewer,
    origin: getOrigin,
    camera: () => view.rig?.getState() ?? null,
    setView: (name) => view.rig?.setView(name),
    zoomToExtents: () => view.rig?.zoomToExtents(),
    lookAt: (world, distance) => view.rig?.lookAt(world, distance),
    frames: () => view.frames,
    render,
    buffers,
    maxAbsBufferCoord: () => buffers().reduce((max, b) => Math.max(max, b.maxAbs), 0),
    canvasStats: () => {
      render()
      const context = gl.getContext()
      const width = context.drawingBufferWidth
      const height = context.drawingBufferHeight
      const pixels = new Uint8Array(width * height * 4)
      context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels)
      const colors = new Set(new Uint32Array(pixels.buffer))
      return { width, height, distinctColors: colors.size }
    },
    mockBoxes: () => buildMockData().boxes.map((b) => ({ centre: b.centre, size: b.size })),
  }
  window.__bakumine = handle
  return () => {
    if (window.__bakumine === handle) delete window.__bakumine
  }
}

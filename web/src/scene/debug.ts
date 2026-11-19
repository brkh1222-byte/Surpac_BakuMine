/**
 * `window.__bakumine`: a small handle for end-to-end tests and console debugging.
 * Installed in dev builds and in mock mode only.
 */
import { Vector3, type BufferGeometry, type Object3D, type Scene, type WebGLRenderer } from 'three'
import { useViewer } from '../state/store'
import { buildMockData } from './mockData'
import { getOrigin, worldToRender, type Vec3 } from './origin'
import { view, type CameraState, type StandardView } from './view'

export interface BufferReport {
  /** Object name (or type) owning the buffer. */
  name: string
  vertices: number
  /** Largest |coordinate| stored in the float32 position buffer. */
  maxAbs: number
}

export interface OrbitTiming {
  frames: number
  /** Frames the scene actually rendered during the orbit. */
  rendered: number
  intervalMeanMs: number
  intervalMedianMs: number
  intervalP95Ms: number
  intervalMaxMs: number
  renderMeanMs: number
  renderMaxMs: number
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
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
  /**
   * Orbits the camera one step per animation frame for `frames` frames and reports the
   * frame times. `interval*`: time between animation frames while the scene re-renders
   * on demand, as during a real orbit (bounded below by the display refresh).
   * `render*`: cost of one frame without the vsync wait (batches of ten renders, each
   * batch closed by a pixel read-back that waits for the GPU).
   */
  measureOrbit: (frames?: number, stepDegrees?: number) => Promise<OrbitTiming>
  /** Client (page) pixel position of a world point, or null when it is not in view. */
  worldToClient: (world: Vec3) => { x: number; y: number } | null
  /** WebGL renderer string and the draw calls / triangles of the last frame. */
  renderInfo: () => { renderer: string; calls: number; triangles: number; lines: number; points: number }
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
      // Fat lines keep their coordinates in `instanceStart` (interleaved start + end);
      // their `position` attribute is only the unit quad.
      const position = geometry?.getAttribute('instanceStart') ?? geometry?.getAttribute('position')
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
    measureOrbit: async (frames = 120, stepDegrees = 2) => {
      const nextFrame = () => new Promise<number>((resolve) => requestAnimationFrame(resolve))
      const step = (stepDegrees * Math.PI) / 180
      // Pass 1: let the on-demand loop render, as it does while the user orbits.
      const intervals: number[] = []
      const framesBefore = view.frames
      let last = await nextFrame()
      for (let i = 0; i < frames; i++) {
        view.rig?.orbitBy(step)
        const now = await nextFrame()
        intervals.push(now - last)
        last = now
      }
      const rendered = view.frames - framesBefore
      // Pass 2: batches of back-to-back renders closed by a one-pixel read-back, which
      // waits for the GPU to finish them: the frame cost without the vsync wait.
      const renders: number[] = []
      const context = gl.getContext()
      const pixel = new Uint8Array(4)
      const batch = 10
      for (let i = 0; i < 5; i++) {
        const start = performance.now()
        for (let k = 0; k < batch; k++) {
          view.rig?.orbitBy(step)
          render()
        }
        context.readPixels(0, 0, 1, 1, context.RGBA, context.UNSIGNED_BYTE, pixel)
        renders.push((performance.now() - start) / batch)
        await nextFrame()
      }
      intervals.sort((a, b) => a - b)
      renders.sort((a, b) => a - b)
      const mean = (list: number[]) => list.reduce((a, b) => a + b, 0) / Math.max(1, list.length)
      return {
        frames,
        rendered,
        intervalMeanMs: mean(intervals),
        intervalMedianMs: quantile(intervals, 0.5),
        intervalP95Ms: quantile(intervals, 0.95),
        intervalMaxMs: intervals[intervals.length - 1] ?? 0,
        renderMeanMs: mean(renders),
        renderMaxMs: renders[renders.length - 1] ?? 0,
      }
    },
    worldToClient: (world) => {
      const camera = view.camera
      if (!camera) return null
      const { verticalExaggeration } = useViewer.getState()
      const p = new Vector3(...worldToRender(world, verticalExaggeration)).project(camera)
      if (p.z < -1 || p.z > 1 || Math.abs(p.x) > 1 || Math.abs(p.y) > 1) return null
      const rect = gl.domElement.getBoundingClientRect()
      return { x: rect.left + ((p.x + 1) / 2) * rect.width, y: rect.top + ((1 - p.y) / 2) * rect.height }
    },
    renderInfo: () => {
      render()
      const context = gl.getContext()
      const extension = context.getExtension('WEBGL_debug_renderer_info')
      const renderer = extension ? String(context.getParameter(extension.UNMASKED_RENDERER_WEBGL)) : 'unknown'
      const { calls, triangles, lines, points } = gl.info.render
      return { renderer, calls, triangles, lines, points }
    },
    mockBoxes: () => buildMockData().boxes.map((b) => ({ centre: b.centre, size: b.size })),
  }
  window.__bakumine = handle
  return () => {
    if (window.__bakumine === handle) delete window.__bakumine
  }
}

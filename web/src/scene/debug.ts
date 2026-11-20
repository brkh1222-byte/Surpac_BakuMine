/**
 * `window.__bakumine`: a small handle for end-to-end tests and console debugging.
 * Installed in dev builds and in mock mode only.
 */
import { Vector3, type BufferGeometry, type Object3D, type Scene, type WebGLRenderer } from 'three'
import { outlineCount, outlinePoints } from '../section/outline'
import { outlineClient, type OutlineStats } from '../section/outlineClient'
import type { SnapTarget } from '../digitise/snap'
import { useBlocks } from '../state/blocks'
import { useDigitise } from '../state/digitise'
import { useSection } from '../state/section'
import { useViewer } from '../state/store'
import { SECTION_PLANES, sectionClipActive, updateSectionPlanes } from './clipping'
import { currentSnapTargets } from './DigitiseLayer'
import { buildMockData } from './mockData'
import { getOrigin, worldToRender, type Vec3 } from './origin'
import { view, type BlockLayerDebug, type CameraState, type StandardView } from './view'

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

export interface OutlineReport {
  /** One entry per polyline: closed flag and world points (flat xyz). */
  polylines: { closed: boolean; points: number[] }[]
}

export interface SectionStepTiming {
  steps: number
  /** Store change to the next presented frame, per step. */
  stepToFrameMs: number[]
  /** Store change until every outline of the new plane is on screen, per step. */
  stepToOutlineMs: number[]
  /** Synchronous main-thread time of the step itself (store update and React commit). */
  syncMs: number[]
  /** Longest stretch the main thread did not get back to an animation frame. */
  maxFrameGapMs: number
  /** Long tasks (> 50 ms) the browser reported during the run; -1 when unsupported. */
  longTasks: number
  outline: OutlineStats
}

export interface DebugHandle {
  store: typeof useViewer
  section: typeof useSection
  /** The slab planes in render space and every content material's plane count. */
  sectionClip: () => {
    active: boolean
    planes: { normal: [number, number, number]; constant: number }[]
    materials: { name: string; planes: number }[]
  }
  /** Mesh outlines on the section plane, by object id (world coordinates). */
  sectionOutlines: () => Record<string, OutlineReport>
  /** Ids of the holes inside the slab, by drillhole object id. */
  sectionHoles: () => Record<string, string[]>
  outlineStats: () => OutlineStats & { busy: boolean }
  /** Steps the section `steps` times forward and reports how long each step took. */
  measureSectionSteps: (steps?: number) => Promise<SectionStepTiming>
  /** Bounding box (client pixels) and count of the canvas pixels within `tolerance` of a colour. */
  pixelBounds: (hex: string, tolerance?: number) => { minX: number; minY: number; maxX: number; maxY: number; count: number } | null
  /** Colour of the canvas pixel at a client position, "#rrggbb". */
  pixelAt: (clientX: number, clientY: number) => string
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
  /** Renders and counts the canvas pixels within `tolerance` (0..255 per channel) of a colour. */
  countPixels: (hex: string, tolerance?: number) => number
  /** Client (page) pixel position of a world point, or null when it is not in view. */
  worldToClient: (world: Vec3) => { x: number; y: number } | null
  /** WebGL renderer string and the draw calls / triangles of the last frame. */
  renderInfo: () => { renderer: string; calls: number; triangles: number; lines: number; points: number }
  blockStore: typeof useBlocks
  /** Section digitising state, and the snap targets of the current section and slab. */
  digitise: typeof useDigitise
  snapTargets: () => SnapTarget[]
  /** What each blocks layer draws now (count, sizes, timings), by object id. */
  blocks: () => Record<string, Omit<BlockLayerDebug, 'block'>>
  /** Instance `k` of a blocks layer: row, world centroid, size, level. */
  blockAt: (objectId: string, k: number) => ReturnType<BlockLayerDebug['block']>
  /** Renders a frame and waits for the GPU to finish it (one-pixel read-back). */
  finishFrame: () => void
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
    if (view.renderFrame) view.renderFrame()
    else gl.render(scene, view.camera)
    view.frames++
  }

  /** Renders and reads the whole framebuffer back (rows bottom-up, RGBA). */
  const readBack = () => {
    render()
    const context = gl.getContext()
    const width = context.drawingBufferWidth
    const height = context.drawingBufferHeight
    const pixels = new Uint8Array(width * height * 4)
    context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels)
    return { width, height, pixels }
  }

  const buffers = (): BufferReport[] => {
    const out: BufferReport[] = []
    scene.traverse((object: Object3D) => {
      const geometry = (object as Object3D & { geometry?: BufferGeometry }).geometry
      // Fat lines keep their coordinates in `instanceStart` (interleaved start + end);
      // their `position` attribute is only the unit quad.
      // Blocks keep theirs in `aCentre` (one per instance); `position` is the unit cube.
      const position = geometry?.getAttribute('instanceStart') ?? geometry?.getAttribute('aCentre') ?? geometry?.getAttribute('position')
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
    section: useSection,
    sectionClip: () => {
      render()
      updateSectionPlanes()
      const materials: { name: string; planes: number }[] = []
      view.content?.traverse((object) => {
        const material = (object as Object3D & { material?: { clippingPlanes?: unknown[] | null } | unknown[] }).material
        if (!material || Array.isArray(material)) return
        materials.push({ name: object.name || object.type, planes: material.clippingPlanes?.length ?? 0 })
      })
      return {
        active: sectionClipActive(),
        planes: SECTION_PLANES.map((p) => ({ normal: [p.normal.x, p.normal.y, p.normal.z], constant: p.constant })),
        materials,
      }
    },
    sectionOutlines: () => {
      const out: Record<string, OutlineReport> = {}
      for (const [id, outline] of view.section.outlines) {
        out[id] = {
          polylines: Array.from({ length: outlineCount(outline) }, (_, k) => ({
            closed: outline.closed[k] === 1,
            points: Array.from(outlinePoints(outline, k)),
          })),
        }
      }
      return out
    },
    sectionHoles: () => Object.fromEntries(view.section.holes),
    outlineStats: () => ({ ...outlineClient.stats, busy: outlineClient.busy() }),
    measureSectionSteps: async (steps = 20) => {
      const nextFrame = () => new Promise<number>((resolve) => requestAnimationFrame(resolve))
      let longTasks = -1
      let observer: PerformanceObserver | null = null
      try {
        if (PerformanceObserver.supportedEntryTypes.includes('longtask')) {
          longTasks = 0
          observer = new PerformanceObserver((list) => {
            longTasks += list.getEntries().length
          })
          observer.observe({ entryTypes: ['longtask'] })
        }
      } catch {
        observer = null
      }
      const stepToFrameMs: number[] = []
      const stepToOutlineMs: number[] = []
      const syncMs: number[] = []
      let maxFrameGapMs = 0
      await nextFrame()
      for (let i = 0; i < steps; i++) {
        const start = performance.now()
        useSection.getState().stepBy(1)
        syncMs.push(performance.now() - start)
        let last = await nextFrame()
        stepToFrameMs.push(performance.now() - start)
        // Wait for the outlines of this plane, then one more frame to show them.
        while (outlineClient.busy()) {
          const now = await nextFrame()
          maxFrameGapMs = Math.max(maxFrameGapMs, now - last)
          last = now
        }
        const shown = await nextFrame()
        maxFrameGapMs = Math.max(maxFrameGapMs, shown - last)
        stepToOutlineMs.push(performance.now() - start)
      }
      observer?.disconnect()
      return { steps, stepToFrameMs, stepToOutlineMs, syncMs, maxFrameGapMs, longTasks, outline: { ...outlineClient.stats } }
    },
    pixelBounds: (hex, tolerance = 24) => {
      const { width, height, pixels } = readBack()
      const value = parseInt(hex.slice(1), 16)
      const r = (value >> 16) & 255
      const g = (value >> 8) & 255
      const b = value & 255
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity
      let count = 0
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4
          if (Math.abs(pixels[i] - r) > tolerance || Math.abs(pixels[i + 1] - g) > tolerance || Math.abs(pixels[i + 2] - b) > tolerance) continue
          count++
          if (x < minX) minX = x
          if (x > maxX) maxX = x
          if (y < minY) minY = y
          if (y > maxY) maxY = y
        }
      }
      if (count === 0) return null
      const rect = gl.domElement.getBoundingClientRect()
      const sx = rect.width / width
      const sy = rect.height / height
      // Framebuffer rows run bottom-up; pixel centres are at +0.5.
      return {
        minX: rect.left + (minX + 0.5) * sx,
        maxX: rect.left + (maxX + 0.5) * sx,
        minY: rect.top + (height - 1 - maxY + 0.5) * sy,
        maxY: rect.top + (height - 1 - minY + 0.5) * sy,
        count,
      }
    },
    pixelAt: (clientX, clientY) => {
      const { width, height, pixels } = readBack()
      const rect = gl.domElement.getBoundingClientRect()
      const x = Math.min(width - 1, Math.max(0, Math.floor(((clientX - rect.left) / rect.width) * width)))
      const y = Math.min(height - 1, Math.max(0, height - 1 - Math.floor(((clientY - rect.top) / rect.height) * height)))
      const i = (y * width + x) * 4
      return '#' + [pixels[i], pixels[i + 1], pixels[i + 2]].map((c) => c.toString(16).padStart(2, '0')).join('')
    },
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
    countPixels: (hex, tolerance = 24) => {
      render()
      const context = gl.getContext()
      const width = context.drawingBufferWidth
      const height = context.drawingBufferHeight
      const pixels = new Uint8Array(width * height * 4)
      context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels)
      const n = parseInt(hex.slice(1), 16)
      const r = (n >> 16) & 255
      const g = (n >> 8) & 255
      const b = n & 255
      let count = 0
      for (let i = 0; i < pixels.length; i += 4) {
        if (Math.abs(pixels[i] - r) <= tolerance && Math.abs(pixels[i + 1] - g) <= tolerance && Math.abs(pixels[i + 2] - b) <= tolerance) count++
      }
      return count
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
    blockStore: useBlocks,
    digitise: useDigitise,
    snapTargets: () => {
      const s = useSection.getState()
      return s.enabled ? currentSnapTargets(s.plane, s.front, s.back) : []
    },
    blocks: () =>
      Object.fromEntries(
        [...view.blocks].map(([id, b]) => {
          const { block: _block, ...rest } = b
          return [id, { ...rest, timing: { ...rest.timing } }]
        }),
      ),
    blockAt: (objectId, k) => view.blocks.get(objectId)?.block(k) ?? null,
    finishFrame: () => {
      render()
      const context = gl.getContext()
      context.readPixels(0, 0, 1, 1, context.RGBA, context.UNSIGNED_BYTE, new Uint8Array(4))
    },
    mockBoxes: () => buildMockData().boxes.map((b) => ({ centre: b.centre, size: b.size })),
  }
  window.__bakumine = handle
  return () => {
    if (window.__bakumine === handle) delete window.__bakumine
  }
}

/**
 * Per-frame view data shared between the camera rig, the reference grid and the DOM
 * overlays. It changes every frame while the camera moves, so it is a plain mutable
 * object rather than React state; anything the UI must re-render on goes to the store.
 */
import type { Group, OrthographicCamera, PerspectiveCamera } from 'three'
import { Vector3 } from 'three'
import type { Vec3 } from './origin'

export type StandardView = 'plan' | 'north' | 'east' | 'iso'

export interface CameraState {
  projection: 'perspective' | 'orthographic'
  /** Camera position, world coordinates (true Z). */
  position: [number, number, number]
  /** Orbit centre, world coordinates (true Z). */
  target: [number, number, number]
  /** Metres per CSS pixel at the orbit centre. */
  metresPerPixel: number
  near: number
  far: number
}

/** What `RigApi.sectionView` frames. Everything in render space. */
export interface SectionViewSpec {
  /** Point of the plane to put at the view centre. */
  centre: readonly [number, number, number]
  /** Unit vector from the plane towards the camera (the plane normal). */
  direction: readonly [number, number, number]
  /** Half extents of the rectangle to fit, metres on screen. */
  halfWidth: number
  halfHeight: number
}

/** Imperative camera commands; set by <CameraRig/> while it is mounted. */
export interface RigApi {
  setView: (view: StandardView) => void
  /** Frames all visible objects (or a default area when the scene is empty). */
  zoomToExtents: () => void
  /** Orbits around a world point from `distance` metres, keeping the view direction. */
  lookAt: (world: Vec3, distance: number) => void
  /** Turns the camera around the orbit centre about the vertical axis, radians. */
  orbitBy: (azimuth: number) => void
  /** Orthographic, square-on to a plane; false when the camera is not orthographic yet. */
  sectionView: (spec: SectionViewSpec) => boolean
  /** Locks rotation (pan and zoom only) or frees it. */
  setRotationLocked: (locked: boolean) => void
  /** Translates camera and orbit centre together, render space. */
  shiftBy: (delta: readonly [number, number, number]) => void
  getState: () => CameraState
}

/** The pointer ray handed to pickers, render space. */
export interface PointerRay {
  origin: Vector3
  /** Unit vector. */
  direction: Vector3
  /** Metres per CSS pixel at distance t along the ray: metresPerPixel0 + slope * t. */
  metresPerPixel0: number
  metresPerPixelSlope: number
}

export interface PickResult {
  /** Distance along the ray, render space. */
  distance: number
  /** Hit point, render space. */
  point: [number, number, number]
  /** Called when this result is what the pointer is over (client pixel position). */
  hover: (clientX: number, clientY: number) => void
  /** Called when the user clicks it. */
  select: () => void
}

/**
 * A layer's own picking routine. Layers that draw merged geometry register one here
 * (and set `userData.noRaycast` on their objects) instead of being raycast by three.
 */
export type Picker = (ray: PointerRay) => PickResult | null

export const view = {
  pickers: new Set<Picker>(),
  /** Metres per CSS pixel at the orbit centre (render space). */
  metresPerPixel: 1,
  /** Orbit centre, render space. */
  target: new Vector3(),
  camera: null as PerspectiveCamera | OrthographicCamera | null,
  /** The exaggerated group that holds every layer; used for picking and debug checks. */
  content: null as Group | null,
  rig: null as RigApi | null,
  /** Number of frames rendered so far. */
  frames: 0,
  /**
   * Draws one frame the way the render loop does. Set while something other than a
   * plain `gl.render(scene, camera)` is needed (the section ghost mode draws two passes).
   */
  renderFrame: null as (() => void) | null,
  /** What the section tools currently show, for tests and debugging. */
  section: {
    /** Mesh outlines on the section plane by object id (world coordinates). */
    outlines: new Map<string, import('../section/outline').Outline>(),
    /** Ids of the holes inside the slab, by drillhole object id. */
    holes: new Map<string, string[]>(),
  },
  /** World xyz of a hole's trace (flat triples, collar first), by drillhole object id. */
  holeTraces: new Map<string, (hole: number) => Float64Array>(),
  /** Drawn pieces of each drillhole object (section digitising snaps to them). */
  holePieces: new Map<string, import('../drillholes/pieces').DrillholePieces>(),
  /** By block model object id. */
  blocks: new Map<string, BlockLayerDebug>(),
}

/** What a blocks layer currently draws, for tests and the performance probe. */
export interface BlockLayerDebug {
  /** Blocks drawn (instances). */
  count: number
  selected: number
  total: number
  lod: 'full' | 'shell'
  clipped: boolean
  /** Attribute the instance values hold. */
  attribute: string | null
  /** Distinct block sizes among the instances (dx, dy, dz), largest first. */
  sizes: number[][]
  /** Largest |coordinate| in the float32 centre buffer. */
  maxAbsCentre: number
  bytesPerBlock: number
  /** ms: request sent -> decoded; buffers built; request sent -> draw issued / GPU done (-1 = not yet). */
  timing: { requestMs: number; buildMs: number; toDrawMs: number; toFrameMs: number }
  /** Instance k: its row in the block table, world centroid, size and level. */
  block: (k: number) => { row: number; centroid: [number, number, number]; size: [number, number, number]; level: number } | null
}

/** DOM nodes the scene updates directly each frame (no React re-render). */
export const overlay = {
  labels: null as HTMLDivElement | null,
  /** Host of the drillhole labels; each drillhole layer adds its own child. */
  holeLabels: null as HTMLDivElement | null,
  gizmo: null as SVGSVGElement | null,
  scaleBar: null as HTMLDivElement | null,
  scaleText: null as HTMLSpanElement | null,
}

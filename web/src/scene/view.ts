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

/** Imperative camera commands; set by <CameraRig/> while it is mounted. */
export interface RigApi {
  setView: (view: StandardView) => void
  /** Frames all visible objects (or a default area when the scene is empty). */
  zoomToExtents: () => void
  /** Orbits around a world point from `distance` metres, keeping the view direction. */
  lookAt: (world: Vec3, distance: number) => void
  getState: () => CameraState
}

export const view = {
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
}

/** DOM nodes the scene updates directly each frame (no React re-render). */
export const overlay = {
  labels: null as HTMLDivElement | null,
  gizmo: null as SVGSVGElement | null,
  scaleBar: null as HTMLDivElement | null,
  scaleText: null as HTMLSpanElement | null,
}

/**
 * Section state (zustand): the active cross-section and how it is shown.
 *
 * A section is a plane (world point + line azimuth + dip, see section/plane.ts) with a
 * slab around it (`front` towards the viewer, `back` away from it), a step size, and a
 * mode: `clip` hides everything outside the slab, `ghost` keeps the whole scene and
 * dims what is outside. `locked` is the section view: orthographic camera square-on to
 * the plane, rotation locked (scene/SectionScene.tsx follows this flag).
 *
 * Everything here is world coordinates and plain numbers; the scene derives its
 * render-space clipping planes from it (scene/clipping.ts).
 */
import { create } from 'zustand'
import type { Vec3 } from '../scene/origin'
import {
  boxInPlane,
  frameOf,
  normaliseAzimuth,
  sectionFromTwoPoints,
  sectionThroughHole,
  stepPlane,
  travelRange,
  type SectionPlane,
} from '../section/plane'
import { allBounds, boundsCentre, useViewer, type Bounds } from './store'

export type SectionMode = 'clip' | 'ghost'
export type SectionPreset = 'we' | 'sn' | 'plan'

export const DEFAULT_HALF_THICKNESS = 12.5
export const DEFAULT_STEP = 25
const MAX_THICKNESS = 1e5

/** The line being drawn in plan: `a` after the first click, `b` follows the pointer. */
export interface SectionDraft {
  a: Vec3 | null
  b: Vec3 | null
}

export interface SectionState {
  enabled: boolean
  plane: SectionPlane
  /** Slab thickness towards the viewer / away from the viewer, metres. */
  front: number
  back: number
  /** Distance of one step forward or back, metres. */
  step: number
  mode: SectionMode
  /** Section view: camera square-on, rotation locked. */
  locked: boolean
  /** Counts "look at section" requests, so that asking again re-fits the view. */
  viewRequest: number
  /** Non-null while the user is drawing the section line in plan view. */
  draft: SectionDraft | null

  setEnabled: (enabled: boolean) => void
  /** Changes centre, azimuth and/or dip. */
  setPlane: (patch: Partial<SectionPlane>) => void
  setThickness: (patch: { front?: number; back?: number }) => void
  setStep: (step: number) => void
  setMode: (mode: SectionMode) => void
  /**
   * W-E section at a northing, S-N section at an easting or plan at an elevation;
   * `coordinate` defaults to the middle of the loaded data. Turns the section on.
   */
  preset: (kind: SectionPreset, coordinate?: number) => void
  /** Moves the plane `count` steps forward (away from the viewer; negative = back). */
  stepBy: (count: number) => void
  /** Moves the plane to an absolute travel position (see `sectionTravel`). */
  setTravel: (position: number) => void
  /** Vertical section along the line a -> b (drawn in plan). Turns the section on. */
  fromTwoPoints: (a: Vec3, b: Vec3) => boolean
  /** Best-fit vertical section through a hole trace (flat world xyz). Turns it on. */
  throughHole: (xyz: ArrayLike<number>) => boolean
  /** Switches to the section view (orthographic, square-on, rotation locked). */
  lookAt: () => void
  setLocked: (locked: boolean) => void
  startDrawing: () => void
  cancelDrawing: () => void
  setDraft: (draft: SectionDraft) => void
}

function finite(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback
}

/** Middle of the loaded data, on whole metres (a section coordinate people can read). */
function dataCentre(): [number, number, number] {
  const bounds = allBounds(useViewer.getState())
  if (!bounds) return [0, 0, 0]
  const c = boundsCentre(bounds)
  return [Math.round(c[0]), Math.round(c[1]), Math.round(c[2])]
}

export const useSection = create<SectionState>()((set, get) => ({
  enabled: false,
  plane: { centre: [0, 0, 0], azimuth: 90, dip: 90 },
  front: DEFAULT_HALF_THICKNESS,
  back: DEFAULT_HALF_THICKNESS,
  step: DEFAULT_STEP,
  mode: 'clip',
  locked: false,
  viewRequest: 0,
  draft: null,

  setEnabled: (enabled) =>
    set((s) => {
      if (enabled === s.enabled) return s
      // First use: put the plane through the middle of the data.
      const untouched = s.plane.centre.every((c) => c === 0)
      const plane = enabled && untouched ? { ...s.plane, centre: dataCentre() } : s.plane
      return { enabled, plane, locked: enabled ? s.locked : false, draft: null }
    }),
  setPlane: (patch) =>
    set((s) => {
      const centre = patch.centre ?? s.plane.centre
      if (!centre.every(Number.isFinite)) return s
      const azimuth = normaliseAzimuth(finite(patch.azimuth ?? s.plane.azimuth, s.plane.azimuth))
      const dip = Math.min(90, Math.max(0, finite(patch.dip ?? s.plane.dip, s.plane.dip)))
      return { plane: { centre: [centre[0], centre[1], centre[2]], azimuth, dip } }
    }),
  setThickness: (patch) =>
    set((s) => {
      const clamp = (value: number | undefined, current: number) =>
        value === undefined || !Number.isFinite(value) ? current : Math.min(MAX_THICKNESS, Math.max(0, value))
      const front = clamp(patch.front, s.front)
      const back = clamp(patch.back, s.back)
      // A slab of zero thickness shows nothing.
      if (front + back <= 0) return s
      return { front, back }
    }),
  setStep: (step) => set((s) => (Number.isFinite(step) && step > 0 ? { step } : s)),
  setMode: (mode) => set({ mode }),
  preset: (kind, coordinate) =>
    set(() => {
      // Through the middle of the data on the axes the preset does not fix.
      const centre = dataCentre()
      const axis = kind === 'we' ? 1 : kind === 'sn' ? 0 : 2
      if (coordinate !== undefined && Number.isFinite(coordinate)) centre[axis] = coordinate
      const plane: SectionPlane =
        kind === 'we' ? { centre, azimuth: 90, dip: 90 } : kind === 'sn' ? { centre, azimuth: 0, dip: 90 } : { centre, azimuth: 90, dip: 0 }
      return { enabled: true, plane, draft: null }
    }),
  stepBy: (count) =>
    set((s) => (s.enabled && Number.isFinite(count) ? { plane: stepPlane(s.plane, count * s.step) } : s)),
  setTravel: (position) =>
    set((s) => (Number.isFinite(position) ? { plane: stepPlane(s.plane, position - sectionTravel(s.plane)) } : s)),
  fromTwoPoints: (a, b) => {
    const plane = sectionFromTwoPoints(a, b, dataCentre()[2])
    if (!plane) return false
    set({ enabled: true, plane, draft: null })
    return true
  },
  throughHole: (xyz) => {
    const plane = sectionThroughHole(xyz)
    if (!plane) return false
    set({ enabled: true, plane, draft: null })
    return true
  },
  lookAt: () => {
    if (!get().enabled) get().setEnabled(true)
    useViewer.getState().setProjection('orthographic')
    set((s) => ({ locked: true, viewRequest: s.viewRequest + 1, draft: null }))
  },
  setLocked: (locked) => set((s) => (s.locked === locked ? s : { locked })),
  startDrawing: () => set({ draft: { a: null, b: null }, locked: false }),
  cancelDrawing: () => set((s) => (s.draft ? { draft: null } : s)),
  setDraft: (draft) => set((s) => (s.draft ? { draft } : s)),
}))

/**
 * Position of the plane along its looking direction, metres: grows as the section steps
 * forward. For a W-E section looking north this is the northing; for a plan, minus the
 * elevation (forward is down).
 */
export function sectionTravel(plane: SectionPlane): number {
  const { n } = frameOf(plane)
  return -(plane.centre[0] * n[0] + plane.centre[1] * n[1] + plane.centre[2] * n[2])
}

/** Travel positions at which the plane enters and leaves a box (the slider range). */
export function sectionTravelRange(plane: SectionPlane, bounds: Bounds): [number, number] {
  const [near, far] = travelRange(plane, bounds)
  const here = sectionTravel(plane)
  return [here + near, here + far]
}

/** Rectangle of the plane (plane coordinates s, t) that covers a box, with a margin. */
export function sectionExtent(plane: SectionPlane, bounds: Bounds, margin = 0.04): { s: [number, number]; t: [number, number] } {
  const box = boxInPlane(plane, bounds)
  const ds = (box.max[0] - box.min[0]) * margin
  const dt = (box.max[1] - box.min[1]) * margin
  return { s: [box.min[0] - ds, box.max[0] + ds], t: [box.min[1] - dt, box.max[1] + dt] }
}

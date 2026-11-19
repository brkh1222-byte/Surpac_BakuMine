/**
 * Viewer state (zustand). UI components and scene layers both read from here.
 *
 * A layer never edits another layer's state: it registers the objects it draws
 * (`registerObject`, normally through `useLayerObject` in scene/layers.tsx) and reads
 * back whether they are visible. The sidebar, "zoom to extents", the clipping planes
 * and the grid elevation all work from that registry, so a new layer type needs no
 * changes here beyond, at most, a new `Category`.
 */
import { create } from 'zustand'
import { ensureOrigin, type Vec3 } from '../scene/origin'

export type Theme = 'dark' | 'light'
export type Projection = 'perspective' | 'orthographic'
export type Category = 'drillholes' | 'surfaces' | 'solids' | 'blockModels'

export const CATEGORIES: readonly { id: Category; label: string }[] = [
  { id: 'drillholes', label: 'Drillholes' },
  { id: 'surfaces', label: 'Surfaces' },
  { id: 'solids', label: 'Solids' },
  { id: 'blockModels', label: 'Block models' },
]

/** Axis-aligned box in WORLD coordinates (float64, true Z). */
export interface Bounds {
  min: Vec3
  max: Vec3
}

export interface SceneObject {
  id: string
  name: string
  category: Category
  bounds: Bounds
  visible: boolean
}

export interface CursorReadout {
  e: number
  n: number
  z: number
  /** What the cursor ray hit. */
  on: 'grid' | 'object'
}

export interface ViewerState {
  theme: Theme
  projection: Projection
  /** Z scale of the scene content. Read-outs always report true Z. */
  verticalExaggeration: number
  categoryVisible: Record<Category, boolean>
  objects: Record<string, SceneObject>
  cursor: CursorReadout | null
  /** Spacing of the labelled grid lines, metres (set by the grid as the view zooms). */
  gridSpacing: number

  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  setProjection: (projection: Projection) => void
  setVerticalExaggeration: (value: number) => void
  setCategoryVisible: (category: Category, visible: boolean) => void
  /** Adds or replaces an object. The first one fixes the scene origin at its bounds centre. */
  registerObject: (object: Omit<SceneObject, 'visible'> & { visible?: boolean }) => void
  removeObject: (id: string) => void
  setObjectVisible: (id: string, visible: boolean) => void
  setCursor: (cursor: CursorReadout | null) => void
  setGridSpacing: (spacing: number) => void
}

export const MIN_EXAGGERATION = 0.1
export const MAX_EXAGGERATION = 50

export function boundsCentre(b: Bounds): [number, number, number] {
  return [
    (b.min[0] + b.max[0]) / 2,
    (b.min[1] + b.max[1]) / 2,
    (b.min[2] + b.max[2]) / 2,
  ]
}

export function unionBounds(list: readonly Bounds[]): Bounds | null {
  if (list.length === 0) return null
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (const b of list) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], b.min[k])
      max[k] = Math.max(max[k], b.max[k])
    }
  }
  return { min: [min[0], min[1], min[2]], max: [max[0], max[1], max[2]] }
}

/** Bounds of a float64 xyz array (world coordinates). */
export function boundsOf(xyz: ArrayLike<number>): Bounds {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < xyz.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = xyz[i + k]
      if (v < min[k]) min[k] = v
      if (v > max[k]) max[k] = v
    }
  }
  return { min: [min[0], min[1], min[2]], max: [max[0], max[1], max[2]] }
}

export function isObjectVisible(state: ViewerState, id: string): boolean {
  const object = state.objects[id]
  return !!object && object.visible && state.categoryVisible[object.category]
}

/** World bounds of everything currently shown, or null when nothing is. */
export function visibleBounds(state: ViewerState): Bounds | null {
  return unionBounds(
    Object.values(state.objects)
      .filter((o) => o.visible && state.categoryVisible[o.category])
      .map((o) => o.bounds),
  )
}

/** World bounds of everything loaded, visible or not. */
export function allBounds(state: ViewerState): Bounds | null {
  return unionBounds(Object.values(state.objects).map((o) => o.bounds))
}

function initialTheme(): Theme {
  if (typeof window === 'undefined') return 'dark'
  const param = new URLSearchParams(window.location.search).get('theme')
  if (param === 'dark' || param === 'light') return param
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export const useViewer = create<ViewerState>()((set) => ({
  theme: initialTheme(),
  projection: 'perspective',
  verticalExaggeration: 1,
  categoryVisible: { drillholes: true, surfaces: true, solids: true, blockModels: true },
  objects: {},
  cursor: null,
  gridSpacing: 100,

  setTheme: (theme) => set({ theme }),
  toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
  setProjection: (projection) => set({ projection }),
  setVerticalExaggeration: (value) => {
    if (!Number.isFinite(value)) return
    set({ verticalExaggeration: Math.min(MAX_EXAGGERATION, Math.max(MIN_EXAGGERATION, value)) })
  },
  setCategoryVisible: (category, visible) =>
    set((s) => ({ categoryVisible: { ...s.categoryVisible, [category]: visible } })),
  registerObject: (object) => {
    ensureOrigin(boundsCentre(object.bounds))
    set((s) => ({
      objects: {
        ...s.objects,
        [object.id]: {
          ...object,
          visible: object.visible ?? s.objects[object.id]?.visible ?? true,
        },
      },
    }))
  },
  removeObject: (id) =>
    set((s) => {
      if (!(id in s.objects)) return s
      const objects = { ...s.objects }
      delete objects[id]
      return { objects }
    }),
  setObjectVisible: (id, visible) =>
    set((s) => {
      const object = s.objects[id]
      if (!object) return s
      return { objects: { ...s.objects, [id]: { ...object, visible } } }
    }),
  setCursor: (cursor) => set({ cursor }),
  setGridSpacing: (gridSpacing) => set((s) => (s.gridSpacing === gridSpacing ? s : { gridSpacing })),
}))

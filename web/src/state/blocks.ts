/**
 * Block model state (zustand): how each model is drawn, which blocks are asked for, what
 * the last request returned, and the picked block. The layer, the sidebar controls, the
 * filter panel, the status bar and the block panel all meet here.
 */
import { create } from 'zustand'
import type { BlockAttributeInfo, BlockDetail, BlockManifestObject, NamedGeometry } from '../api/blocks'
import { defaultFilter, filterToWhere, type BlockFilter } from '../blocks/filter'
import type { MissingDisplay } from '../blocks/colour'
import type { BlockDefinition } from '../blocks/structure'
import { DEFAULT_STEPS, roundNice, sanitiseRange, type NumericSettings } from '../colour/legend'

export interface BlockDisplay {
  name: string
  definition: BlockDefinition
  attributes: BlockAttributeInfo[]
  /** Solids and surfaces the filter may name. */
  solids: NamedGeometry[]
  surfaces: NamedGeometry[]
  /** Attribute the blocks are coloured by; null = one neutral colour. */
  colorBy: string | null
  /** Legend settings per numeric attribute. */
  settings: Record<string, NumericSettings>
  /** Blocks without a value of the colour attribute: left out, or drawn grey. */
  missing: MissingDisplay
  edges: boolean
  opacity: number
  /** Blocks are drawn at this fraction of their size (gaps between blocks below 1). */
  shrink: number
  filter: BlockFilter
  /** The filter as the `where` request parameter (constraint JSON); null = every block. */
  where: string | null
}

export type BlockLoadState = 'idle' | 'loading' | 'ready' | 'error'

/** Outcome of the latest block request of a model. */
export interface BlockStatus {
  state: BlockLoadState
  /** The server's message for an invalid filter, or a transport error. */
  error: string | null
  /** Blocks on screen (after hiding those without a value, the shell or the section slab). */
  shown: number
  /** Blocks the filter selects. */
  selected: number
  /**
   * Selected blocks that have a value of the colour attribute, when the others are hidden
   * (`present`: that attribute); all selected blocks otherwise.
   */
  valued: number
  present: string | null
  total: number
  /** Volume of the selected blocks, m3. */
  volume: number
  lod: 'full' | 'shell'
  clipped: boolean
}

/** The picked block. Geometry comes from the client; `detail` from the server. */
export interface BlockPick {
  objectId: string
  /** Row in the model's block table. */
  row: number
  blockId: string
  ijk: [number, number, number]
  level: number
  centroid: [number, number, number]
  size: [number, number, number]
  detail: BlockDetail | null
  error: string | null
}

export interface BlocksState {
  models: Record<string, BlockDisplay>
  status: Record<string, BlockStatus>
  pick: BlockPick | null

  register: (object: BlockManifestObject, definition: BlockDefinition) => void
  unregister: (id: string) => void
  setColorBy: (id: string, name: string | null) => void
  setNumericSettings: (id: string, name: string, patch: Partial<NumericSettings>) => void
  resetNumericSettings: (id: string, name: string) => void
  setDisplay: (id: string, patch: Partial<Pick<BlockDisplay, 'missing' | 'edges' | 'opacity' | 'shrink'>>) => void
  setFilter: (id: string, filter: BlockFilter) => void
  setStatus: (id: string, patch: Partial<BlockStatus>) => void
  setPick: (pick: BlockPick | null) => void
  patchPick: (objectId: string, row: number, patch: Partial<BlockPick>) => void
}

export const MIN_SHRINK = 0.5
export const MIN_OPACITY = 0.1

/** Default legend range of a numeric attribute: 0 (or a round low value) to the 98th percentile. */
export function defaultBlockSettings(attribute: BlockAttributeInfo): NumericSettings {
  const fallback: NumericSettings = { mode: 'continuous', min: 0, max: 1, steps: DEFAULT_STEPS }
  if (attribute.type !== 'numeric') return fallback
  const low = attribute.display_min ?? attribute.min
  const high = attribute.display_max ?? attribute.max
  if (low == null || high == null) return fallback
  const min = (attribute.min ?? low) >= 0 ? 0 : roundNice(low, 'down')
  const range = sanitiseRange(min, roundNice(high, 'up')) ?? sanitiseRange(min, min + 1)
  return range ? { ...fallback, ...range } : fallback
}

const EMPTY_STATUS: BlockStatus = { state: 'idle', error: null, shown: 0, selected: 0, valued: 0, present: null, total: 0, volume: 0, lod: 'full', clipped: false }

function clamp(value: number, low: number, high: number, fallback: number): number {
  return Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback
}

export const useBlocks = create<BlocksState>()((set) => ({
  models: {},
  status: {},
  pick: null,

  register: (object, definition) =>
    set((s) => {
      const previous = s.models[object.id]
      const attributes = object.attributes
      if (previous && previous.attributes.length === attributes.length && previous.attributes.every((a, i) => a.name === attributes[i].name && a.type === attributes[i].type)) {
        return { models: { ...s.models, [object.id]: { ...previous, attributes, definition, name: object.name } } }
      }
      const settings: Record<string, NumericSettings> = {}
      for (const a of attributes) if (a.type === 'numeric') settings[a.name] = defaultBlockSettings(a)
      const hints = (object as { display?: { color_by?: unknown; category_filter?: unknown } }).display ?? {}
      const wanted = typeof hints.color_by === 'string' ? hints.color_by : null
      const colorBy = attributes.find((a) => a.name === wanted)?.name ?? attributes.find((a) => a.type === 'numeric')?.name ?? attributes[0]?.name ?? null
      const filter = defaultFilter(attributes, hints.category_filter)
      const model: BlockDisplay = {
        name: object.name,
        definition,
        attributes,
        solids: object.constraint_geometry?.solids ?? [],
        surfaces: object.constraint_geometry?.surfaces ?? [],
        colorBy,
        settings,
        missing: 'hidden',
        edges: true,
        opacity: 1,
        shrink: 1,
        filter,
        where: filterToWhere(filter),
      }
      return { models: { ...s.models, [object.id]: model }, status: { ...s.status, [object.id]: { ...EMPTY_STATUS } } }
    }),
  unregister: (id) =>
    set((s) => {
      if (!(id in s.models)) return s
      const models = { ...s.models }
      const status = { ...s.status }
      delete models[id]
      delete status[id]
      return { models, status, pick: s.pick?.objectId === id ? null : s.pick }
    }),
  setColorBy: (id, name) =>
    set((s) => {
      const m = s.models[id]
      if (!m || (name !== null && !m.attributes.some((a) => a.name === name))) return s
      return { models: { ...s.models, [id]: { ...m, colorBy: name } } }
    }),
  setNumericSettings: (id, name, patch) =>
    set((s) => {
      const m = s.models[id]
      const current = m?.settings[name]
      if (!m || !current) return s
      const next = { ...current, ...patch }
      const range = sanitiseRange(next.min, next.max)
      if (!range) return s
      const steps = Math.min(12, Math.max(2, Math.round(next.steps) || DEFAULT_STEPS))
      return { models: { ...s.models, [id]: { ...m, settings: { ...m.settings, [name]: { ...next, ...range, steps } } } } }
    }),
  resetNumericSettings: (id, name) =>
    set((s) => {
      const m = s.models[id]
      const attribute = m?.attributes.find((a) => a.name === name)
      if (!m || !attribute) return s
      const mode = m.settings[name]?.mode ?? 'continuous'
      return { models: { ...s.models, [id]: { ...m, settings: { ...m.settings, [name]: { ...defaultBlockSettings(attribute), mode } } } } }
    }),
  setDisplay: (id, patch) =>
    set((s) => {
      const m = s.models[id]
      if (!m) return s
      const next = { ...m, ...patch }
      next.opacity = clamp(next.opacity, MIN_OPACITY, 1, m.opacity)
      next.shrink = clamp(next.shrink, MIN_SHRINK, 1, m.shrink)
      return { models: { ...s.models, [id]: next } }
    }),
  setFilter: (id, filter) =>
    set((s) => {
      const m = s.models[id]
      if (!m) return s
      return { models: { ...s.models, [id]: { ...m, filter, where: filterToWhere(filter) } } }
    }),
  setStatus: (id, patch) =>
    set((s) => ({ status: { ...s.status, [id]: { ...(s.status[id] ?? EMPTY_STATUS), ...patch } } })),
  setPick: (pick) => set((s) => (s.pick === null && pick === null ? s : { pick })),
  patchPick: (objectId, row, patch) =>
    set((s) => (s.pick && s.pick.objectId === objectId && s.pick.row === row ? { pick: { ...s.pick, ...patch } } : s)),
}))

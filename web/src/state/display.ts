/**
 * Display state (zustand): how each object is drawn, what the pointer is over, what is
 * selected, and which legends are showing. Layers write what they know (their
 * attributes, their legend, the hover and selection details); the sidebar, the legend
 * stack, the tooltip and the hole panel only read this store, so they work the same
 * for API data and for client-generated (mock) data.
 */
import { create } from 'zustand'
import type { AttributeInfo } from '../api/client'
import { DEFAULT_STEPS, roundNice, sanitiseRange, type LegendSpec, type NumericSettings } from '../colour/legend'

/** "table/name": identifies an attribute of a drillhole object. */
export function attributeKey(attribute: Pick<AttributeInfo, 'table' | 'name'>): string {
  return `${attribute.table}/${attribute.name}`
}

export interface DrillholeDisplay {
  attributes: AttributeInfo[]
  /** Key of the attribute the traces are coloured by; null = one neutral colour. */
  colorBy: string | null
  /** Legend settings per numeric attribute key. */
  settings: Record<string, NumericSettings>
  showLabels: boolean
}

export interface ValueReadout {
  name: string
  /** Formatted value with unit, or "no value". */
  text: string
}

/** What the pointer is over (an interval of a drillhole, or a stretch without one). */
export interface HoverInfo {
  objectId: string
  holeId: string
  hole: number
  /** Index of the piece under the pointer (drillholes/pieces.ts). */
  piece: number
  /** Null over a stretch of the hole that has no interval in the active table. */
  from: number | null
  to: number | null
  table: string | null
  values: ValueReadout[]
  /** Pointer position, client pixels. */
  x: number
  y: number
}

export interface HoleIntervalRow {
  from: number
  to: number
  values: string[]
  /** CSS colour of the interval as drawn; null without colouring. */
  color: string | null
}

/** The selected hole, as shown in the side panel. */
export interface HoleDetails {
  objectId: string
  hole: number
  holeId: string
  collar: [number, number, number]
  maxDepth: number
  table: string | null
  columns: string[]
  rows: HoleIntervalRow[]
}

/** One legend in the viewport. `onChange` makes its range and mode editable. */
export interface LegendEntry {
  spec: LegendSpec
  /** Object the legend belongs to, shown under the title. */
  subject: string
  onChange?: (patch: Partial<NumericSettings>) => void
  /** Resets the range to the default (robust data range). */
  onReset?: () => void
}

export interface DisplayState {
  drillholes: Record<string, DrillholeDisplay>
  hover: HoverInfo | null
  selection: HoleDetails | null
  legends: Record<string, LegendEntry>

  /** Declares a drillhole object's attributes; keeps existing choices on re-register. */
  registerDrillholes: (id: string, attributes: AttributeInfo[]) => void
  unregisterDrillholes: (id: string) => void
  setColorBy: (id: string, key: string | null) => void
  setNumericSettings: (id: string, key: string, patch: Partial<NumericSettings>) => void
  resetNumericSettings: (id: string, key: string) => void
  setShowLabels: (id: string, show: boolean) => void
  setHover: (hover: HoverInfo | null) => void
  setSelection: (selection: HoleDetails | null) => void
  setLegend: (id: string, entry: LegendEntry | null) => void
}

/** Default legend range of a numeric attribute: 0 (or the low percentile) to the 98th. */
export function defaultSettings(attribute: AttributeInfo): NumericSettings {
  const fallback: NumericSettings = { mode: 'continuous', min: 0, max: 1, steps: DEFAULT_STEPS }
  if (attribute.type !== 'numeric') return fallback
  const low = attribute.display_min ?? attribute.min
  const high = attribute.display_max ?? attribute.max
  if (low == null || high == null) return fallback
  // Grades start at zero: anchor the ramp there when the data are non-negative.
  const min = (attribute.min ?? low) >= 0 ? 0 : roundNice(low, 'down')
  const range = sanitiseRange(min, roundNice(high, 'up')) ?? sanitiseRange(min, min + 1)
  return range ? { ...fallback, ...range } : fallback
}

function defaultColorBy(attributes: AttributeInfo[]): string | null {
  const first = attributes.find((a) => a.type === 'numeric') ?? attributes[0]
  return first ? attributeKey(first) : null
}

function sameAttributes(a: AttributeInfo[], b: AttributeInfo[]): boolean {
  return a.length === b.length && a.every((x, i) => attributeKey(x) === attributeKey(b[i]) && x.type === b[i].type)
}

export const useDisplay = create<DisplayState>()((set) => ({
  drillholes: {},
  hover: null,
  selection: null,
  legends: {},

  registerDrillholes: (id, attributes) =>
    set((s) => {
      const previous = s.drillholes[id]
      if (previous && sameAttributes(previous.attributes, attributes)) {
        return { drillholes: { ...s.drillholes, [id]: { ...previous, attributes } } }
      }
      const settings: Record<string, NumericSettings> = {}
      for (const a of attributes) if (a.type === 'numeric') settings[attributeKey(a)] = defaultSettings(a)
      return {
        drillholes: {
          ...s.drillholes,
          [id]: { attributes, colorBy: defaultColorBy(attributes), settings, showLabels: previous?.showLabels ?? true },
        },
      }
    }),
  unregisterDrillholes: (id) =>
    set((s) => {
      const drillholes = { ...s.drillholes }
      delete drillholes[id]
      return {
        drillholes,
        hover: s.hover?.objectId === id ? null : s.hover,
        selection: s.selection?.objectId === id ? null : s.selection,
      }
    }),
  setColorBy: (id, key) =>
    set((s) => {
      const d = s.drillholes[id]
      if (!d || (key !== null && !d.attributes.some((a) => attributeKey(a) === key))) return s
      return { drillholes: { ...s.drillholes, [id]: { ...d, colorBy: key } } }
    }),
  setNumericSettings: (id, key, patch) =>
    set((s) => {
      const d = s.drillholes[id]
      const current = d?.settings[key]
      if (!d || !current) return s
      const next = { ...current, ...patch }
      const range = sanitiseRange(next.min, next.max)
      if (!range) return s
      const steps = Math.min(12, Math.max(2, Math.round(next.steps) || DEFAULT_STEPS))
      return {
        drillholes: { ...s.drillholes, [id]: { ...d, settings: { ...d.settings, [key]: { ...next, ...range, steps } } } },
      }
    }),
  resetNumericSettings: (id, key) =>
    set((s) => {
      const d = s.drillholes[id]
      const attribute = d?.attributes.find((a) => attributeKey(a) === key)
      if (!d || !attribute) return s
      const mode = d.settings[key]?.mode ?? 'continuous'
      return {
        drillholes: {
          ...s.drillholes,
          [id]: { ...d, settings: { ...d.settings, [key]: { ...defaultSettings(attribute), mode } } },
        },
      }
    }),
  setShowLabels: (id, showLabels) =>
    set((s) => {
      const d = s.drillholes[id]
      return d ? { drillholes: { ...s.drillholes, [id]: { ...d, showLabels } } } : s
    }),
  setHover: (hover) => set((s) => (s.hover === null && hover === null ? s : { hover })),
  setSelection: (selection) => set({ selection }),
  setLegend: (id, entry) =>
    set((s) => {
      const legends = { ...s.legends }
      if (entry) legends[id] = entry
      else if (id in legends) delete legends[id]
      else return s
      return { legends }
    }),
}))

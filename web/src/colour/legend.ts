/**
 * Legend model: how attribute values map to colours, independent of what is drawn.
 * A layer builds a `LegendSpec`, colours its buffers with `numericColor` /
 * `codeColor` and hands the same spec to <Legend/> (ui/Legend.tsx), so the picture and
 * its legend cannot disagree. Used by drillholes and meshes; blocks will reuse it.
 */
import type { Theme } from '../state/store'
import { categoryColor, explicitColor, hexToRgb, MISSING_COLOR, rampColor, rgbToHex, type RampName, type Rgb } from './ramps'

export type NumericMode = 'continuous' | 'stepped'

export interface NumericLegendSpec {
  kind: 'numeric'
  title: string
  unit: string | null
  ramp: RampName
  mode: NumericMode
  /** Colour range; values outside are drawn in the end colours. */
  min: number
  max: number
  /** Number of equal classes in stepped mode. */
  steps: number
  /** Extent of the data, when known: tells whether the range clips values. */
  dataMin?: number | null
  dataMax?: number | null
  /** Whether some items have no value (adds the "no value" row). */
  hasMissing: boolean
}

export interface CategoricalLegendSpec {
  kind: 'categorical'
  title: string
  categories: string[]
  /**
   * Colours supplied by the data, category -> `#rrggbb` (e.g. the unit dictionary of a
   * geology model). A category listed here has that colour in both themes, whatever
   * else is in `categories`; the others take the palette by their position in the list.
   */
  colours?: Readonly<Record<string, string>>
  hasMissing: boolean
}

export type LegendSpec = NumericLegendSpec | CategoricalLegendSpec

/** User-editable part of a numeric legend. */
export interface NumericSettings {
  mode: NumericMode
  min: number
  max: number
  steps: number
}

export const DEFAULT_STEPS = 6

/** Position of `value` on the ramp, 0..1, or null for a missing value. */
export function rampPosition(spec: NumericLegendSpec, value: number): number | null {
  if (!Number.isFinite(value)) return null
  const span = spec.max - spec.min
  const t = span > 0 ? (value - spec.min) / span : value >= spec.max ? 1 : 0
  const clamped = Math.min(1, Math.max(0, t))
  if (spec.mode === 'continuous') return clamped
  const steps = Math.max(1, Math.floor(spec.steps))
  const bin = Math.min(steps - 1, Math.floor(clamped * steps))
  // One colour per class: the ramp at the class centre.
  return (bin + 0.5) / steps
}

/** sRGB colour of a numeric value; the "no value" colour when it is NaN. */
export function numericColor(spec: NumericLegendSpec, value: number, theme: Theme): Rgb {
  const t = rampPosition(spec, value)
  return t === null ? hexToRgb(MISSING_COLOR[theme]) : rampColor(spec.ramp, theme, t)
}

type CategoryColouring = Pick<CategoricalLegendSpec, 'categories' | 'colours'>

const resolved = new WeakMap<object, Partial<Record<Theme, { hex: string[]; rgb: Rgb[] }>>>()

function resolve(spec: CategoryColouring, theme: Theme): { hex: string[]; rgb: Rgb[] } {
  let perTheme = resolved.get(spec)
  if (!perTheme) resolved.set(spec, (perTheme = {}))
  let entry = perTheme[theme]
  // Specs are built once and not mutated; the length test catches the one cheap case.
  if (!entry || entry.hex.length !== spec.categories.length) {
    const hex = spec.categories.map((name, i) => explicitColor(spec.colours, name) ?? categoryColor(theme, i))
    entry = perTheme[theme] = { hex, rgb: hex.map(hexToRgb) }
  }
  return entry
}

/**
 * `#rrggbb` per category, in list order: the colour the data supplies for that category
 * when there is one, else the palette entry of its position (cycling past 20).
 */
export function categoryColors(spec: CategoryColouring, theme: Theme): readonly string[] {
  return resolve(spec, theme).hex
}

/**
 * sRGB colour of a category code; the "no value" colour for -1. Pass the legend's spec
 * so that colours supplied by the data apply; without it the palette is used.
 */
export function codeColor(code: number, theme: Theme, spec?: CategoryColouring): Rgb {
  if (!(code >= 0)) return hexToRgb(MISSING_COLOR[theme])
  const index = Math.floor(code)
  if (spec && index < spec.categories.length) return resolve(spec, theme).rgb[index]
  return hexToRgb(categoryColor(theme, index))
}

/** Class boundaries of a stepped legend: steps + 1 values from min to max. */
export function steppedBreaks(spec: Pick<NumericLegendSpec, 'min' | 'max' | 'steps'>): number[] {
  const steps = Math.max(1, Math.floor(spec.steps))
  return Array.from({ length: steps + 1 }, (_, i) => spec.min + ((spec.max - spec.min) * i) / steps)
}

/** Decimals needed to tell apart values `step` apart (0..6). */
export function decimalsFor(step: number): number {
  if (!(step > 0) || !Number.isFinite(step)) return 0
  return Math.min(6, Math.max(0, Math.ceil(-Math.log10(step)) + 1))
}

export function formatValue(value: number, decimals: number): string {
  return value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals, useGrouping: false })
}

export interface LegendRow {
  color: string
  label: string
}

/** Rows of a stepped or categorical legend, top to bottom (high values first). */
export function legendRows(spec: LegendSpec, theme: Theme): LegendRow[] {
  const rows: LegendRow[] = []
  if (spec.kind === 'categorical') {
    const colors = categoryColors(spec, theme)
    spec.categories.forEach((name, i) => rows.push({ color: colors[i], label: name }))
  } else if (spec.mode === 'stepped') {
    const breaks = steppedBreaks(spec)
    const decimals = decimalsFor((spec.max - spec.min) / Math.max(1, breaks.length - 1))
    for (let i = breaks.length - 2; i >= 0; i--) {
      const low = formatValue(breaks[i], decimals)
      const high = formatValue(breaks[i + 1], decimals)
      const top = i === breaks.length - 2
      const clipsHigh = top && spec.dataMax != null && spec.dataMax > spec.max
      const clipsLow = i === 0 && spec.dataMin != null && spec.dataMin < spec.min
      const label = clipsHigh ? `≥ ${low}` : clipsLow ? `< ${high}` : `${low} – ${high}`
      rows.push({ color: rgbToHex(rampColor(spec.ramp, theme, (i + 0.5) / (breaks.length - 1))), label })
    }
  }
  if (spec.hasMissing) rows.push({ color: MISSING_COLOR[theme], label: 'No value' })
  return rows
}

/** Tick labels of a continuous legend: low, middle, high. */
export function continuousTicks(spec: NumericLegendSpec): string[] {
  const decimals = decimalsFor((spec.max - spec.min) / 4)
  const mid = (spec.min + spec.max) / 2
  const low = formatValue(spec.min, decimals)
  const high = formatValue(spec.max, decimals)
  return [
    spec.dataMin != null && spec.dataMin < spec.min ? `≤ ${low}` : low,
    formatValue(mid, decimals),
    spec.dataMax != null && spec.dataMax > spec.max ? `≥ ${high}` : high,
  ]
}

/** `value` rounded up (or down) to two significant digits: 1.9205 -> 2, 0.0437 -> 0.044. */
export function roundNice(value: number, direction: 'up' | 'down'): number {
  if (!Number.isFinite(value) || value === 0) return value
  const unit = Math.pow(10, Math.floor(Math.log10(Math.abs(value))) - 1)
  const scaled = value / unit
  const rounded = direction === 'up' ? Math.ceil(scaled - 1e-9) : Math.floor(scaled + 1e-9)
  // Multiplying by a negative power of ten leaves float noise (0.044000000000000004).
  return Number((rounded * unit).toPrecision(12))
}

/**
 * `[min, max]` widened to round numbers on a step of half the span's power of ten:
 * 1172.4 .. 1232.4 -> 1170 .. 1235. Null when the input is not a range.
 */
export function niceRange(min: number, max: number): { min: number; max: number } | null {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min)) return null
  const step = Math.pow(10, Math.floor(Math.log10(max - min))) / 2
  const clean = (v: number) => Number(v.toPrecision(12))
  return { min: clean(Math.floor(min / step + 1e-9) * step), max: clean(Math.ceil(max / step - 1e-9) * step) }
}

/** A sane range from user input: finite, and max above min. Null when unusable. */
export function sanitiseRange(min: number, max: number): { min: number; max: number } | null {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return null
  return min < max ? { min, max } : { min: max, max: min }
}

/** "Au (g/t)" style title. */
export function legendTitle(spec: LegendSpec): string {
  return spec.kind === 'numeric' && spec.unit ? `${spec.title} (${spec.unit})` : spec.title
}

/**
 * Colouring of blocks: pure functions, no three.js.
 *
 * The GPU keeps ONE float per block (`valueBuffer`: the raw attribute value or category
 * code, `MISSING` for no value) and a 256-texel colour table (`rampTexture`). The
 * fragment shader turns the value into a table position with the legend's range and
 * mode (`tablePosition` is the same arithmetic in TypeScript, tested against
 * colour/legend.ts). So changing the range, the mode or the theme rewrites a uniform or
 * 1 kB of texture, never a per-block buffer.
 */
import type { BlockAttributeInfo, BlockColumn } from '../api/blocks'
import { codeColor, numericColor, type LegendSpec, type NumericSettings } from '../colour/legend'
import { categoryColor, hexToRgb, MISSING_COLOR, rampColor, type Rgb } from '../colour/ramps'
import type { Theme } from '../state/store'

/** Stands for "no value" in the per-block value buffer (NaN is not reliable in GLSL). */
export const MISSING = -3e38
/** Anything below this is `MISSING` (shader and CPU use the same test). */
export const MISSING_BELOW = -1e37
export const TABLE_SIZE = 256

export type MissingDisplay = 'hidden' | 'neutral'

/** One float per block for the shader: value or category code; `MISSING` for none. */
export function valueBuffer(column: BlockColumn | null, count: number, out: Float32Array = new Float32Array(count)): Float32Array {
  if (!column) return out.fill(0)
  if (column.type === 'numeric') {
    const v = column.values
    for (let i = 0; i < count; i++) out[i] = Number.isNaN(v[i]) ? MISSING : v[i]
  } else {
    const c = column.codes
    for (let i = 0; i < count; i++) out[i] = c[i] < 0 ? MISSING : c[i]
  }
  return out
}

export function countMissing(values: Float32Array): number {
  let n = 0
  for (let i = 0; i < values.length; i++) if (values[i] < MISSING_BELOW) n++
  return n
}

/** The legend of the colour attribute, or null when blocks are drawn in one colour. */
export function blockLegend(
  info: BlockAttributeInfo | undefined,
  settings: NumericSettings | undefined,
  hasMissing: boolean,
): LegendSpec | null {
  if (!info) return null
  if (info.type === 'categorical') return { kind: 'categorical', title: info.name, categories: info.categories, hasMissing }
  if (!settings) return null
  return {
    kind: 'numeric',
    title: info.name,
    unit: info.unit,
    // The same ramp as the drillhole grades: one grade, one colour, in every layer.
    ramp: 'grade',
    mode: settings.mode,
    min: settings.min,
    max: settings.max,
    steps: settings.steps,
    dataMin: info.min,
    dataMax: info.max,
    hasMissing,
  }
}

/**
 * The colour table: RGBA bytes, sRGB. Numeric: the ramp sampled at i / 255. Categorical:
 * texel i is the colour of category code i.
 */
export function rampTexture(spec: LegendSpec | null, theme: Theme, out: Uint8Array = new Uint8Array(TABLE_SIZE * 4)): Uint8Array {
  for (let i = 0; i < TABLE_SIZE; i++) {
    const c: Rgb = !spec
      ? hexToRgb(NEUTRAL_COLOR[theme])
      : spec.kind === 'numeric'
        ? rampColor(spec.ramp, theme, i / (TABLE_SIZE - 1))
        : hexToRgb(categoryColor(theme, i))
    out[i * 4] = Math.round(c[0] * 255)
    out[i * 4 + 1] = Math.round(c[1] * 255)
    out[i * 4 + 2] = Math.round(c[2] * 255)
    out[i * 4 + 3] = 255
  }
  return out
}

/** Colour of blocks without a colour attribute. */
export const NEUTRAL_COLOR: Record<Theme, string> = { dark: '#8d99a5', light: '#7b858e' }

/** What the shader needs besides the table: range, classes, kind. */
export interface ShaderColouring {
  categorical: boolean
  min: number
  max: number
  /** Number of classes; 0 = continuous. */
  steps: number
}

export function shaderColouring(spec: LegendSpec | null): ShaderColouring {
  if (!spec || spec.kind === 'categorical') return { categorical: true, min: 0, max: 1, steps: 0 }
  return { categorical: false, min: spec.min, max: spec.max, steps: spec.mode === 'stepped' ? Math.max(1, Math.floor(spec.steps)) : 0 }
}

/**
 * Position in the colour table, 0..1 along the ramp (numeric) or the texel index
 * (categorical) - exactly what the fragment shader computes from the block's value.
 */
export function tablePosition(value: number, c: ShaderColouring): number {
  if (c.categorical) return Math.min(value, TABLE_SIZE - 1)
  const span = c.max - c.min
  let x = span > 0 ? (value - c.min) / span : value >= c.max ? 1 : 0
  x = Math.min(1, Math.max(0, x))
  if (c.steps > 0) x = (Math.min(Math.floor(x * c.steps), c.steps - 1) + 0.5) / c.steps
  return x
}

/** Colour a block gets on screen (lit face, sRGB 0..1), read from the table as the GPU does. */
export function tableColor(table: Uint8Array, value: number, c: ShaderColouring, theme: Theme): Rgb {
  if (value < MISSING_BELOW) return hexToRgb(MISSING_COLOR[theme])
  const t = tablePosition(value, c)
  const x = c.categorical ? t : t * (TABLE_SIZE - 1)
  const i = Math.min(TABLE_SIZE - 2, Math.floor(x))
  const f = x - i
  const at = (k: number) => (table[i * 4 + k] * (1 - f) + table[(i + 1) * 4 + k] * f) / 255
  return [at(0), at(1), at(2)]
}

/** The legend's own colour for a value (colour/legend.ts): what `tableColor` must match. */
export function legendColor(spec: LegendSpec | null, value: number, theme: Theme): Rgb {
  if (!spec) return hexToRgb(NEUTRAL_COLOR[theme])
  if (value < MISSING_BELOW || Number.isNaN(value)) return hexToRgb(MISSING_COLOR[theme])
  return spec.kind === 'numeric' ? numericColor(spec, value, theme) : codeColor(value, theme)
}

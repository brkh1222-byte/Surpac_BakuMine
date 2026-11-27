/**
 * Colouring and read-outs of drillhole intervals: pure functions shared by the layer,
 * the tooltip and the hole panel. No three.js here.
 */
import type { AttributeInfo, DrillholeSet, SegmentAttribute } from '../api/client'
import {
  codeColor,
  formatValue,
  numericColor,
  type LegendSpec,
  type NumericSettings,
} from '../colour/legend'
import { hexToRgb, rgbToHex, srgbToLinear, type Rgb } from '../colour/ramps'
import type { HoleDetails, ValueReadout } from '../state/display'
import type { Theme } from '../state/store'
import type { DrillholePieces } from './pieces'

/** Trace colour without an attribute, and halo / collar colours, per theme. */
export const INK: Record<Theme, { trace: string; hover: string; select: string; collar: string }> = {
  dark: { trace: '#b9c2cb', hover: '#ffffff', select: '#e2a03a', collar: '#e8edf1' },
  light: { trace: '#3d454c', hover: '#0b0b0b', select: '#a45f00', collar: '#4a535b' },
}

export function findAttribute(set: DrillholeSet, colorBy: string | null): SegmentAttribute | null {
  if (!colorBy || !set.segments) return null
  return set.segments.attributes.find((a) => `${set.segments?.table}/${a.name}` === colorBy) ?? null
}

/** The legend of the active attribute, or null when traces are drawn in one colour. */
export function drillholeLegend(
  attribute: SegmentAttribute | null,
  info: AttributeInfo | undefined,
  settings: NumericSettings | undefined,
): LegendSpec | null {
  if (!attribute) return null
  if (attribute.type === 'categorical') {
    const spec: LegendSpec = {
      kind: 'categorical',
      title: attribute.name,
      categories: attribute.categories,
      hasMissing: attribute.codes.includes(-1),
    }
    // Colours the data supplies (manifest entry), keyed by category name.
    if (info?.type === 'categorical' && info.colours) spec.colours = info.colours
    return spec
  }
  if (!settings) return null
  const numeric = info?.type === 'numeric' ? info : undefined
  return {
    kind: 'numeric',
    title: attribute.name,
    unit: attribute.unit,
    ramp: 'grade',
    mode: settings.mode,
    min: settings.min,
    max: settings.max,
    steps: settings.steps,
    dataMin: numeric?.min ?? null,
    dataMax: numeric?.max ?? null,
    hasMissing: attribute.values.some((v) => Number.isNaN(v)),
  }
}

/** sRGB colour of a segment (-1 = stretch without interval) under `spec`. */
export function segmentColor(attribute: SegmentAttribute | null, spec: LegendSpec | null, segment: number, theme: Theme): Rgb {
  if (!attribute || !spec) return hexToRgb(INK[theme].trace)
  if (segment < 0) return codeColor(-1, theme)
  if (attribute.type === 'numeric') {
    return spec.kind === 'numeric' ? numericColor(spec, attribute.values[segment], theme) : codeColor(-1, theme)
  }
  return codeColor(attribute.codes[segment], theme, spec.kind === 'categorical' ? spec : undefined)
}

/** Linear-RGB instance colours (start + end, 6 per piece) for the line buffer. */
export function pieceColors(
  pieces: DrillholePieces,
  attribute: SegmentAttribute | null,
  spec: LegendSpec | null,
  theme: Theme,
  out: Float32Array = new Float32Array(pieces.count * 6),
): Float32Array {
  let lastSegment = Number.NaN
  let r = 0
  let g = 0
  let b = 0
  for (let piece = 0; piece < pieces.count; piece++) {
    const segment = pieces.interval[piece]
    if (segment !== lastSegment) {
      const c = segmentColor(attribute, spec, segment, theme)
      r = srgbToLinear(c[0])
      g = srgbToLinear(c[1])
      b = srgbToLinear(c[2])
      lastSegment = segment
    }
    const i = piece * 6
    out[i] = out[i + 3] = r
    out[i + 1] = out[i + 4] = g
    out[i + 2] = out[i + 5] = b
  }
  return out
}

/** Values of one interval as text; `units: false` leaves the unit out (table cells). */
export function readouts(set: DrillholeSet, segment: number, units = true): ValueReadout[] {
  const seg = set.segments
  if (!seg || segment < 0) return []
  return seg.attributes.map((a) => {
    if (a.type === 'categorical') {
      const code = a.codes[segment]
      return { name: a.name, text: code < 0 ? 'no value' : (a.categories[code] ?? 'no value') }
    }
    const value = a.values[segment]
    if (Number.isNaN(value)) return { name: a.name, text: 'no value' }
    const text = formatValue(value, Math.abs(value) >= 100 ? 1 : 3)
    return { name: a.name, text: a.unit && units ? `${text} ${a.unit}` : text }
  })
}

export function holeDetails(
  objectId: string,
  set: DrillholeSet,
  hole: number,
  attribute: SegmentAttribute | null,
  spec: LegendSpec | null,
  theme: Theme,
): HoleDetails {
  const seg = set.segments
  const rows: HoleDetails['rows'] = []
  if (seg) {
    const mine: number[] = []
    for (let s = 0; s < seg.count; s++) if (seg.hole[s] === hole) mine.push(s)
    mine.sort((a, b) => seg.from[a] - seg.from[b])
    for (const s of mine) {
      rows.push({
        from: seg.from[s],
        to: seg.to[s],
        values: readouts(set, s, false).map((v) => v.text),
        color: attribute && spec ? rgbToHex(segmentColor(attribute, spec, s, theme)) : null,
      })
    }
  }
  return {
    objectId,
    hole,
    holeId: set.holeIds[hole],
    collar: [set.collarXyz[hole * 3], set.collarXyz[hole * 3 + 1], set.collarXyz[hole * 3 + 2]],
    maxDepth: set.maxDepth[hole],
    table: seg?.table ?? null,
    columns: seg ? seg.attributes.map((a) => (a.type === 'numeric' && a.unit ? `${a.name} (${a.unit})` : a.name)) : [],
    rows,
  }
}

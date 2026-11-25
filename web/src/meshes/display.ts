/**
 * Display settings of a triangle mesh (DTM surface or closed solid) and the pure
 * helpers the mesh layer colours with. No three.js here.
 */
import { DEFAULT_STEPS, niceRange, numericColor, sanitiseRange, type NumericLegendSpec, type NumericSettings } from '../colour/legend'
import { srgbToLinear } from '../colour/ramps'
import type { Theme } from '../state/store'

export type MeshMode = 'shaded' | 'edges' | 'wireframe' | 'transparent'
export type MeshShading = 'smooth' | 'flat'

export const MESH_MODES: readonly { id: MeshMode; label: string }[] = [
  { id: 'shaded', label: 'Shaded' },
  { id: 'edges', label: 'Shaded + edges' },
  { id: 'wireframe', label: 'Wireframe' },
  { id: 'transparent', label: 'Transparent' },
]

export interface MeshDisplay {
  /** A closed solid, or an open surface (DTM). */
  closed: boolean
  mode: MeshMode
  /** Object colour, "#rrggbb". */
  color: string
  /** Opacity in transparent mode, 0.05..1. */
  opacity: number
  shading: MeshShading
  /** Colour the vertices by their elevation instead of the object colour. */
  colorByElevation: boolean
  /** Legend settings of the elevation colouring. */
  elevation: NumericSettings
  /** Lowest and highest vertex, true Z. */
  zRange: [number, number]
  /** Set for a mesh of a geology model: it takes no colour of the surface / solid palettes. */
  group?: string
}

/** Object colours handed out in order of registration, per kind. */
const SURFACE_COLORS = ['#8c9b76', '#9a8f74', '#7f9a94']
const SOLID_COLORS = ['#c2577a', '#5a86c4', '#c98a3c', '#7d68c0']

export function defaultElevationSettings(zRange: readonly [number, number]): NumericSettings {
  const range = niceRange(zRange[0], zRange[1]) ?? sanitiseRange(zRange[0], zRange[0] + 1) ?? { min: 0, max: 1 }
  return { mode: 'continuous', steps: DEFAULT_STEPS, ...range }
}

/**
 * Defaults: a solid is opaque and shaded; a surface is see-through, because an opaque
 * topography hides everything a mine model is about. `index` is the number of meshes of
 * the same kind registered before this one (picks the colour).
 */
export function defaultMeshDisplay(closed: boolean, zRange: readonly [number, number], index = 0): MeshDisplay {
  const colors = closed ? SOLID_COLORS : SURFACE_COLORS
  return {
    closed,
    mode: closed ? 'shaded' : 'transparent',
    color: colors[index % colors.length],
    opacity: closed ? 0.35 : 0.45,
    shading: 'smooth',
    colorByElevation: false,
    elevation: defaultElevationSettings(zRange),
    zRange: [zRange[0], zRange[1]],
  }
}

/** Colour of a fault mesh: neutral, so that it reads against every unit colour. */
export const FAULT_COLOR = '#9aa3ad'

/** What the manifest says about a mesh of a geology model. */
export interface GeologyMeshStyle {
  /** "Units", "Faults" or "Surfaces". */
  group: string
  /** Colour of the unit or zone in the dictionary, `#rrggbb`; none for faults. */
  colour?: string
}

/**
 * Display of a mesh of a geology model: a unit solid is opaque in its dictionary colour;
 * a fault is neutral and see-through; a base surface is see-through in its zone colour.
 */
export function geologyMeshDisplay(style: GeologyMeshStyle, closed: boolean, zRange: readonly [number, number]): MeshDisplay {
  const display = defaultMeshDisplay(closed, zRange)
  const given = style.colour && /^#[0-9a-f]{6}$/i.test(style.colour) ? style.colour.toLowerCase() : null
  display.group = style.group
  if (style.group === 'Faults') {
    display.color = given ?? FAULT_COLOR
    display.mode = 'transparent'
    display.opacity = 0.4
  } else if (style.group === 'Surfaces') {
    display.color = given ?? display.color
    display.mode = 'transparent'
    display.opacity = 0.5
  } else {
    display.color = given ?? display.color
    display.mode = 'shaded'
  }
  return display
}

export function clampOpacity(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0.05, value)) : 1
}

export function elevationLegend(display: MeshDisplay): NumericLegendSpec {
  return {
    kind: 'numeric',
    title: 'Elevation',
    unit: 'm',
    ramp: 'elevation',
    mode: display.elevation.mode,
    min: display.elevation.min,
    max: display.elevation.max,
    steps: display.elevation.steps,
    dataMin: display.zRange[0],
    dataMax: display.zRange[1],
    hasMissing: false,
  }
}

/** Lowest and highest z of flat xyz triples. */
export function zRangeOf(xyz: ArrayLike<number>): [number, number] {
  let min = Infinity
  let max = -Infinity
  for (let i = 2; i < xyz.length; i += 3) {
    const z = xyz[i]
    if (z < min) min = z
    if (z > max) max = z
  }
  return min <= max ? [min, max] : [0, 0]
}

/**
 * Linear-RGB vertex colours by TRUE elevation (world z of `vertices`, flat xyz).
 * Colours are quantised to 256 ramp steps, so a million vertices cost a table lookup each.
 */
export function elevationColors(
  vertices: ArrayLike<number>,
  spec: NumericLegendSpec,
  theme: Theme,
  out: Float32Array = new Float32Array(vertices.length),
): Float32Array {
  const steps = 256
  const table = new Float32Array(steps * 3)
  const span = spec.max - spec.min
  for (let i = 0; i < steps; i++) {
    const c = numericColor(spec, spec.min + (span * i) / (steps - 1), theme)
    table[i * 3] = srgbToLinear(c[0])
    table[i * 3 + 1] = srgbToLinear(c[1])
    table[i * 3 + 2] = srgbToLinear(c[2])
  }
  const scale = span > 0 ? (steps - 1) / span : 0
  for (let i = 0; i < vertices.length; i += 3) {
    const t = Math.round((vertices[i + 2] - spec.min) * scale)
    const k = (t < 0 ? 0 : t > steps - 1 ? steps - 1 : t) * 3
    out[i] = table[k]
    out[i + 1] = table[k + 1]
    out[i + 2] = table[k + 2]
  }
  return out
}

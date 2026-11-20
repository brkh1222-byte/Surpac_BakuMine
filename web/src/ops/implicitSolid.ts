/**
 * Implicit solid from drillhole intervals: the dialog's request body (pure, unit-tested),
 * the endpoint (`POST /api/v1/ops/solids/implicit`, bakumine/api/routes_geometry.py) and
 * its result.
 */
import { nums } from './builders'
import { postJson } from './geometryApi'
import { parsePlane } from './orientation'

export interface ImplicitForm {
  database: string
  table: string
  column: string
  /** One code, or several separated by commas. */
  codes: string
  /** '' until the user chose: the dialog asks for the orientation. */
  mode: '' | 'plane' | 'angles' | 'auto' | 'none'
  dip: string
  dipAzimuth: string
  pitch: string
  /** "bearing, plunge, dip" */
  angles: string
  /** "4:2:1": extent along major : semi-major : minor (along strike : down dip : across). */
  ratios: string
  maxExtrapolation: string
  /** Object id of the DTM that clips the solid; '' = none. */
  topography: string
  resolution: string
  minVolume: string
  name: string
}

export function parseRatios(text: string): number[] | null {
  const parts = text.split(/[\s:,;]+/).filter(Boolean).map(Number)
  return parts.length === 3 && parts.every((r) => Number.isFinite(r) && r > 0) ? parts : null
}

function optional(text: string, what: string, allowZero = false): number | null | string {
  if (text.trim() === '') return null
  const value = Number(text)
  if (!Number.isFinite(value) || value < 0 || (!allowZero && value === 0)) return `${what}: a positive number`
  return value
}

export function buildImplicitRequest(f: ImplicitForm): Record<string, unknown> | string {
  if (!f.database) return 'Pick the drillhole database'
  if (!f.table || !f.column) return 'Pick the interval table and the column with the codes'
  const codes = f.codes.split(',').map((c) => c.trim()).filter(Boolean)
  if (codes.length === 0) return 'Give the code of the domain'
  let anisotropy: Record<string, unknown>
  if (f.mode === '') return 'Give the orientation of the body (dip and dip azimuth), or choose automatic / none'
  if (f.mode === 'auto' || f.mode === 'none') anisotropy = { mode: f.mode }
  else {
    const ratios = parseRatios(f.ratios)
    if (!ratios) return 'Ratios: three positive numbers, along strike : down dip : across (for example 4:2:1)'
    if (f.mode === 'plane') {
      const plane = parsePlane(f.dip, f.dipAzimuth, f.pitch)
      if (typeof plane === 'string') return plane
      anisotropy = { mode: 'plane', ...plane, ratios }
    } else {
      const angles = nums(f.angles)
      if (!angles || angles.length !== 3) return 'Angles: three numbers (bearing, plunge, dip)'
      anisotropy = { mode: 'angles', angles, ratios }
    }
  }
  const body: Record<string, unknown> = { database: f.database, table: f.table, column: f.column, codes, anisotropy }
  for (const [key, text, what, zero] of [
    ['max_extrapolation', f.maxExtrapolation, 'Max extrapolation', false],
    ['resolution', f.resolution, 'Resolution', false],
    ['min_component_volume', f.minVolume, 'Min volume', true],
  ] as const) {
    const value = optional(text, what, zero)
    if (typeof value === 'string') return value
    if (value !== null) body[key] = value
  }
  if (f.topography) body.topography = f.topography
  if (f.name.trim()) body.name = f.name.trim()
  return body
}

export interface AxisDirection {
  azimuth: number
  plunge: number
}

export interface AnisotropyUsed {
  /** "user", "auto", "none", "none (auto failed)" */
  source: string
  method: string
  /** relative extent along the major, semi-major and minor axes */
  ratios?: number[]
  angles?: number[]
  major?: AxisDirection
  semi_major?: AxisDirection
  minor?: AxisDirection
  plane?: { dip: number; dip_azimuth: number }
  hole_angle_deg?: number
  /** stacked bodies: 1 - runner-up / best indicator correlation between holes (0..1) */
  confidence?: number
  correlation?: number
  /** the best plane when the automatic orientation was not determined (not used) */
  candidate?: { dip: number | null; dip_azimuth: number | null }
}

export interface ImplicitResult {
  id: string
  name: string
  volume: number
  valid: boolean
  outward: boolean
  issues: string[]
  n_faces: number
  n_components: number
  n_holes: number
  n_holes_intersecting: number
  resolution: number
  touches_grid_boundary: boolean
  anisotropy: AnisotropyUsed
  warnings: string[]
}

export const implicitApi = {
  build: (body: unknown) => postJson<ImplicitResult>('/solids/implicit', body),
}

const pad = (value: number, digits: number) => String(Math.round(value)).padStart(digits, '0')

/** "Anisotropy (user): plane dips 56 towards 310; major 028/18, semi-major 276/50, minor 130/34; extent 1 : 0.56 : 0.06." */
export function describeAnisotropy(a: AnisotropyUsed): string {
  const confidence = a.confidence == null ? '' : ` Confidence ${a.confidence.toFixed(2)}.`
  if (!a.major || !a.semi_major || !a.minor || !a.plane || !a.ratios) return `Anisotropy: ${a.source} (isotropic).${confidence}`
  const axis = (label: string, d: AxisDirection) => `${label} ${pad(d.azimuth, 3)}/${pad(d.plunge, 2)}`
  return (
    `Anisotropy (${a.source}): plane dips ${Math.round(a.plane.dip)} towards ${pad(a.plane.dip_azimuth, 3)}; ` +
    `${axis('major', a.major)}, ${axis('semi-major', a.semi_major)}, ${axis('minor', a.minor)}; ` +
    `extent ${a.ratios.map((r) => Number(r.toFixed(2))).join(' : ')}.${confidence}`
  )
}

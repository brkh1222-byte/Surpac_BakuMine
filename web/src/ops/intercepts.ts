/**
 * Mineralised intervals / significant intercepts (`POST /api/v1/ops/intercepts`, see
 * bakumine/api/ops_intercepts.py): the request builder of the dialog and the call.
 * The parameters are the fields of `InterceptParams` in
 * bakumine/core/drillholes/mineralised.py, where every rule is defined.
 */
import { nums } from './builders'

export interface InterceptsForm {
  database: string
  table: string
  grade: string
  cutoff: string
  minThickness: string
  maxInternalWaste: string
  /** blank = no limit */
  maxTotalWaste: string
  accumulationRule: boolean
  gapTolerance: string
  /** blank = no top cut */
  topCut: string
  /** higher cut-offs for "including" rows, e.g. "1, 5"; blank = none */
  including: string
  includingMinThickness: string
  /** both blank = down-hole length; both given = true thickness */
  bodyDip: string
  bodyDipAzimuth: string
  /** name of the interval table to store; blank = do not store */
  saveAs: string
  overwrite: boolean
}

export interface InterceptRow {
  hole_id: string
  depth_from: number
  depth_to: number
  length: number
  true_thickness: number | null
  grade: number
  grade_cut: number | null
  grade_x_m: number
  waste_length: number
  missing_length: number
  cutoff: number
  rule: string
  level: 'intercept' | 'including'
  parent: number
}

export interface InterceptsResult {
  grade: string
  cutoff: number
  n_holes: number
  n_holes_mineralised: number
  n_intercepts: number
  n_including: number
  mineralised_length: number
  waste_length: number
  grade_x_m: number
  mean_grade: number | null
  interval_table: string | null
  file: string
  truncated: boolean
  table: InterceptRow[]
}

function number(text: string, what: string, min = 0): number | string {
  const value = Number(text.trim())
  if (text.trim() === '' || !Number.isFinite(value) || value < min) return `${what}: a number${min === 0 ? ' ≥ 0' : ''}`
  return value
}

/** The request body, or a message saying what is missing. */
export function buildInterceptsRequest(f: InterceptsForm): Record<string, unknown> | string {
  if (!f.database) return 'Import a drillhole database first'
  if (!f.table) return 'Pick the assay table'
  if (!f.grade) return 'Pick the grade column'
  const params: Record<string, unknown> = {}
  const required: [string, string, string, number][] = [
    ['cutoff', f.cutoff, 'Cut-off', -Infinity],
    ['min_thickness', f.minThickness, 'Minimum thickness', 0],
    ['max_internal_waste', f.maxInternalWaste, 'Internal waste', 0],
    ['gap_tolerance', f.gapTolerance, 'Gap tolerance', 0],
  ]
  for (const [key, text, what, min] of required) {
    const value = number(text, what, min)
    if (typeof value === 'string') return value
    params[key] = value
  }
  const optional: [string, string, string, number][] = [
    ['max_total_waste', f.maxTotalWaste, 'Total internal waste', 0],
    ['top_cut', f.topCut, 'Top cut', -Infinity],
    ['including_min_thickness', f.includingMinThickness, 'Including: minimum thickness', 0],
    ['body_dip', f.bodyDip, 'Body dip', -Infinity],
    ['body_dip_azimuth', f.bodyDipAzimuth, 'Body dip azimuth', -Infinity],
  ]
  for (const [key, text, what, min] of optional) {
    if (text.trim() === '') continue
    const value = number(text, what, min)
    if (typeof value === 'string') return value
    params[key] = value
  }
  if ((f.bodyDip.trim() === '') !== (f.bodyDipAzimuth.trim() === '')) return 'True thickness needs both the dip and the dip azimuth of the body'
  if (f.accumulationRule) params.accumulation_rule = true
  if (f.including.trim()) {
    const cutoffs = nums(f.including)
    if (!cutoffs || cutoffs.some((c) => c <= (params.cutoff as number))) return 'Including: cut-offs above the main cut-off'
    params.including_cutoffs = cutoffs
  }
  const body: Record<string, unknown> = { database: f.database, table: f.table, grade: f.grade, params }
  const name = f.saveAs.trim()
  if (name) {
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(name) || name === 'collar' || name === 'survey') return 'Table name: letters, digits, _ or -'
    if (name === f.table) return 'Store the intervals under another name than the assay table'
    body.save_as = name
    if (f.overwrite) body.overwrite = true
  }
  return body
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`/api/v1/ops${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const detail = ((await response.json().catch(() => null)) as { detail?: unknown } | null)?.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`)
  }
  return (await response.json()) as T
}

export const runIntercepts = (body: unknown) => postJson<InterceptsResult>('/intercepts', body)

/** "BM012 34.0–41.5 m: 7.5 m @ 1.82 (13.7 g×m)" */
export function describeIntercept(row: InterceptRow, digits = 2): string {
  const span = `${row.depth_from.toFixed(1)}–${row.depth_to.toFixed(1)} m`
  const text = `${span}: ${row.length.toFixed(1)} m @ ${row.grade.toFixed(digits)} (${row.grade_x_m.toFixed(1)} g×m)`
  return row.level === 'including' ? `incl. ${text}` : `${row.hole_id} ${text}`
}

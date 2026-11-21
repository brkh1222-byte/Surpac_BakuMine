/**
 * Report client: the request body of `POST /api/v1/scene/objects/{id}/report` (and
 * `report.csv`) built from the panel's settings, and number formatting for the table.
 * The body mirrors `bakumine.core.report.report` and `bakumine blockmodel report`.
 */
import { API_BASE, ApiError } from '../api/client'

export interface ReportSettings {
  /** Grade attribute names; the first one is also the cut-off attribute. */
  grades: string[]
  /** Density: an attribute name, or a constant as text ("2.7"); '' = none. */
  density: string
  groupBy: string[]
  /** Comma-separated cut-offs ("0.5, 1"); '' = none. */
  cutoffs: string
  /** Bench height as text; '' = no benches. */
  benchHeight: string
  where: string
}

export interface ReportRequest {
  where?: string
  group_by: string[]
  cutoffs?: number[]
  cutoff_attribute?: string
  grade_attributes: string[]
  density?: string | number
  bench_height?: number
}

export interface ReportTable {
  columns: string[]
  rows: Record<string, string | number | null>[]
  notes: string[]
}

/** Numbers separated by commas, semicolons or spaces (dot decimals); null when any is not a number. */
export function parseNumbers(text: string): number[] | null {
  const out: number[] = []
  for (const item of text.split(/[\s,;]+/).filter(Boolean)) {
    const value = Number(item)
    if (!Number.isFinite(value)) return null
    out.push(value)
  }
  return out
}

/**
 * The settings without the attributes the model no longer has (an undo removed them):
 * grades and density among ``numeric``, group-by among ``categorical``. A density given as
 * a number is kept. The same object is returned when nothing was dropped.
 */
export function keepKnown(s: ReportSettings, numeric: string[], categorical: string[]): ReportSettings {
  const grades = s.grades.filter((g) => numeric.includes(g))
  const groupBy = s.groupBy.filter((g) => categorical.includes(g))
  const text = s.density.trim()
  const density = !text || Number.isFinite(Number(text.replace(',', '.'))) || numeric.includes(text) ? s.density : ''
  if (grades.length === s.grades.length && groupBy.length === s.groupBy.length && density === s.density) return s
  return { ...s, grades, groupBy, density }
}

/** The request for the settings; a string is returned when a field is not valid. */
export function buildReportRequest(s: ReportSettings, units: Record<string, string | null> = {}): ReportRequest | string {
  const body: ReportRequest = {
    group_by: s.groupBy,
    grade_attributes: s.grades.map((g) => (units[g] ? `${g}:${units[g]}` : g)),
  }
  if (s.where.trim()) body.where = s.where.trim()
  if (s.cutoffs.trim()) {
    const cutoffs = parseNumbers(s.cutoffs)
    if (!cutoffs || cutoffs.length === 0) return 'Cut-offs must be numbers, e.g. 0.5, 1'
    if (s.grades.length === 0) return 'Pick a grade attribute for the cut-offs'
    body.cutoffs = cutoffs
    body.cutoff_attribute = s.grades[0]
  }
  const density = s.density.trim()
  if (density) {
    const value = Number(density.replace(',', '.'))
    body.density = Number.isFinite(value) ? value : density
  }
  if (s.benchHeight.trim()) {
    const height = Number(s.benchHeight.replace(',', '.'))
    if (!(height > 0)) return 'Bench height must be a positive number'
    body.bench_height = height
  }
  return body
}

async function post(objectId: string, suffix: string, body: ReportRequest, fetchFn: typeof fetch): Promise<Response> {
  let response: Response
  try {
    response = await fetchFn(`${API_BASE}/scene/objects/${encodeURIComponent(objectId)}/${suffix}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new ApiError('The API is not reachable. Start it with `make dev`.')
  }
  if (!response.ok) {
    let detail = response.statusText
    try {
      const json = (await response.json()) as { detail?: unknown }
      if (typeof json.detail === 'string') detail = json.detail
    } catch {
      // not JSON
    }
    throw new ApiError(detail, response.status)
  }
  return response
}

export async function fetchReport(objectId: string, body: ReportRequest, fetchFn: typeof fetch = fetch): Promise<ReportTable> {
  const response = await post(objectId, 'report', body, fetchFn)
  return (await response.json()) as ReportTable
}

export async function fetchReportCsv(objectId: string, body: ReportRequest, fetchFn: typeof fetch = fetch): Promise<string> {
  const response = await post(objectId, 'report.csv', body, fetchFn)
  return response.text()
}

/** Table cell text: counts as integers with spaces, grades to 3 decimals, null as an en dash. */
export function formatCell(column: string, value: string | number | null): string {
  if (value === null || value === undefined) return '–'
  if (typeof value === 'string') return value
  const decimals = column.endsWith('_grade') || column === 'cutoff' ? 3 : column.startsWith('bench_') ? 1 : 0
  return value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).replace(/,/g, ' ')
}

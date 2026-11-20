/**
 * Client of the operation endpoints (`/api/v1/ops`, see bakumine/api/routes_ops.py) and the
 * store of what the dialogs pick from (the project's databases, composites, models, ...).
 * After an operation changed the project, `changed()` refreshes the summary and reloads the
 * scene so new objects appear in the tree.
 */
import { create } from 'zustand'
import { clearResources } from '../api/useResource'

const OPS = '/api/v1/ops'

export interface IntervalTableInfo {
  columns: string[]
  attributes: string[]
  rows: number
}

export interface ProjectObjectInfo {
  id: string
  kind: 'database' | 'composite' | 'string' | 'dtm' | 'solid' | 'model' | 'constraint'
  name: string
  tables?: Record<string, IntervalTableInfo>
  columns?: string[]
  rows?: number
  attributes?: { name: string; kind: string }[]
  n_blocks?: number
  error?: string
}

export interface ProjectSummary {
  name: string
  objects: ProjectObjectInfo[]
}

/** How one uploaded file is read: a source of an import recipe (bakumine/core/drillholes/recipe.py). */
export interface SourceSpec {
  sheets?: string[]
  header_row?: number
  header_rows?: number
  map?: Record<string, string>
  hole_id_from_sheet?: boolean
  sample_depth?: { length: number; patterns?: string[] }
  qc?: { mode: 'separate' | 'off' }
  /** preview only: the sheet whose raw rows are shown */
  preview_sheet?: string
}

export interface HoleIdRule {
  pattern: string
  template: string
}

export interface SampleDepthCheck {
  rows: number
  duplicates: number
  backwards: number
  gaps: number
  overlaps: number
  unmatched: number
  unmatched_examples: string[]
}

export interface UploadPreview {
  name: string
  kind: 'collar' | 'survey' | 'interval'
  /** position of the file among the files uploaded for the table */
  index: number
  n_files: number
  file: string
  rows: number
  columns: string[]
  preview: Record<string, string>[]
  mapping: Record<string, string>
  missing: string[]
  unmatched: string[]
  sheets: { name: string; rows: number }[]
  selected: { name: string; header_row: number; header_rows: number; rows: number; skipped: string[] }[]
  /** the first raw rows of `grid_sheet`, for picking the header row */
  grid: string[][]
  grid_sheet: string
  grid_header_row: number
  grid_header_rows: number
  /** the rows as they will be imported (canonical names, derived depths) */
  prepared: { columns: string[]; rows: string[][] }
  qc: Record<string, number>
  sample_depth: (SampleDepthCheck & { preview: { hole_id: string; sample_id: string; depth_from: number; depth_to: number }[] }) | null
  notes: string[]
}

export interface MergeReport {
  precedence: string
  keys: string[]
  merged: number
  filled: number
  conflicts: number
  rows: Record<string, string | number | null>[]
}

export interface CoordinateReport {
  crs: string
  order: string
  tolerance: number
  converted: number
  checked: number
  disagree: number
  rows: { hole_id: string; difference: number }[]
}

export interface ValidationResult {
  n_holes: number
  errors: number
  warnings: number
  infos: number
  holes_with_errors: number
  counts: { code: string; severity: string; description: string; count: number }[]
  issues: { severity: string; code: string; table: string; hole_id: string; row: number | null; message: string }[]
  truncated: boolean
}

/** Which holes an import keeps: names or wildcard patterns. */
export interface HoleFilter {
  include?: string[]
  exclude?: string[]
}

export interface HoleFilterReport {
  include: string[]
  exclude: string[]
  tables: Record<string, { rows_kept: number; rows_dropped: number; holes_dropped: number }>
  /** patterns that matched no hole in any table */
  unmatched: string[]
}

export interface DroppedRows {
  total_rows?: number
  rows_without_hole_id?: number
  collars_without_coordinates?: number
  holes_without_coordinates?: string[]
}

export interface ImportResult extends ValidationResult {
  id: string
  name: string
  tables: Record<string, number>
  notes: string[]
  qc: Record<string, Record<string, number>>
  conflicts: Record<string, MergeReport>
  coordinates: CoordinateReport | null
  survey: { generated_from_collar: number; holes_without_azimuth_or_dip: string[] } | null
  sample_depth: Record<string, SampleDepthCheck[]>
  /** per table: total_rows, rows_without_hole_id, collars_without_coordinates (counts) */
  dropped_rows: Record<string, DroppedRows>
  hole_filter: HoleFilterReport | null
  /** per interval table: attribute -> unit read from its header */
  units: Record<string, Record<string, string>>
  dip_convention: string
  coordinate_order: string
  recipe: unknown
}

export interface JobState {
  id: string
  status: 'queued' | 'running' | 'done' | 'cancelled' | 'error'
  progress: number
  message: string
  summary: EstimationSummary | null
  error: string | null
}

export interface EstimationSummary {
  attribute: string
  cancelled: boolean
  n_targeted: number
  n_estimated: number
  n_unestimated: number
  seconds: number
  table: Record<string, string | number | null>[]
}

async function errorText(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null)
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg ?? '').join('; ')
  return `Request failed (${response.status})`
}

async function call<T>(method: string, path: string, body?: unknown, raw?: { data: BodyInit }): Promise<T> {
  const response = await fetch(`${OPS}${path}`, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: raw ? raw.data : body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!response.ok) throw new Error(await errorText(response))
  return (await response.json()) as T
}

export const opsApi = {
  project: () => call<ProjectSummary>('GET', '/project'),
  /** `add`: one more file of the same table (merged on import) instead of replacing it. */
  upload: (table: string, file: File, add = false) =>
    call<UploadPreview>('PUT', `/uploads/${encodeURIComponent(table)}?filename=${encodeURIComponent(file.name)}${add ? '&add=true' : ''}`, undefined, { data: file }),
  previewUpload: (table: string, index: number, spec: SourceSpec, holeId?: HoleIdRule | null) =>
    call<UploadPreview>('POST', `/uploads/${encodeURIComponent(table)}/preview`, { index, spec, hole_id: holeId ?? undefined }),
  removeUpload: (table: string) => call<unknown>('DELETE', `/uploads/${encodeURIComponent(table)}`),
  importDb: (body: unknown) => call<ImportResult>('POST', '/import', body),
  composite: (body: unknown) => call<Record<string, unknown>>('POST', '/composite', body),
  createModel: (body: unknown) => call<Record<string, unknown>>('POST', '/models', body),
  codeModel: (id: string, body: unknown) => call<{ n_blocks: number; steps: Record<string, unknown>[] }>('POST', `/models/${id}/code`, body),
  estimate: (body: unknown) => call<JobState>('POST', '/estimate', body),
  job: (id: string) => call<JobState>('GET', `/jobs/${id}`),
  cancel: (id: string) => call<JobState>('POST', `/jobs/${id}/cancel`),
}

interface OpsState {
  summary: ProjectSummary | null
  error: string | null
  refresh: () => Promise<void>
  /** An operation changed the project: refresh the lists and reload the scene. */
  changed: () => Promise<void>
}

export const useOps = create<OpsState>()((set, get) => ({
  summary: null,
  error: null,
  refresh: async () => {
    try {
      set({ summary: await opsApi.project(), error: null })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },
  changed: async () => {
    await get().refresh()
    clearResources()
  },
}))

/** Project objects of one kind. */
export function ofKind(summary: ProjectSummary | null, kind: ProjectObjectInfo['kind']): ProjectObjectInfo[] {
  return summary?.objects.filter((o) => o.kind === kind) ?? []
}

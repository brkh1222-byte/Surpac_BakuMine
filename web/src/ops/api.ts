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

export interface UploadPreview {
  name: string
  kind: 'collar' | 'survey' | 'interval'
  file: string
  rows: number
  columns: string[]
  preview: Record<string, string>[]
  mapping: Record<string, string>
  missing: string[]
  unmatched: string[]
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

export interface ImportResult extends ValidationResult {
  id: string
  name: string
  tables: Record<string, number>
  notes: string[]
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
  upload: (table: string, file: File) =>
    call<UploadPreview>('PUT', `/uploads/${encodeURIComponent(table)}?filename=${encodeURIComponent(file.name)}`, undefined, { data: file }),
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

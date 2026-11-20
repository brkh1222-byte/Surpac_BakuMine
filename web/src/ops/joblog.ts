/**
 * Job log (P6.T7): the project's `jobs.jsonl` as served by `GET /api/v1/ops/log`, and undo
 * (`POST /api/v1/ops/undo`) of the newest model-changing job. Pure helpers for the panel.
 */

export interface LoggedObject {
  id: string
  kind: string
  name: string
  path: string
}

export interface LoggedJob {
  id: string
  time: string
  operation: string
  params: Record<string, unknown>
  inputs: LoggedObject[]
  outputs: LoggedObject[]
  status: 'done' | 'error' | 'cancelled' | 'running'
  duration_s: number | null
  summary: Record<string, unknown> | null
  error: string | null
  modifies: string | null
  snapshot: boolean
  undone: boolean
}

export interface JobLog {
  /** newest first */
  jobs: LoggedJob[]
  /** id of the job an undo would revert, or null */
  undo: string | null
  keep_snapshots: number
}

export interface UndoResult {
  undone: string
  operation: string
  model: string
}

const OPERATION_LABELS: Record<string, string> = {
  import: 'Import',
  validate: 'Validate',
  composite: 'Composite',
  surface_from_points: 'Surface from points',
  import_mesh: 'Import surface / solid',
  implicit_solid: 'Implicit solid',
  create_model: 'Create model',
  code: 'Code model',
  estimate: 'Estimate',
  report: 'Report',
  undo: 'Undo',
}

export function operationLabel(operation: string): string {
  return OPERATION_LABELS[operation] ?? operation
}

/** "0.42 s", "12.3 s", "2 min 05 s"; an em dash when not known. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  if (seconds < 10) return `${seconds.toFixed(2)} s`
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds - minutes * 60)
  return `${minutes} min ${String(rest).padStart(2, '0')} s`
}

/** Status shown for a job: an undone job says so whatever it ended with. */
export function jobStatus(job: LoggedJob): string {
  return job.undone ? 'undone' : job.status
}

/** The job the Undo button reverts, if the server offers one. */
export function undoTarget(log: JobLog | null): LoggedJob | null {
  if (!log?.undo) return null
  return log.jobs.find((j) => j.id === log.undo) ?? null
}

/** One line about what a job worked on: the undone job of an undo, else its objects. */
export function jobSubject(job: LoggedJob, all: LoggedJob[] = []): string {
  if (job.operation === 'undo') {
    const target = all.find((j) => j.id === job.params.job)
    return target ? `${operationLabel(target.operation)} ${target.id}` : String(job.params.job ?? '')
  }
  const names = (job.outputs.length ? job.outputs : job.inputs.slice(0, 1)).map((o) => o.name)
  return names.join(', ')
}

/** Parameters as indented JSON for the expanded row. */
export function formatParams(params: Record<string, unknown>): string {
  return JSON.stringify(params, null, 2)
}

async function call<T>(method: string, path: string): Promise<T> {
  const response = await fetch(`/api/v1/ops${path}`, { method })
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    const detail = (body as { detail?: unknown } | null)?.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`)
  }
  return (await response.json()) as T
}

export const jobLogApi = {
  log: () => call<JobLog>('GET', '/log'),
  undo: () => call<UndoResult>('POST', '/undo'),
}

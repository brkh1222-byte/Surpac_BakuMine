/**
 * Blast BakuMine API client. The endpoints are planned (plan_blast.md B5); until they are
 * served, every call falls back to the browser-made demo bundle and says so (`source:
 * 'demo'` plus the reason), so the tab never shows fixture numbers as if they were results.
 *
 *   GET  /api/v1/blast/result     the latest bundle of the open project (404 if none)
 *   POST /api/v1/blast/simulate   design parameters -> bundle
 *   GET  /api/v1/blast/defaults   suggested design parameters for the project
 */
import { API_BASE } from '../api/client'
import { DEFAULT_DESIGN, makeDemoBundle } from './fixture'
import { validateBundle } from './math'
import type { BlastBundle, DesignParams } from './types'

type FetchFn = typeof fetch

export interface Loaded {
  bundle: BlastBundle
  source: 'api' | 'demo'
  /** Why the demo is shown instead of an API result; null for API results. */
  reason: string | null
}

export class BlastApiError extends Error {
  /** HTTP status; 0 = not reachable or not a valid bundle. */
  readonly status: number
  constructor(message: string, status = 0) {
    super(message)
    this.status = status
  }
}

async function json(url: string, init: RequestInit | undefined, fetchFn: FetchFn): Promise<unknown> {
  let response: Response
  try {
    response = await fetchFn(url, init)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new BlastApiError('the API is not reachable')
  }
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: unknown } | null
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      // not JSON: keep the status text
    }
    throw new BlastApiError(`${response.status}${detail ? `: ${detail}` : ''}`, response.status)
  }
  try {
    return await response.json()
  } catch {
    throw new BlastApiError(`${url} did not answer with JSON`)
  }
}

function asBundle(value: unknown): BlastBundle {
  const errors = validateBundle(value)
  if (errors.length > 0) throw new BlastApiError(`the API result is not a valid bundle: ${errors.slice(0, 3).join('; ')}`)
  return value as BlastBundle
}

function demo(design: DesignParams, reason: string): Loaded {
  return { bundle: makeDemoBundle(design), source: 'demo', reason }
}

function why(error: unknown, what: string): string {
  if (error instanceof BlastApiError) {
    if (error.status === 404) return `${what}: nothing there yet (404)`
    return `${what}: ${error.message}`
  }
  return `${what}: ${error instanceof Error ? error.message : String(error)}`
}

/** The latest result of the open project, else the demo. */
export async function loadResult(design: DesignParams = DEFAULT_DESIGN, signal?: AbortSignal, fetchFn: FetchFn = fetch): Promise<Loaded> {
  try {
    return { bundle: asBundle(await json(`${API_BASE}/blast/result`, { signal }, fetchFn)), source: 'api', reason: null }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    return demo(design, why(error, 'No blast result from the API'))
  }
}

/** Runs the simulation; when the API cannot, the demo is regenerated from the same design. */
export async function simulate(design: DesignParams, signal?: AbortSignal, fetchFn: FetchFn = fetch): Promise<Loaded> {
  try {
    const value = await json(
      `${API_BASE}/blast/simulate`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(design), signal },
      fetchFn,
    )
    return { bundle: asBundle(value), source: 'api', reason: null }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    return demo(design, why(error, 'Simulation not run by the API'))
  }
}

/** Design parameters the API suggests for the project; null when it has none. */
export async function loadDefaults(signal?: AbortSignal, fetchFn: FetchFn = fetch): Promise<DesignParams | null> {
  try {
    const value = await json(`${API_BASE}/blast/defaults`, { signal }, fetchFn)
    if (typeof value !== 'object' || value === null) return null
    const v = value as Partial<DesignParams>
    return {
      ...DEFAULT_DESIGN,
      ...v,
      timing: { ...DEFAULT_DESIGN.timing, ...(v.timing ?? {}) },
      rock: { ...DEFAULT_DESIGN.rock, ...(v.rock ?? {}) },
      monitors: Array.isArray(v.monitors) ? v.monitors : [],
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    return null
  }
}

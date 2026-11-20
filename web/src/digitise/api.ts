/**
 * Client of the digitising endpoints (bakumine/api/routes_digitise.py):
 *
 *   GET  /api/v1/digitise/strings?set=      the saved strings
 *   PUT  /api/v1/digitise/strings?set=      save (replace) the set
 *   GET  /api/v1/digitise/strings.str?set=  Surpac .str download
 *   POST /api/v1/digitise/solid             solid from the set's closed strings
 */
import { API_BASE, ApiError } from '../api/client'
import type { Vec3 } from '../scene/origin'
import type { DigitString } from './edit'

export const DEFAULT_SET = 'digitised'

export interface SolidResult {
  id: string
  name: string
  valid: boolean
  watertight: boolean
  outward: boolean
  volume: number
  n_vertices: number
  n_faces: number
  n_strings: number
  n_sections: number
  skipped_open: number
  issues: string[]
}

type FetchFn = typeof fetch

/** The server's message for a failed response ("422: ..."). */
export async function errorText(response: Response): Promise<string> {
  let detail: string = response.statusText
  try {
    const body = (await response.json()) as { detail?: unknown }
    const d = body?.detail
    if (typeof d === 'string') detail = d
    else if (d && typeof d === 'object' && 'message' in d && typeof d.message === 'string') detail = d.message
    else if (Array.isArray(d)) detail = d.map((e: { msg?: string }) => e?.msg ?? String(e)).join('; ')
  } catch {
    // not JSON
  }
  return `${response.status}: ${detail}`
}

async function send(url: string, init: RequestInit, fetchFn: FetchFn): Promise<unknown> {
  let response: Response
  try {
    response = await fetchFn(url, init)
  } catch {
    throw new ApiError('The API is not reachable. Start it with `make dev`.')
  }
  if (!response.ok) throw new ApiError(await errorText(response), response.status)
  return response.json()
}

function parseStrings(body: unknown): DigitString[] {
  const list = (body as { strings?: unknown } | null)?.strings
  if (!Array.isArray(list)) throw new ApiError('Bad response from the strings endpoint')
  return list.map((s: { id: string; section: string; closed: boolean; points: number[][] }) => ({
    id: String(s.id),
    section: String(s.section ?? ''),
    closed: !!s.closed,
    points: s.points.map((p) => [p[0], p[1], p[2]] as Vec3),
  }))
}

export async function loadStrings(set = DEFAULT_SET, fetchFn: FetchFn = fetch): Promise<DigitString[]> {
  return parseStrings(await send(`${API_BASE}/digitise/strings?set=${encodeURIComponent(set)}`, {}, fetchFn))
}

/** Saves the strings (empty strings are left out); returns them as the server stored them. */
export async function saveStrings(strings: readonly DigitString[], set = DEFAULT_SET, fetchFn: FetchFn = fetch): Promise<DigitString[]> {
  const body = {
    strings: strings
      .filter((s) => s.points.length > 0)
      .map((s) => ({ id: s.id, section: s.section, closed: s.closed, points: s.points })),
  }
  const result = await send(
    `${API_BASE}/digitise/strings?set=${encodeURIComponent(set)}`,
    { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    fetchFn,
  )
  return parseStrings(result)
}

export async function buildSolid(name: string, set = DEFAULT_SET, fetchFn: FetchFn = fetch): Promise<SolidResult> {
  return (await send(
    `${API_BASE}/digitise/solid`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ set, name }) },
    fetchFn,
  )) as SolidResult
}

export function strUrl(set = DEFAULT_SET): string {
  return `${API_BASE}/digitise/strings.str?set=${encodeURIComponent(set)}`
}

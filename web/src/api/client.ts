/**
 * Typed client of the scene API (`/api/v1`, see bakumine/api/routes_scene.py).
 *
 *   fetchManifest()                -> SceneManifest         (JSON)
 *   fetchDrillholes(object, table) -> DrillholeSet          (BKMA)
 *   fetchMesh(object)              -> MeshData              (BKMA)
 *   fetchBlocks(object, query)     -> BlockSet              (BKMA; api/blocks.ts)
 *
 * Every coordinate array is WORLD float64 (x = east, y = north, z = up, metres), flat
 * xyz triples. Layers turn them into float32 with `worldToScene`; nothing here does.
 * The browser's HTTP cache revalidates payloads by ETag, so a reload is a 304.
 */
import type { Bounds, Category } from '../state/store'
import { BkmaError, decodeBkma, type BkmaPayload } from './bkma'

export const API_BASE = '/api/v1'

export interface NumericAttributeInfo {
  table: string
  name: string
  type: 'numeric'
  unit: string | null
  /** Null when the column has no values at all. */
  min: number | null
  max: number | null
  /** 2nd and 98th percentile: a robust default legend range. */
  display_min: number | null
  display_max: number | null
  count: number
  missing: number
}

export interface CategoricalAttributeInfo {
  table: string
  name: string
  type: 'categorical'
  categories: string[]
  count: number
  missing: number
}

export type AttributeInfo = NumericAttributeInfo | CategoricalAttributeInfo

export type ObjectKind = 'drillholes' | 'mesh' | 'blocks'

export interface ManifestObject {
  id: string
  name: string
  category: Category
  kind: ObjectKind
  /** Path of the data endpoint. */
  href: string
  /** World bounds; null for an object without geometry. */
  bounds: Bounds | null
  attributes: AttributeInfo[]
  stats: Record<string, unknown>
}

export interface SceneManifest {
  api_version: number
  project: { id: string; name: string }
  version: string
  objects: ManifestObject[]
}

export interface NumericSegmentAttribute {
  name: string
  type: 'numeric'
  unit: string | null
  /** One value per segment; NaN = missing. */
  values: Float64Array
}

export interface CategoricalSegmentAttribute {
  name: string
  type: 'categorical'
  categories: string[]
  /** Index into `categories` per segment; -1 = missing. */
  codes: Int32Array
}

export type SegmentAttribute = NumericSegmentAttribute | CategoricalSegmentAttribute

/** Intervals of one table, located in space. All arrays have one entry per segment. */
export interface DrillholeSegments {
  table: string
  count: number
  /** Index into `DrillholeSet.holeIds`. */
  hole: Uint32Array
  /** Row of the interval in the source table. */
  row: Uint32Array
  from: Float64Array
  to: Float64Array
  /** World xyz at `from` / `to`, 3 per segment. */
  start: Float64Array
  end: Float64Array
  attributes: SegmentAttribute[]
  /** Intervals the server could not locate on a trace. */
  dropped: number
}

export interface DrillholeSet {
  objectId: string
  holeIds: string[]
  /** Points of hole h are rows holeOffsets[h] .. holeOffsets[h + 1] - 1. */
  holeOffsets: Uint32Array
  pointDepth: Float64Array
  /** World xyz of the densified path, 3 per point. */
  pointXyz: Float64Array
  /** World xyz, 3 per hole. */
  collarXyz: Float64Array
  maxDepth: Float64Array
  segments: DrillholeSegments | null
  /** Size of the decoded payload, bytes. */
  byteLength: number
}

export interface MeshData {
  objectId: string
  /** World xyz, 3 per vertex. */
  vertices: Float64Array
  /** Vertex indices, 3 per triangle. */
  faces: Uint32Array
  /** Unit vertex normals, 3 per vertex; null when not requested. */
  normals: Float32Array | null
  closed: boolean
  byteLength: number
}

export class ApiError extends Error {
  /** HTTP status, or 0 when the server could not be reached or the payload was bad. */
  readonly status: number
  constructor(message: string, status = 0) {
    super(message)
    this.status = status
  }
}

type FetchFn = typeof fetch

export async function request(url: string, signal: AbortSignal | undefined, fetchFn: FetchFn): Promise<Response> {
  let response: Response
  try {
    response = await fetchFn(url, { signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('The API is not reachable. Start it with `make dev`.')
  }
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body: unknown = await response.json()
      const text = (body as { detail?: unknown } | null)?.detail
      if (typeof text === 'string') detail = text
    } catch {
      // not JSON: keep the status text
    }
    // The dev proxy answers 500/502/504 when uvicorn is down.
    const unreachable = response.status >= 500 && !detail
    throw new ApiError(
      unreachable ? 'The API is not reachable. Start it with `make dev`.' : `${response.status}: ${detail}`,
      response.status,
    )
  }
  return response
}

export async function requestBkma(url: string, kind: string, signal: AbortSignal | undefined, fetchFn: FetchFn) {
  const response = await request(url, signal, fetchFn)
  try {
    const payload = decodeBkma(await response.arrayBuffer())
    if (payload.kind !== kind) throw new BkmaError(`expected a "${kind}" payload, got "${payload.kind}"`)
    return payload
  } catch (error) {
    if (error instanceof BkmaError) throw new ApiError(`Bad response from ${url}: ${error.message}`)
    throw error
  }
}

export async function fetchManifest(signal?: AbortSignal, fetchFn: FetchFn = fetch): Promise<SceneManifest> {
  const response = await request(`${API_BASE}/scene`, signal, fetchFn)
  const body: unknown = await response.json().catch(() => null)
  const manifest = body as Partial<SceneManifest> | null
  if (!manifest || !Array.isArray(manifest.objects) || !manifest.project) {
    throw new ApiError('Bad response from the scene manifest endpoint')
  }
  return manifest as SceneManifest
}

export function stringList(value: unknown, what: string): string[] {
  if (!Array.isArray(value) || !value.every((v) => typeof v === 'string')) {
    throw new BkmaError(`${what} is not a list of strings`)
  }
  return value as string[]
}

/** Turns a decoded "drillholes" payload into the typed structure. */
export function decodeDrillholes(payload: BkmaPayload): DrillholeSet {
  const meta = payload.meta
  const holeIds = stringList(meta.hole_ids, 'meta.hole_ids')
  const set: DrillholeSet = {
    objectId: String(meta.object_id ?? ''),
    holeIds,
    holeOffsets: payload.get('hole_offsets', 'u4'),
    pointDepth: payload.get('point_depth', 'f8'),
    pointXyz: payload.get('point_xyz', 'f8', 3),
    collarXyz: payload.get('collar_xyz', 'f8', 3),
    maxDepth: payload.get('max_depth', 'f8'),
    segments: null,
    byteLength: payload.byteLength,
  }
  if (set.holeOffsets.length !== holeIds.length + 1 || set.maxDepth.length !== holeIds.length) {
    throw new BkmaError('per-hole arrays do not match meta.hole_ids')
  }
  if (typeof meta.table === 'string') {
    const hole = payload.get('seg_hole', 'u4')
    const attributes: SegmentAttribute[] = []
    for (const raw of Array.isArray(meta.attributes) ? (meta.attributes as Record<string, unknown>[]) : []) {
      const name = String(raw.name)
      const array = String(raw.array)
      if (raw.type === 'numeric') {
        attributes.push({
          name,
          type: 'numeric',
          unit: typeof raw.unit === 'string' ? raw.unit : null,
          values: payload.get(array, 'f8'),
        })
      } else {
        attributes.push({
          name,
          type: 'categorical',
          categories: stringList(raw.categories, `categories of ${name}`),
          codes: payload.get(array, 'i4'),
        })
      }
    }
    set.segments = {
      table: meta.table,
      count: hole.length,
      hole,
      row: payload.get('seg_row', 'u4'),
      from: payload.get('seg_from', 'f8'),
      to: payload.get('seg_to', 'f8'),
      start: payload.get('seg_start', 'f8', 3),
      end: payload.get('seg_end', 'f8', 3),
      attributes,
      dropped: typeof meta.dropped_segments === 'number' ? meta.dropped_segments : 0,
    }
  }
  return set
}

/** Turns a decoded "mesh" payload into the typed structure. */
export function decodeMesh(payload: BkmaPayload): MeshData {
  return {
    objectId: String(payload.meta.object_id ?? ''),
    vertices: payload.get('vertices', 'f8', 3),
    faces: payload.get('faces', 'u4', 3),
    normals: payload.has('normals') ? payload.get('normals', 'f4', 3) : null,
    closed: payload.meta.closed === true,
    byteLength: payload.byteLength,
  }
}

export function wrapDecode<T>(url: string, decode: () => T): T {
  try {
    return decode()
  } catch (error) {
    if (error instanceof BkmaError) throw new ApiError(`Bad response from ${url}: ${error.message}`)
    throw error
  }
}

/** Traces and collars and, with `table`, the located intervals of that table. */
export async function fetchDrillholes(
  object: Pick<ManifestObject, 'href'>,
  table: string | null,
  signal?: AbortSignal,
  fetchFn: FetchFn = fetch,
): Promise<DrillholeSet> {
  const url = table ? `${object.href}?table=${encodeURIComponent(table)}` : object.href
  const payload = await requestBkma(url, 'drillholes', signal, fetchFn)
  return wrapDecode(url, () => decodeDrillholes(payload))
}

export async function fetchMesh(
  object: Pick<ManifestObject, 'href'>,
  signal?: AbortSignal,
  fetchFn: FetchFn = fetch,
): Promise<MeshData> {
  const url = `${object.href}?normals=1`
  const payload = await requestBkma(url, 'mesh', signal, fetchFn)
  return wrapDecode(url, () => decodeMesh(payload))
}

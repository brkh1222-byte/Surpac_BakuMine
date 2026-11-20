/**
 * Block models over the scene API (bakumine/api/blocks.py):
 *
 *   fetchBlocks(object, { attributes, where, lod, clip })  -> BlockSet     (BKMA)
 *   fetchBlockDetail(object, row)                           -> BlockDetail  (JSON)
 *   fetchBlockInfo(objectId)                                -> BlockManifestObject
 *
 * A BlockSet carries no coordinates: `structure` (parent, level, sub) and the model
 * `definition` give centroids and sizes through blocks/structure.ts. `where` and `clip`
 * are constraints - the JSON of bakumine.core.blockmodel.constraints or a short
 * attribute expression - evaluated by the server; an invalid one is an ApiError with
 * status 400 whose message is the engine's.
 */
import { denseStructure, parseDefinition, type BlockDefinition, type BlockStructure } from '../blocks/structure'
import { BkmaError, type BkmaPayload } from './bkma'
import { API_BASE, ApiError, request, requestBkma, stringList, wrapDecode, type ManifestObject } from './client'

export interface BlockNumericInfo {
  table: string
  name: string
  type: 'numeric'
  unit: string | null
  min: number | null
  max: number | null
  display_min: number | null
  display_max: number | null
  percentiles: Record<string, number> | null
  count: number
  missing: number
  description?: string
}

export interface BlockCategoricalInfo {
  table: string
  name: string
  type: 'categorical'
  categories: string[]
  count: number
  missing: number
  description?: string
}

export type BlockAttributeInfo = BlockNumericInfo | BlockCategoricalInfo

export interface NamedGeometry {
  id: string
  name: string
}

/** Manifest entry of a block model (`kind: "blocks"`). */
export interface BlockManifestObject extends Omit<ManifestObject, 'attributes'> {
  kind: 'blocks'
  attributes: BlockAttributeInfo[]
  /** Raw definition; parse with `parseDefinition`. */
  definition: unknown
  /** Solids and surfaces a constraint may name (by id). */
  constraint_geometry?: { solids: NamedGeometry[]; surfaces: NamedGeometry[] }
}

export type BlockLod = 'auto' | 'full' | 'shell'

export interface BlockQuery {
  /** Attribute columns to send. */
  attributes: string[]
  /** Constraint JSON text or attribute expression; null = every block. */
  where: string | null
  lod: BlockLod
  /** Constraint that limits what is sent (the section slab); counts still follow `where`. */
  clip: string | null
}

export interface BlockNumericColumn {
  name: string
  type: 'numeric'
  unit: string | null
  /** NaN = missing. */
  values: Float32Array
}

export interface BlockCategoricalColumn {
  name: string
  type: 'categorical'
  categories: string[]
  /** -1 = missing. */
  codes: Int16Array | Int32Array
}

export type BlockColumn = BlockNumericColumn | BlockCategoricalColumn

export interface BlockSet {
  objectId: string
  definition: BlockDefinition
  /** Blocks in this set. */
  count: number
  /** Blocks in the whole model. */
  total: number
  /** Blocks the `where` constraint selects (>= count when a shell or a clip applies). */
  selected: number
  /** Volume of the selected blocks, m3. */
  selectedVolume: number
  /** What was applied: the exposed shell of the selection, or all of it. */
  lod: 'full' | 'shell'
  clipped: boolean
  /** Row of each block in the model's table; null = all rows, in order. */
  rows: Uint32Array | null
  structure: BlockStructure
  columns: BlockColumn[]
  bytesPerBlock: number
  byteLength: number
}

function count(meta: Record<string, unknown>, key: string): number {
  const value = meta[key]
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new BkmaError(`meta.${key} is not a count`)
  return value
}

/** Turns a decoded "blocks" payload into the typed structure. */
export function decodeBlocks(payload: BkmaPayload): BlockSet {
  const meta = payload.meta
  const n = count(meta, 'n_blocks')
  let definition: BlockDefinition
  try {
    definition = parseDefinition(meta.definition)
  } catch (error) {
    throw new BkmaError(error instanceof Error ? error.message : String(error))
  }
  const rows = meta.rows === 'all' ? null : payload.get('row', 'u4')
  if (rows && rows.length !== n) throw new BkmaError('row does not match meta.n_blocks')
  let structure: BlockStructure
  if (meta.structure === 'dense') {
    structure = denseStructure(rows, n)
  } else {
    structure = { count: n, parent: payload.get('parent', 'u4'), level: payload.get('level', 'u1'), sub: payload.get('sub', 'u1', 3) }
    if (structure.parent.length !== n || structure.level.length !== n || structure.sub.length !== n * 3) {
      throw new BkmaError('structure arrays do not match meta.n_blocks')
    }
  }
  const columns: BlockColumn[] = []
  for (const raw of Array.isArray(meta.attributes) ? (meta.attributes as Record<string, unknown>[]) : []) {
    const name = String(raw.name)
    const array = String(raw.array)
    if (raw.type === 'numeric') {
      const values = payload.get(array, 'f4')
      if (values.length !== n) throw new BkmaError(`attribute ${name} does not match meta.n_blocks`)
      columns.push({ name, type: 'numeric', unit: typeof raw.unit === 'string' ? raw.unit : null, values })
    } else {
      const codes = payload.entry(array).dtype === 'i4' ? payload.get(array, 'i4') : payload.get(array, 'i2')
      if (codes.length !== n) throw new BkmaError(`attribute ${name} does not match meta.n_blocks`)
      columns.push({ name, type: 'categorical', categories: stringList(raw.categories, `categories of ${name}`), codes })
    }
  }
  return {
    objectId: String(meta.object_id ?? ''),
    definition,
    count: n,
    total: count(meta, 'n_total'),
    selected: count(meta, 'n_selected'),
    selectedVolume: typeof meta.selected_volume === 'number' ? meta.selected_volume : 0,
    lod: meta.lod === 'shell' ? 'shell' : 'full',
    clipped: meta.clipped === true,
    rows,
    structure,
    columns,
    bytesPerBlock: typeof meta.bytes_per_block === 'number' ? meta.bytes_per_block : 0,
    byteLength: payload.byteLength,
  }
}

/** The query string of a block request (also its cache key). */
export function blockQueryString(query: BlockQuery): string {
  const params = new URLSearchParams()
  if (query.attributes.length > 0) params.set('attributes', query.attributes.join(','))
  if (query.where) params.set('where', query.where)
  if (query.lod !== 'auto') params.set('lod', query.lod)
  if (query.clip) params.set('clip', query.clip)
  return params.toString()
}

export async function fetchBlocks(
  object: Pick<ManifestObject, 'href'>,
  query: BlockQuery,
  signal?: AbortSignal,
  fetchFn: typeof fetch = fetch,
): Promise<BlockSet> {
  const text = blockQueryString(query)
  const url = text ? `${object.href}?${text}` : object.href
  const payload = await requestBkma(url, 'blocks', signal, fetchFn)
  return wrapDecode(url, () => decodeBlocks(payload))
}

/** One block as the server knows it (full precision; null = missing). */
export interface BlockDetail {
  row: number
  /** `BlockModel.block_ids()` value, decimal text (int64). */
  block_id: string
  ijk: [number, number, number]
  parent: number
  level: number
  sub: [number, number, number]
  centroid: [number, number, number]
  size: [number, number, number]
  volume: number
  attributes: Record<string, number | string | null>
}

export async function fetchBlockDetail(
  object: Pick<ManifestObject, 'href'>,
  row: number,
  signal?: AbortSignal,
  fetchFn: typeof fetch = fetch,
): Promise<BlockDetail> {
  const response = await request(`${object.href}/detail?row=${row}`, signal, fetchFn)
  const body = (await response.json().catch(() => null)) as BlockDetail | null
  if (!body || typeof body.row !== 'number' || !body.attributes) throw new ApiError('Bad response from the block detail endpoint')
  return body
}

/** The manifest entry of one block model (used for generated `perf-<n>` models). */
export async function fetchBlockInfo(objectId: string, signal?: AbortSignal, fetchFn: typeof fetch = fetch): Promise<BlockManifestObject> {
  const response = await request(`${API_BASE}/scene/objects/${encodeURIComponent(objectId)}/blocks/info`, signal, fetchFn)
  const body = (await response.json().catch(() => null)) as BlockManifestObject | null
  if (!body || body.kind !== 'blocks' || !body.definition) throw new ApiError('Bad response from the block info endpoint')
  return body
}

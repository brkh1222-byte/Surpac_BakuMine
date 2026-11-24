/**
 * Client of the geometry operation endpoints (bakumine/api/routes_geometry.py): surfaces
 * from points or from elevation files, and import of a surface / solid file.
 */

const OPS = '/api/v1/ops'

/** What an uploaded file holds; `roles` says what it can be used for. */
export interface GeometryFile {
  file: string
  roles: ('points' | 'strings' | 'mesh' | 'raster')[]
  /** points table */
  columns?: string[]
  rows?: number
  /** the x, y, z columns found by header name, or null */
  xyz?: string[] | null
  skipped?: number
  n_strings?: number
  n_string_points?: number
  n_vertices?: number
  n_faces?: number
  /** a companion file that must be uploaded with it (the .str of a .dtm) */
  needs?: string
  /** elevation grid: columns x rows, cell size, CRS (null = not in the file), height range */
  cells?: number[]
  cell?: number[]
  crs?: string | null
  geographic?: boolean
  heights?: number[] | null
}

/** Grid laid over the area (the summary of `grid_from_rasters`). */
export interface RasterGrid {
  nodes: number[]
  spacing: number
  rectangle: number[]
  nodes_with_height: number
  nodes_without_height: number
}

/** Collar z - surface z after the shift. */
export interface CollarCheck {
  collars: number
  on_surface: number
  off_surface: string[]
  tolerance: number
  median: number | null
  mean: number | null
  spread: { std: number; p90_abs: number; max_abs: number } | null
  suspect: { hole_id: string; dz: number; off_median: number }[]
}

export interface RasterSurfaceResult {
  id: string
  name: string
  kind: 'dtm'
  crs: string | null
  grid: RasterGrid
  notes: string[]
  z_shift: number
  collars: CollarCheck | null
  n_vertices: number
  n_faces: number
  bounds: number[][]
}

export interface GeometryUpload {
  token: string
  file: string
  files: GeometryFile[]
}

export interface SurfaceResult {
  id: string
  name: string
  sources: Record<string, number>
  n_points: number
  n_breaklines: number
  n_duplicates: number
  n_conflicts: number
  duplicates: string
  n_extension_points: number
  rectangle: number[] | null
  extend_z: number | 'nearest' | null
  n_vertices: number
  n_faces: number
  plan_area: number
  bounds: number[][]
  /** with an elevation file: the DEM nodes around the data */
  dem?: {
    nodes: number
    kept: number
    dropped: number
    drop_within: number
    z_shift: number
    tie: { points: number; median: number; p90_abs: number; max_abs: number } | null
    crs: string | null
    notes: string[]
  }
}

export interface MeshImportResult {
  id: string
  name: string
  kind: 'dtm' | 'solid'
  source: string
  closed: boolean
  n_vertices: number
  n_faces: number
  valid?: boolean
  volume?: number | null
  issues?: string[]
  /** what the file itself limits (a binary STL in large coordinates) */
  warnings?: string[]
  scale?: number
}

async function errorText(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null)
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg ?? '').join('; ')
  return `Request failed (${response.status})`
}

async function send<T>(method: string, path: string, body: BodyInit | undefined, json: boolean): Promise<T> {
  const response = await fetch(`${OPS}${path}`, { method, headers: json ? { 'Content-Type': 'application/json' } : undefined, body })
  if (!response.ok) throw new Error(await errorText(response))
  return (await response.json()) as T
}

/** POST a JSON body to an operation endpoint; throws the server's message on failure. */
export const postJson = <T>(path: string, body: unknown) => send<T>('POST', path, JSON.stringify(body), true)

export const geometryApi = {
  /** Uploads one file; `token` adds it to an earlier upload (the .str of a .dtm). */
  upload: (file: File, token?: string) =>
    send<GeometryUpload>('PUT', `/geometry/uploads?filename=${encodeURIComponent(file.name)}${token ? `&token=${token}` : ''}`, file, false),
  surface: (body: unknown) => postJson<SurfaceResult>('/surfaces/from-points', body),
  surfaceFromRaster: (body: unknown) => postJson<RasterSurfaceResult>('/surfaces/from-raster', body),
  importMesh: (body: unknown) => postJson<MeshImportResult>('/meshes/import', body),
}

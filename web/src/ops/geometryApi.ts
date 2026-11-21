/**
 * Client of the geometry operation endpoints (bakumine/api/routes_geometry.py): surfaces
 * from points and import of a surface / solid file.
 */

const OPS = '/api/v1/ops'

/** What an uploaded file holds; `roles` says what it can be used for. */
export interface GeometryFile {
  file: string
  roles: ('points' | 'strings' | 'mesh')[]
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
  importMesh: (body: unknown) => postJson<MeshImportResult>('/meshes/import', body),
}

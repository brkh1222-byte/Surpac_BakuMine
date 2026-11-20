/**
 * Request body of the surface dialog (pure, unit-tested); the endpoint is in
 * bakumine/api/routes_geometry.py.
 */
import { nums } from './builders'

/** A source of points or strings: a project object id, or an uploaded file. */
export type SourceRef = string | { upload: string; file: string }

export interface PointsSource {
  upload: string
  file: string
  /** "x, y, z" column names; blank = found by header name. */
  columns: string
}

export interface SurfaceForm {
  name: string
  /** Drillhole databases whose collars are used (object ids). */
  databases: string[]
  points: PointsSource[]
  /** String files used as points. */
  strings: SourceRef[]
  /** String files whose segments become triangle edges. */
  breaklines: SourceRef[]
  extendMode: 'none' | 'padding' | 'model' | 'rectangle'
  padding: string
  /** "xmin, ymin, xmax, ymax" */
  rectangle: string
  model: string
  /** Elevation of the added boundary points; blank = that of the nearest data point. */
  extendZ: string
  duplicates: string
}

export function buildSurfaceRequest(f: SurfaceForm): Record<string, unknown> | string {
  if (!f.name.trim()) return 'Give the surface a name'
  if (f.databases.length + f.points.length + f.strings.length + f.breaklines.length === 0) return 'Pick at least one source of points'
  const body: Record<string, unknown> = { name: f.name.trim(), duplicates: f.duplicates || 'mean' }
  if (f.databases.length) body.databases = f.databases
  if (f.points.length) {
    const points: Record<string, unknown>[] = []
    for (const p of f.points) {
      const item: Record<string, unknown> = { upload: p.upload, file: p.file }
      if (p.columns.trim()) {
        const columns = p.columns.split(',').map((c) => c.trim()).filter(Boolean)
        if (columns.length !== 3) return `${p.file}: name three columns (x, y, z)`
        item.columns = columns
      }
      points.push(item)
    }
    body.points = points
  }
  if (f.strings.length) body.strings = f.strings
  if (f.breaklines.length) body.breaklines = f.breaklines
  if (f.extendMode !== 'none') {
    const extend: Record<string, unknown> = { mode: f.extendMode }
    const padding = f.padding.trim() === '' ? 0 : Number(f.padding)
    if (!Number.isFinite(padding) || padding < 0) return 'Padding: a distance in metres'
    if (f.extendMode === 'padding' && !(padding > 0)) return 'Padding must be positive'
    if (padding > 0) extend.padding = padding
    if (f.extendMode === 'model') {
      if (!f.model) return 'Pick the block model to extend to'
      extend.model = f.model
    }
    if (f.extendMode === 'rectangle') {
      const r = nums(f.rectangle)
      if (!r || r.length !== 4 || !(r[2] > r[0]) || !(r[3] > r[1])) return 'Rectangle: xmin, ymin, xmax, ymax'
      extend.rectangle = r
    }
    if (f.extendZ.trim() === '' || f.extendZ.trim().toLowerCase() === 'nearest') extend.z = 'nearest'
    else if (Number.isFinite(Number(f.extendZ))) extend.z = Number(f.extendZ)
    else return 'Boundary elevation: a number, or blank for the nearest point'
    body.extend = extend
  }
  return body
}

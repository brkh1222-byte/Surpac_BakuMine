/**
 * Request bodies of the surface dialogs (pure, unit-tested); the endpoints are in
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
  /** Elevation files (DEM) that fill the area around the data. */
  dem?: DemForm
}

/** Uploaded elevation grids and how they are read. */
export interface DemForm {
  rasters: { upload: string; file: string }[]
  /** CRS of files that do not say it; blank = from the file. */
  rasterCrs: string
  /** Projected CRS of the model; blank = that of the database or of the files. */
  crs: string
  /** Grid spacing in metres; blank = the cell size of the file. */
  spacing: string
  /** 'median' (tie to the data), 'none' or metres. */
  zShift: string
  /** DEM nodes this close to the data are left out; blank = the grid spacing. */
  dropWithin?: string
}

/** The DEM part of a request: the server's names; a string is what is wrong. */
function demBlock(d: DemForm): Record<string, unknown> | string {
  if (d.rasters.length === 0) return 'Upload an elevation file (.tif or .asc)'
  const block: Record<string, unknown> = { rasters: d.rasters.map((r) => ({ upload: r.upload, file: r.file })) }
  if (d.rasterCrs.trim()) block.raster_crs = d.rasterCrs.trim()
  if (d.crs.trim()) block.crs = d.crs.trim()
  if (d.spacing.trim()) {
    const spacing = Number(d.spacing)
    if (!(Number.isFinite(spacing) && spacing > 0)) return 'Grid spacing: a positive distance in metres'
    block.spacing = spacing
  }
  const shift = d.zShift.trim().toLowerCase()
  if (shift === '' || shift === 'median' || shift === 'none') block.z_shift = shift || 'median'
  else if (Number.isFinite(Number(shift))) block.z_shift = Number(shift)
  else return 'Vertical shift: median, none or metres'
  if (d.dropWithin !== undefined && d.dropWithin.trim()) {
    const drop = Number(d.dropWithin)
    if (!(Number.isFinite(drop) && drop >= 0)) return 'Drop within: a distance in metres'
    block.drop_within = drop
  }
  return block
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
    if (f.dem && extend.z !== 'nearest') return 'The boundary elevation does not apply with an elevation file'
    body.extend = extend
  }
  if (f.dem) {
    const block = demBlock(f.dem)
    if (typeof block === 'string') return block
    body.raster = block
  }
  return body
}

export interface RasterForm extends DemForm {
  name: string
  /** Database whose collars tie the heights and give the area; '' = none. */
  database: string
  areaMode: 'collars' | 'model' | 'rectangle'
  model: string
  /** "xmin, ymin, xmax, ymax" */
  rectangle: string
  padding: string
  /** Collars further than this from the shifted surface are listed (metres). */
  tolerance: string
}

/** Body of POST /ops/surfaces/from-raster (`bakumine surface from-raster`). */
export function buildRasterRequest(f: RasterForm): Record<string, unknown> | string {
  if (!f.name.trim()) return 'Give the surface a name'
  const block = demBlock({ ...f, dropWithin: undefined })
  if (typeof block === 'string') return block
  if (!f.database) {
    if (f.areaMode === 'collars') return 'Pick the drillhole database whose collars give the area'
    if (block.z_shift === 'median') return 'A median shift needs the collars: pick a database, or set the shift to none or metres'
  }
  const area: Record<string, unknown> = { mode: f.areaMode }
  if (f.areaMode === 'model') {
    if (!f.model) return 'Pick the block model whose extent is the area'
    area.model = f.model
  }
  if (f.areaMode === 'rectangle') {
    const r = nums(f.rectangle)
    if (!r || r.length !== 4 || !(r[2] > r[0]) || !(r[3] > r[1])) return 'Rectangle: xmin, ymin, xmax, ymax'
    area.rectangle = r
  }
  const padding = f.padding.trim() === '' ? 0 : Number(f.padding)
  if (!Number.isFinite(padding) || padding < 0) return 'Padding: a distance in metres'
  if (padding > 0) area.padding = padding
  const body: Record<string, unknown> = { name: f.name.trim(), ...block, area }
  if (f.database) body.database = f.database
  if (f.tolerance.trim()) {
    const t = Number(f.tolerance)
    if (!(Number.isFinite(t) && t > 0)) return 'Collar tolerance: a positive distance in metres'
    body.collar_tolerance = t
  }
  return body
}

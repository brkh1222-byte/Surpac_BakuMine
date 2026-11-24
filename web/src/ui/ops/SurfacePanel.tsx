/**
 * Surfaces dialog: build a topography (DTM) from points - drillhole collars, CSV / XLSX point
 * tables and string files, in any combination, with optional breaklines, an extension to a
 * rectangle and an elevation file (DEM) around the data - or from an elevation file alone
 * (GeoTIFF / ASCII grid: satellite DEM, drone or LiDAR grid) tied to the collars, and import
 * an existing surface or solid file (DXF / OBJ / PLY / Surpac .dtm). Runs the same code as
 * `bakumine surface from-points`, `surface from-raster` and `surface import`; the result is
 * added to the project and shows in the object tree.
 */
import { useState } from 'react'
import { ofKind, useOps } from '../../ops/api'
import { geometryApi, type CollarCheck, type GeometryFile, type GeometryUpload } from '../../ops/geometryApi'
import { buildRasterRequest, buildSurfaceRequest, type RasterForm, type SourceRef, type SurfaceForm } from '../../ops/geometryBuilders'
import { Check, Error, Panel, Select, Text, message } from './common'

type StringUse = '' | 'points' | 'breaklines'

interface UploadedSource {
  token: string
  info: GeometryFile
  /** points table: "x, y, z" column names when they are not found by header name */
  columns: string
  /** string file: how it is used */
  use: StringUse
}

const USES = [{ value: 'points', label: 'points' }, { value: 'breaklines', label: 'breaklines' }, { value: '', label: 'not used' }]
const fmt = (value: number, digits = 1) => value.toLocaleString('en-US', { maximumFractionDigits: digits })
const signed = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}`
const SHIFTS = [{ value: 'median', label: 'median of the collars (tie)' }, { value: 'none', label: 'none' }, { value: 'metres', label: 'metres…' }]

/** One line on an uploaded elevation grid. */
const describeRaster = (f: GeometryFile) =>
  `${f.file}: ${f.cells?.join(' x ')} cells of ${f.cell?.map((c) => fmt(c, 6)).join(' x ')}, CRS ${f.crs ?? 'not in the file'}${f.heights ? `, heights ${fmt(f.heights[0])} to ${fmt(f.heights[1])}` : ''}`

/** The collar check of a surface in words. */
function collarText(c: CollarCheck | null): string {
  if (!c) return ''
  if (c.median === null || !c.spread) return ` None of the ${c.collars} collars lies on the surface.`
  let text = ` Collars: ${c.on_surface} of ${c.collars} on the surface; after the shift 90 % within ${fmt(c.spread.p90_abs, 2)} m, worst ${fmt(c.spread.max_abs, 2)} m.`
  if (c.suspect.length) {
    const names = c.suspect.slice(0, 10).map((s) => `${s.hole_id} (${signed(s.off_median)} m)`).join(', ')
    text += ` ${c.suspect.length} collar(s) differ by more than ${c.tolerance} m: ${names}${c.suspect.length > 10 ? ' …' : ''}. Check their heights, or a dump or pit made after the elevation file.`
  }
  return text
}

/** Uploads elevation files (one upload each). */
async function uploadRasters(files: FileList | null, onError: (text: string) => void): Promise<UploadedSource[]> {
  const out: UploadedSource[] = []
  for (const file of Array.from(files ?? [])) {
    try {
      const up = await geometryApi.upload(file)
      const info = up.files.find((f) => f.file === up.file) ?? up.files[0]
      if (!info.roles.includes('raster')) onError(`${file.name}: not an elevation grid`)
      else out.push({ token: up.token, info, columns: '', use: '' })
    } catch (e) {
      onError(message(e))
    }
  }
  return out
}

function FromPoints() {
  const summary = useOps((s) => s.summary)
  const databases = ofKind(summary, 'database')
  const stringObjects = ofKind(summary, 'string')
  const models = ofKind(summary, 'model')
  const [collars, setCollars] = useState<Record<string, boolean>>({})
  const [objectUse, setObjectUse] = useState<Record<string, StringUse>>({})
  const [uploads, setUploads] = useState<UploadedSource[]>([])
  const [extendMode, setExtendMode] = useState<SurfaceForm['extendMode']>('none')
  const [padding, setPadding] = useState('100')
  const [rectangle, setRectangle] = useState('')
  const [modelId, setModelId] = useState('')
  const [extendZ, setExtendZ] = useState('')
  const [duplicates, setDuplicates] = useState('mean')
  const [name, setName] = useState('Topography')
  const [demCrs, setDemCrs] = useState('')
  const [dropWithin, setDropWithin] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const rasters = uploads.filter((u) => u.info.roles.includes('raster'))

  // a single database is used unless it is unticked
  const useCollars = (id: string) => collars[id] ?? databases.length === 1
  const patchUpload = (i: number, p: Partial<UploadedSource>) => setUploads((list) => list.map((u, j) => (j === i ? { ...u, ...p } : u)))

  const addFiles = async (files: FileList | null) => {
    if (!files) return
    setError(null)
    for (const file of Array.from(files)) {
      try {
        const up: GeometryUpload = await geometryApi.upload(file)
        const info = up.files.find((f) => f.file === up.file) ?? up.files[0]
        if (!info.roles.includes('points') && !info.roles.includes('strings') && !info.roles.includes('raster')) {
          setError(`${file.name}: no points or strings found (for a mesh use "Import a file" below)`)
          continue
        }
        setUploads((list) => [...list, { token: up.token, info, columns: '', use: info.roles.includes('strings') ? 'points' : '' }])
      } catch (e) {
        setError(message(e))
      }
    }
  }

  const run = async () => {
    const strings: SourceRef[] = []
    const breaklines: SourceRef[] = []
    for (const o of stringObjects) {
      if (objectUse[o.id] === 'points') strings.push(o.id)
      if (objectUse[o.id] === 'breaklines') breaklines.push(o.id)
    }
    for (const u of uploads) {
      if (!u.info.roles.includes('strings')) continue
      if (u.use === 'points') strings.push({ upload: u.token, file: u.info.file })
      if (u.use === 'breaklines') breaklines.push({ upload: u.token, file: u.info.file })
    }
    const tables = uploads.filter((u) => u.info.roles.includes('points'))
    const unmapped = tables.find((u) => !u.info.xyz && !u.columns.trim())
    if (unmapped) return setError(`${unmapped.info.file}: name the x, y, z columns`)
    const body = buildSurfaceRequest({
      name,
      databases: databases.filter((d) => useCollars(d.id)).map((d) => d.id),
      points: tables.map((u) => ({ upload: u.token, file: u.info.file, columns: u.columns })),
      strings,
      breaklines,
      extendMode,
      padding,
      rectangle,
      model: modelId || models[0]?.id || '',
      extendZ,
      duplicates,
      dem: rasters.length
        ? { rasters: rasters.map((u) => ({ upload: u.token, file: u.info.file })), rasterCrs: '', crs: demCrs, spacing: '', zShift: 'median', dropWithin }
        : undefined,
    })
    if (typeof body === 'string') return setError(body)
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await geometryApi.surface(body)
      const parts = Object.entries(r.sources).map(([what, n]) => `${n} ${what}`)
      if (r.n_breaklines) parts.push(`${r.n_breaklines} breaklines`)
      let text = `Saved “${r.name}”: ${fmt(r.n_faces, 0)} triangles from ${parts.join(', ')}; plan area ${fmt(r.plan_area, 0)} m², Z ${fmt(r.bounds[0][2])} to ${fmt(r.bounds[1][2])}.`
      if (r.n_extension_points) text += ` Extended with ${r.n_extension_points} boundary points at ${r.extend_z === 'nearest' ? 'the elevation of the nearest data point' : `elevation ${r.extend_z}`}.`
      if (r.n_duplicates) text += ` ${r.n_duplicates} duplicate points merged${r.n_conflicts ? ` (${r.n_conflicts} with different Z: ${r.duplicates})` : ''}.`
      if (r.dem) {
        text += ` Elevation file: ${fmt(r.dem.kept, 0)} of ${fmt(r.dem.nodes, 0)} grid nodes added (${fmt(r.dem.dropped, 0)} within ${fmt(r.dem.drop_within)} m of the data left out), shifted ${signed(r.dem.z_shift)} m`
        text += r.dem.tie ? `; at ${r.dem.tie.points} data points 90 % within ${fmt(r.dem.tie.p90_abs, 2)} m, worst ${fmt(r.dem.tie.max_abs, 2)} m.` : '.'
      }
      setResult(text)
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h4 className="ops-heading">Topography from points</h4>
      {databases.map((d) => (
        <Check key={d.id} label={`Collars of ${d.name}`} id={`surface-db-${d.id}`} checked={useCollars(d.id)} onChange={(v) => setCollars((c) => ({ ...c, [d.id]: v }))} />
      ))}
      {stringObjects.map((o) => (
        <Select key={o.id} label={o.name} id={`surface-string-${o.id}`} value={objectUse[o.id] ?? ''} onChange={(v) => setObjectUse((u) => ({ ...u, [o.id]: v as StringUse }))} options={[...USES].reverse()} />
      ))}
      <label className="control">
        <span>Files</span>
        <input type="file" multiple accept=".csv,.txt,.xlsx,.str,.dxf,.tif,.tiff,.asc" data-testid="surface-files" title="CSV / XLSX tables of points, .str or .dxf strings, .tif / .asc elevation grids (used around the data)" onChange={(e) => { void addFiles(e.target.files); e.target.value = '' }} />
      </label>
      {uploads.map((u, i) => (
        <div key={`${u.token}-${u.info.file}`} className="ops-block" data-testid={`surface-upload-${i}`}>
          <span className="sidebar-note">
            {u.info.roles.includes('raster') ? `${describeRaster(u.info)}; fills the area around the data` : `${u.info.file}: ${u.info.roles.includes('points') ? `${u.info.rows} rows${u.info.xyz ? ` (${u.info.xyz.join(', ')})` : ''}` : `${u.info.n_strings} strings, ${u.info.n_string_points} points`}`}
          </span>
          {u.info.roles.includes('points') && <Text label="Columns" id={`surface-upload-${i}-columns`} value={u.columns} onChange={(v) => patchUpload(i, { columns: v })} placeholder={u.info.xyz ? u.info.xyz.join(', ') : 'x, y, z column names'} list={`surface-columns-${i}`} />}
          {u.info.columns && <datalist id={`surface-columns-${i}`}>{u.info.columns.map((c) => <option key={c} value={c} />)}</datalist>}
          {u.info.roles.includes('strings') && <Select label="Use as" id={`surface-upload-${i}-use`} value={u.use} onChange={(v) => patchUpload(i, { use: v as StringUse })} options={USES} />}
          <div className="control"><button type="button" onClick={() => setUploads((list) => list.filter((_, j) => j !== i))}>Remove</button></div>
        </div>
      ))}
      <Select label="Extend" id="surface-extend" value={extendMode} onChange={(v) => setExtendMode(v as SurfaceForm['extendMode'])} options={[{ value: 'none', label: 'no (data only)' }, { value: 'padding', label: 'box of the data + padding' }, { value: 'model', label: 'block model extent' }, { value: 'rectangle', label: 'rectangle' }]} />
      {extendMode === 'model' && <Select label="Model" id="surface-extend-model" value={modelId || models[0]?.id || ''} onChange={setModelId} options={models.map((m) => ({ value: m.id, label: m.name }))} />}
      {extendMode === 'rectangle' && <Text label="Rectangle" id="surface-rectangle" value={rectangle} onChange={setRectangle} placeholder="xmin, ymin, xmax, ymax" />}
      {extendMode !== 'none' && (
        <>
          <Text label="Padding (m)" id="surface-padding" value={padding} onChange={setPadding} />
          {rasters.length === 0 && <Text label="Boundary Z" id="surface-extend-z" value={extendZ} onChange={setExtendZ} placeholder="nearest point" />}
          {rasters.length === 0 && <p className="sidebar-note">Points are added on the rectangle (corners and along the sides, outside the data only), at the given elevation or, when blank, at the elevation of the nearest data point.</p>}
        </>
      )}
      {rasters.length > 0 && (
        <>
          <Text label="Model CRS" id="surface-dem-crs" value={demCrs} onChange={setDemCrs} placeholder="of the database (EPSG:326..)" />
          <Text label="Drop within (m)" id="surface-dem-drop" value={dropWithin} onChange={setDropWithin} placeholder="grid spacing" />
          <p className="sidebar-note">The elevation file fills the area (the box of the data, or the extension above) on a grid of its cell size, moved up or down by the median height difference to the data; its nodes closer than the drop distance to the data are left out.</p>
        </>
      )}
      <Select label="Same XY" id="surface-duplicates" value={duplicates} onChange={setDuplicates} options={[{ value: 'mean', label: 'mean Z' }, { value: 'min', label: 'lowest Z' }, { value: 'max', label: 'highest Z' }, { value: 'first', label: 'first' }, { value: 'last', label: 'last' }, { value: 'error', label: 'stop with an error' }]} />
      <Text label="Name" id="surface-name" value={name} onChange={setName} />
      <div className="control">
        <button type="button" data-testid="surface-run" disabled={busy} onClick={() => void run()}>{busy ? 'Building…' : 'Build surface'}</button>
      </div>
      <Error message={error} id="surface-error" />
      {result && <p className="sidebar-note" data-testid="surface-result">{result}</p>}
    </>
  )
}

function FromRaster() {
  const summary = useOps((s) => s.summary)
  const databases = ofKind(summary, 'database')
  const models = ofKind(summary, 'model')
  const [files, setFiles] = useState<UploadedSource[]>([])
  const [database, setDatabase] = useState<string | null>(null)
  const [areaMode, setAreaMode] = useState<RasterForm['areaMode']>('collars')
  const [modelId, setModelId] = useState('')
  const [rectangle, setRectangle] = useState('')
  const [padding, setPadding] = useState('100')
  const [crs, setCrs] = useState('')
  const [rasterCrs, setRasterCrs] = useState('')
  const [spacing, setSpacing] = useState('')
  const [shiftMode, setShiftMode] = useState('median')
  const [shiftMetres, setShiftMetres] = useState('')
  const [tolerance, setTolerance] = useState('')
  const [name, setName] = useState('Topography DEM')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)

  const db = database ?? databases[0]?.id ?? ''
  const noCrs = files.some((f) => !f.info.crs)

  const run = async () => {
    if (shiftMode === 'metres' && !shiftMetres.trim()) return setError('Shift: give the metres added to the heights')
    const body = buildRasterRequest({
      name,
      rasters: files.map((u) => ({ upload: u.token, file: u.info.file })),
      rasterCrs,
      crs,
      spacing,
      zShift: shiftMode === 'metres' ? shiftMetres : shiftMode,
      database: db,
      areaMode,
      model: modelId || models[0]?.id || '',
      rectangle,
      padding,
      tolerance,
    })
    if (typeof body === 'string') return setError(body)
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await geometryApi.surfaceFromRaster(body)
      let text = `Saved “${r.name}”: ${fmt(r.n_faces, 0)} triangles on a ${r.grid.nodes.join(' x ')} grid at ${fmt(r.grid.spacing)} m${r.crs ? ` in ${r.crs}` : ''}, Z ${fmt(r.bounds[0][2])} to ${fmt(r.bounds[1][2])}; vertical shift ${signed(r.z_shift)} m.`
      if (r.grid.nodes_without_height) text += ` ${fmt(r.grid.nodes_without_height, 0)} nodes off the file.`
      text += collarText(r.collars)
      setResult(text)
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h4 className="ops-heading">Topography from an elevation file</h4>
      <label className="control">
        <span>Files</span>
        <input type="file" multiple accept=".tif,.tiff,.asc" data-testid="raster-files" title="GeoTIFF or ESRI ASCII grid: satellite DEM tiles (Copernicus, ALOS, SRTM), a drone or LiDAR grid" onChange={(e) => { const list = e.target.files; setError(null); void uploadRasters(list, setError).then((got) => setFiles((f) => [...f, ...got])); e.target.value = '' }} />
      </label>
      {files.map((u, i) => (
        <div key={`${u.token}-${u.info.file}`} className="ops-block" data-testid={`raster-upload-${i}`}>
          <span className="sidebar-note">{describeRaster(u.info)}</span>
          <div className="control"><button type="button" onClick={() => setFiles((list) => list.filter((_, j) => j !== i))}>Remove</button></div>
        </div>
      ))}
      {noCrs && <Text label="File CRS" id="raster-file-crs" value={rasterCrs} onChange={setRasterCrs} placeholder="blank = the model CRS" />}
      <Text label="Model CRS" id="raster-crs" value={crs} onChange={setCrs} placeholder="of the database (EPSG:326..)" />
      <Select label="Collars" id="raster-database" value={db} onChange={setDatabase} options={[...databases.map((d) => ({ value: d.id, label: d.name })), { value: '', label: 'none' }]} />
      <Select label="Area" id="raster-area" value={areaMode} onChange={(v) => setAreaMode(v as RasterForm['areaMode'])} options={[{ value: 'collars', label: 'box of the collars' }, { value: 'model', label: 'block model extent' }, { value: 'rectangle', label: 'rectangle' }]} />
      {areaMode === 'model' && <Select label="Model" id="raster-model" value={modelId || models[0]?.id || ''} onChange={setModelId} options={models.map((m) => ({ value: m.id, label: m.name }))} />}
      {areaMode === 'rectangle' && <Text label="Rectangle" id="raster-rectangle" value={rectangle} onChange={setRectangle} placeholder="xmin, ymin, xmax, ymax" />}
      <Text label="Padding (m)" id="raster-padding" value={padding} onChange={setPadding} />
      <Text label="Spacing (m)" id="raster-spacing" value={spacing} onChange={setSpacing} placeholder="cell size of the file" />
      <Select label="Vertical shift" id="raster-shift" value={shiftMode} onChange={setShiftMode} options={SHIFTS} />
      {shiftMode === 'metres' && <Text label="Shift (m)" id="raster-shift-metres" value={shiftMetres} onChange={setShiftMetres} />}
      <Text label="Collar tolerance (m)" id="raster-tolerance" value={tolerance} onChange={setTolerance} placeholder="5" />
      <p className="sidebar-note">A satellite DEM gives heights above a geoid, not the local survey heights: the median shift ties it to the collars. Collars that still differ by more than the tolerance are listed.</p>
      <Text label="Name" id="raster-name" value={name} onChange={setName} />
      <div className="control">
        <button type="button" data-testid="raster-run" disabled={busy || files.length === 0} onClick={() => void run()}>{busy ? 'Building…' : 'Build surface'}</button>
      </div>
      <Error message={error} id="raster-error" />
      {result && <p className="sidebar-note" data-testid="raster-result">{result}</p>}
    </>
  )
}

function ImportFile() {
  const [upload, setUpload] = useState<GeometryUpload | null>(null)
  const [kind, setKind] = useState('auto')
  const [scale, setScale] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)

  const mesh = upload?.files.find((f) => f.roles.includes('mesh'))
  const needs = upload?.files.find((f) => f.needs && !upload.files.some((o) => o.file.toLowerCase() === f.needs?.toLowerCase()))

  const addFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setError(null)
    setResult(null)
    try {
      // files chosen together belong together (a .dtm and its .str); a .str chosen later
      // joins the upload that is waiting for it
      let current = needs ? upload : null
      for (const file of Array.from(files)) current = await geometryApi.upload(file, current?.token)
      setUpload(current)
    } catch (e) {
      setError(message(e))
    }
  }

  const run = async () => {
    if (!upload || !mesh) return setError(needs ? `Also choose ${needs.needs} (the string file of the .dtm)` : 'Choose a .dxf, .obj, .ply, .stl or .dtm file')
    const factor = scale.trim() === '' ? undefined : Number(scale.trim().replace(',', '.'))
    if (factor !== undefined && !(Number.isFinite(factor) && factor > 0)) return setError('Scale: a positive number (0.3048 for a file in feet)')
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await geometryApi.importMesh({ upload: upload.token, file: mesh.file, kind, name: name.trim() || undefined, ...(factor !== undefined && factor !== 1 ? { scale: factor } : {}) })
      let text = `Saved “${r.name}” as a ${r.kind === 'dtm' ? 'surface (DTM)' : 'solid'}: ${fmt(r.n_faces, 0)} triangles.`
      if (r.kind === 'solid') text += r.valid ? ` Valid solid${r.volume != null ? `, volume ${fmt(r.volume, 0)} m³` : ''}.` : ` Not a valid solid: ${(r.issues ?? []).join('; ') || 'see validation'}.`
      if (r.warnings?.length) text += ` Warning: ${r.warnings.join(' ')}`
      setResult(text)
      setUpload(null)
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h4 className="ops-heading">Import a surface or solid file</h4>
      <label className="control">
        <span>File</span>
        <input type="file" multiple accept=".dxf,.obj,.ply,.stl,.dtm,.str" data-testid="mesh-import-file" title="DXF, OBJ, PLY, STL, or a Surpac text .dtm together with its .str" onChange={(e) => { void addFiles(e.target.files); e.target.value = '' }} />
      </label>
      {upload && (
        <p className="sidebar-note" data-testid="mesh-import-info">
          {mesh ? `${mesh.file}: ${fmt(mesh.n_vertices ?? 0, 0)} vertices, ${fmt(mesh.n_faces ?? 0, 0)} triangles` : needs ? `${needs.file} needs ${needs.needs}: choose it too` : 'No triangles found in the file'}
        </p>
      )}
      <Select label="As" id="mesh-import-kind" value={kind} onChange={setKind} options={[{ value: 'auto', label: 'automatic (closed = solid)' }, { value: 'dtm', label: 'surface (DTM)' }, { value: 'solid', label: 'solid' }]} />
      <Text label="Scale" id="mesh-import-scale" value={scale} onChange={setScale} placeholder="1 (0.3048 for a file in feet)" />
      <Text label="Name" id="mesh-import-name" value={name} onChange={setName} placeholder="(file name)" />
      <div className="control">
        <button type="button" data-testid="mesh-import-run" disabled={busy || !upload} onClick={() => void run()}>{busy ? 'Importing…' : 'Import'}</button>
      </div>
      <Error message={error} id="mesh-import-error" />
      {result && <p className="sidebar-note" data-testid="mesh-import-result">{result}</p>}
    </>
  )
}

export function SurfacePanel() {
  return (
    <Panel id="surface" title="Surfaces and solids">
      <FromPoints />
      <FromRaster />
      <ImportFile />
    </Panel>
  )
}

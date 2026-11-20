/**
 * Surfaces dialog: build a topography (DTM) from points - drillhole collars, CSV / XLSX point
 * tables and string files, in any combination, with optional breaklines and an extension to
 * a rectangle - and import an existing surface or solid file (DXF / OBJ / PLY / Surpac .dtm).
 * Runs the same code as `bakumine surface from-points` and `bakumine surface import`; the
 * result is added to the project and shows in the object tree.
 */
import { useState } from 'react'
import { ofKind, useOps } from '../../ops/api'
import { geometryApi, type GeometryFile, type GeometryUpload } from '../../ops/geometryApi'
import { buildSurfaceRequest, type SourceRef, type SurfaceForm } from '../../ops/geometryBuilders'
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
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)

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
        if (!info.roles.includes('points') && !info.roles.includes('strings')) {
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
        <input type="file" multiple accept=".csv,.txt,.xlsx,.str,.dxf" data-testid="surface-files" title="CSV / XLSX tables of points, .str or .dxf strings" onChange={(e) => { void addFiles(e.target.files); e.target.value = '' }} />
      </label>
      {uploads.map((u, i) => (
        <div key={`${u.token}-${u.info.file}`} className="ops-block" data-testid={`surface-upload-${i}`}>
          <span className="sidebar-note">
            {u.info.file}: {u.info.roles.includes('points') ? `${u.info.rows} rows${u.info.xyz ? ` (${u.info.xyz.join(', ')})` : ''}` : `${u.info.n_strings} strings, ${u.info.n_string_points} points`}
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
          <Text label="Boundary Z" id="surface-extend-z" value={extendZ} onChange={setExtendZ} placeholder="nearest point" />
          <p className="sidebar-note">Points are added on the rectangle (corners and along the sides, outside the data only), at the given elevation or, when blank, at the elevation of the nearest data point.</p>
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

function ImportFile() {
  const [upload, setUpload] = useState<GeometryUpload | null>(null)
  const [kind, setKind] = useState('auto')
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
    if (!upload || !mesh) return setError(needs ? `Also choose ${needs.needs} (the string file of the .dtm)` : 'Choose a .dxf, .obj, .ply or .dtm file')
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await geometryApi.importMesh({ upload: upload.token, file: mesh.file, kind, name: name.trim() || undefined })
      let text = `Saved “${r.name}” as a ${r.kind === 'dtm' ? 'surface (DTM)' : 'solid'}: ${fmt(r.n_faces, 0)} triangles.`
      if (r.kind === 'solid') text += r.valid ? ` Valid solid${r.volume != null ? `, volume ${fmt(r.volume, 0)} m³` : ''}.` : ` Not a valid solid: ${(r.issues ?? []).join('; ') || 'see validation'}.`
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
        <input type="file" multiple accept=".dxf,.obj,.ply,.dtm,.str" data-testid="mesh-import-file" title="DXF, OBJ, PLY, or a Surpac text .dtm together with its .str" onChange={(e) => { void addFiles(e.target.files); e.target.value = '' }} />
      </label>
      {upload && (
        <p className="sidebar-note" data-testid="mesh-import-info">
          {mesh ? `${mesh.file}: ${fmt(mesh.n_vertices ?? 0, 0)} vertices, ${fmt(mesh.n_faces ?? 0, 0)} triangles` : needs ? `${needs.file} needs ${needs.needs}: choose it too` : 'No triangles found in the file'}
        </p>
      )}
      <Select label="As" id="mesh-import-kind" value={kind} onChange={setKind} options={[{ value: 'auto', label: 'automatic (closed = solid)' }, { value: 'dtm', label: 'surface (DTM)' }, { value: 'solid', label: 'solid' }]} />
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
      <ImportFile />
    </Panel>
  )
}

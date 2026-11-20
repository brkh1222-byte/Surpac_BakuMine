/**
 * Implicit solid dialog: a closed solid of the intervals logged with one or more codes,
 * built directly from the drillholes (`bakumine solid implicit`, the RBF implicit model of
 * bakumine/core/geometry/implicit.py).
 *
 * The anisotropy decides the shape between the holes, so the dialog asks for it: the dip and
 * dip azimuth of the body (and the ratios of its extent along strike : down dip : across).
 * "Automatic" estimates the plane from one point per intercept; the result always states the
 * anisotropy that was used and warns when the automatic one follows the drilling.
 */
import { useState } from 'react'
import { ofKind, useOps } from '../../ops/api'
import { buildImplicitRequest, describeAnisotropy, implicitApi, type ImplicitForm, type ImplicitResult } from '../../ops/implicitSolid'
import { describePlane, parsePlane, planeAngles } from '../../ops/orientation'
import { Error, Panel, Select, Text, message, opts } from './common'

const KEYS = ['hole_id', 'depth_from', 'depth_to']

const INITIAL: ImplicitForm = {
  database: '', table: '', column: '', codes: '', mode: '',
  dip: '', dipAzimuth: '', pitch: '', angles: '', ratios: '',
  maxExtrapolation: '', topography: '', resolution: '', minVolume: '', name: '',
}

const MODES = [
  { value: '', label: '(choose)' },
  { value: 'plane', label: 'dip / dip azimuth' },
  { value: 'angles', label: 'bearing / plunge / dip' },
  { value: 'auto', label: 'automatic (from the intercepts)' },
  { value: 'none', label: 'none (isotropic)' },
]

/** What a plane input means, or nothing while it is incomplete. */
export function PlaneHint({ dip, dipAzimuth, pitch, id }: { dip: string; dipAzimuth: string; pitch: string; id: string }) {
  const plane = parsePlane(dip, dipAzimuth, pitch)
  if (typeof plane === 'string') return null
  const angles = planeAngles(plane).map((a) => Number(a.toFixed(1)))
  return (
    <p className="sidebar-note" data-testid={id}>
      {describePlane(plane)}. Bearing, plunge, dip: {angles.join(', ')}.
    </p>
  )
}

export function ImplicitSolidPanel() {
  const summary = useOps((s) => s.summary)
  const databases = ofKind(summary, 'database')
  const surfaces = ofKind(summary, 'dtm')
  const [form, setForm] = useState<ImplicitForm>(INITIAL)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ImplicitResult | null>(null)
  const patch = (p: Partial<ImplicitForm>) => setForm((f) => ({ ...f, ...p }))

  const db = databases.find((d) => d.id === form.database) ?? databases[0]
  const tables = db?.tables ?? {}
  const tableNames = Object.keys(tables)
  const table = tables[form.table] ? form.table : tableNames.includes('geology') ? 'geology' : tableNames[0] ?? ''
  const columns = tables[table]?.columns.filter((c) => !KEYS.includes(c)) ?? []
  const column = columns.includes(form.column) ? form.column : columns[0] ?? ''

  const run = async () => {
    const body = buildImplicitRequest({ ...form, database: db?.id ?? '', table, column })
    if (typeof body === 'string') return setError(body)
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await implicitApi.build(body))
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel id="implicit" title="Implicit solid">
      {databases.length === 0 && <p className="sidebar-note">Import a drillhole database first.</p>}
      {databases.length > 1 && <Select label="Database" id="implicit-db" value={db?.id ?? ''} onChange={(v) => patch({ database: v })} options={databases.map((d) => ({ value: d.id, label: d.name }))} />}
      <Select label="Table" id="implicit-table" value={table} onChange={(v) => patch({ table: v, column: '' })} options={opts(tableNames)} />
      <Select label="Column" id="implicit-column" value={column} onChange={(v) => patch({ column: v })} options={opts(columns)} />
      <Text label="Code(s)" id="implicit-codes" value={form.codes} onChange={(v) => patch({ codes: v })} placeholder="ORE (comma separated for several)" />

      <h4 className="ops-heading">Orientation of the body</h4>
      <Select label="Given as" id="implicit-mode" value={form.mode} onChange={(v) => patch({ mode: v as ImplicitForm['mode'] })} options={MODES} />
      {form.mode === 'plane' && (
        <>
          <Text label="Dip" id="implicit-dip" value={form.dip} onChange={(v) => patch({ dip: v })} placeholder="0 to 90, from horizontal" />
          <Text label="Dip azimuth" id="implicit-dip-azimuth" value={form.dipAzimuth} onChange={(v) => patch({ dipAzimuth: v })} placeholder="direction it dips towards" />
          <Text label="Pitch" id="implicit-pitch" value={form.pitch} onChange={(v) => patch({ pitch: v })} placeholder="0 = long axis along strike" />
          <PlaneHint dip={form.dip} dipAzimuth={form.dipAzimuth} pitch={form.pitch} id="implicit-plane-hint" />
        </>
      )}
      {form.mode === 'angles' && <Text label="Angles" id="implicit-angles" value={form.angles} onChange={(v) => patch({ angles: v })} placeholder="bearing, plunge, dip" />}
      {(form.mode === 'plane' || form.mode === 'angles') && <Text label="Ratios" id="implicit-ratios" value={form.ratios} onChange={(v) => patch({ ratios: v })} placeholder="along strike : down dip : across, e.g. 4:2:1" />}
      {form.mode === '' && <p className="sidebar-note">Give the dip and dip azimuth of the body: the orientation decides how the intercepts are joined between the holes.</p>}
      {form.mode === 'auto' && <p className="sidebar-note">One body: estimated from one point per intercept. Stacked lenses: the plane along which the ore of neighbouring holes lines up best, with a confidence; when no plane is clearly best nothing is chosen and the solid is isotropic. Check the anisotropy reported with the result.</p>}

      <h4 className="ops-heading">Limits</h4>
      <Text label="Max extrap. (m)" id="implicit-extrapolation" value={form.maxExtrapolation} onChange={(v) => patch({ maxExtrapolation: v })} placeholder="distance from the nearest hole (no limit)" />
      <Select label="Topography" id="implicit-topography" value={form.topography} onChange={(v) => patch({ topography: v })} options={[{ value: '', label: '(no clip)' }, ...surfaces.map((s) => ({ value: s.id, label: s.name }))]} />
      <Text label="Resolution (m)" id="implicit-resolution" value={form.resolution} onChange={(v) => patch({ resolution: v })} placeholder="automatic" />
      <Text label="Min volume" id="implicit-min-volume" value={form.minVolume} onChange={(v) => patch({ minVolume: v })} placeholder="m³; smaller shells are removed" />
      <Text label="Name" id="implicit-name" value={form.name} onChange={(v) => patch({ name: v })} placeholder={form.codes.trim() ? `${form.codes.split(',')[0].trim()} solid` : 'solid'} />
      <div className="control">
        <button type="button" data-testid="implicit-run" disabled={busy || !db} onClick={() => void run()}>{busy ? 'Building…' : 'Build solid'}</button>
      </div>
      <Error message={error} id="implicit-error" />
      {result && (
        <div data-testid="implicit-result" data-valid={result.valid}>
          <p className="sidebar-note" data-testid="implicit-volume">
            Saved “{result.name}”: volume {result.volume.toLocaleString('en-US', { maximumFractionDigits: 0 })} m³, {result.valid ? 'valid solid' : 'NOT a valid solid'}, {result.n_components} part{result.n_components === 1 ? '' : 's'}, {result.n_faces.toLocaleString('en-US')} triangles; {result.n_holes_intersecting} of {result.n_holes} holes intersect it.
          </p>
          <p className="sidebar-note" data-testid="implicit-anisotropy">{describeAnisotropy(result.anisotropy)}</p>
          {!result.valid && result.issues.length > 0 && <p className="sidebar-error" role="alert">{result.issues.join('; ')}</p>}
          {result.touches_grid_boundary && <p className="sidebar-error" role="alert">The solid reaches the edge of its grid and is cut flat there: set a maximum extrapolation.</p>}
          {result.warnings.map((w, i) => (
            <p key={i} className="sidebar-error" role="alert" data-testid={`implicit-warning-${i}`}>{w}</p>
          ))}
        </div>
      )}
    </Panel>
  )
}

/**
 * Estimation dialog (P6.T5): method (NN / IDW / OK), grade attribute, domain constraint and
 * sample filter, search ellipsoid, variogram and passes. The run is a background job
 * (`bakumine.core.estimation` with its progress and cancel hooks): the panel polls progress and
 * can cancel; a cancelled run leaves the model unchanged.
 *
 * The variogram is a nugget and up to three nested structures, each with its own type, sill,
 * ranges and orientation; ranges and orientation left alone follow the search ellipsoid.
 * Orientations (search ellipsoid and structures) are entered as bearing / plunge / dip or as
 * the dip / dip azimuth / pitch of a plane (`ops/orientation.ts`).
 */
import { useEffect, useRef, useState } from 'react'
import { ofKind, opsApi, useOps, type JobState } from '../../ops/api'
import { MAX_STRUCTURES, STRUCTURE_TYPES, buildEstimateSpec, type EstimateForm, type PassForm, type StructureForm } from '../../ops/builders'
import { PlaneHint } from './ImplicitSolidPanel'
import { Error, Panel, Select, Text, message, opts } from './common'

const pass = (name: string, factor: string, min: string): PassForm => ({ name, radiusFactor: factor, minSamples: min, maxSamples: '16', maxPerHole: '' })
const structure = (sill: string): StructureForm => ({ type: 'spherical', sill, ranges: '', orientation: 'search', angles: '', dip: '', dipAzimuth: '', pitch: '' })

const INITIAL: EstimateForm = {
  attribute: '',
  method: 'ok',
  power: '2',
  radii: '100, 100, 100',
  orientation: 'angles',
  angles: '0, 0, 0',
  dip: '',
  dipAzimuth: '',
  pitch: '',
  nugget: '0',
  structures: [structure('1'), structure('')],
  domainName: '',
  blockWhere: '',
  sampleColumn: '',
  sampleValues: '',
  passes: [pass('pass 1', '1', '1')],
}

export function EstimatePanel() {
  const summary = useOps((s) => s.summary)
  const models = ofKind(summary, 'model')
  const composites = ofKind(summary, 'composite')
  const [modelId, setModelId] = useState('')
  const [compId, setCompId] = useState('')
  const [form, setForm] = useState<EstimateForm>(INITIAL)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<JobState | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const model = models.find((m) => m.id === modelId) ?? models[0]
  const comp = composites.find((c) => c.id === compId) ?? composites[0]
  // composited attributes are the columns that have a `<name>_coverage` partner
  const grades = comp?.columns?.filter((c) => comp.columns?.includes(`${c}_coverage`)) ?? []
  const patch = (p: Partial<EstimateForm>) => setForm((f) => ({ ...f, ...p }))
  const patchStructure = (i: number, p: Partial<StructureForm>) => patch({ structures: form.structures.map((x, j) => (j === i ? { ...x, ...p } : x)) })
  const patchPass = (i: number, p: Partial<PassForm>) => patch({ passes: form.passes.map((x, j) => (j === i ? { ...x, ...p } : x)) })
  const running = job?.status === 'queued' || job?.status === 'running'

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const poll = (id: string) => {
    timer.current = window.setTimeout(async () => {
      try {
        const state = await opsApi.job(id)
        setJob(state)
        if (state.status === 'queued' || state.status === 'running') poll(id)
        else if (state.status === 'done') await useOps.getState().changed()
      } catch (e) {
        setError(message(e))
      }
    }, 400)
  }

  const run = async () => {
    if (!model || !comp) return setError('Needs a block model and composites')
    const attribute = form.attribute || grades[0] || ''
    const spec = buildEstimateSpec({ ...form, attribute })
    if (typeof spec === 'string') return setError(spec)
    // a grade that is already a block attribute is estimated into its own name; the samples
    // come from the composite column of the same name
    setError(null)
    setJob(null)
    try {
      const started = await opsApi.estimate({ model: model.id, composites: comp.id, plan: spec })
      setJob(started)
      poll(started.id)
    } catch (e) {
      setError(message(e))
    }
  }

  const cancel = async () => {
    if (!job) return
    try {
      setJob(await opsApi.cancel(job.id))
    } catch (e) {
      setError(message(e))
    }
  }

  return (
    <Panel id="estimate" title="Estimation">
      {(models.length === 0 || composites.length === 0) && <p className="sidebar-note">Needs a block model and composites.</p>}
      {models.length > 1 && <Select label="Model" id="estimate-model" value={model?.id ?? ''} onChange={setModelId} options={models.map((m) => ({ value: m.id, label: m.name }))} />}
      {composites.length > 1 && <Select label="Composites" id="estimate-composites" value={comp?.id ?? ''} onChange={setCompId} options={composites.map((m) => ({ value: m.id, label: m.name }))} />}
      <Select label="Method" id="estimate-method" value={form.method} onChange={(v) => patch({ method: v as EstimateForm['method'] })} options={[{ value: 'nn', label: 'Nearest neighbour' }, { value: 'idw', label: 'Inverse distance' }, { value: 'ok', label: 'Ordinary kriging' }]} />
      <Select label="Grade" id="estimate-attribute" value={form.attribute || grades[0] || ''} onChange={(v) => patch({ attribute: v })} options={opts(grades)} />
      {form.method === 'idw' && <Text label="Power" id="estimate-power" value={form.power} onChange={(v) => patch({ power: v })} />}

      <h4 className="ops-heading">Domain</h4>
      <Text label="Name" id="estimate-domain" value={form.domainName} onChange={(v) => patch({ domainName: v })} placeholder="all" />
      <Text label="Blocks where" id="estimate-block-where" value={form.blockWhere} onChange={(v) => patch({ blockWhere: v })} placeholder='rock == "ORE"' />
      <Select label="Sample column" id="estimate-sample-column" value={form.sampleColumn} onChange={(v) => patch({ sampleColumn: v })} options={opts(comp?.columns?.filter((c) => c === 'domain' || !grades.includes(c)) ?? [], '(all samples)')} />
      {form.sampleColumn && <Text label="Equals" id="estimate-sample-values" value={form.sampleValues} onChange={(v) => patch({ sampleValues: v })} placeholder="ORE (comma separated for several)" />}

      <h4 className="ops-heading">Search ellipsoid</h4>
      <Text label="Radii (m)" id="estimate-radii" value={form.radii} onChange={(v) => patch({ radii: v })} placeholder="major, semi-major, minor" />
      <Select label="Orientation" id="estimate-orientation" value={form.orientation ?? 'angles'} onChange={(v) => patch({ orientation: v as 'angles' | 'plane' })} options={[{ value: 'angles', label: 'bearing / plunge / dip' }, { value: 'plane', label: 'dip / dip azimuth / pitch' }]} />
      {form.orientation === 'plane' ? (
        <>
          <Text label="Dip" id="estimate-dip" value={form.dip ?? ''} onChange={(v) => patch({ dip: v })} placeholder="0 to 90, from horizontal" />
          <Text label="Dip azimuth" id="estimate-dip-azimuth" value={form.dipAzimuth ?? ''} onChange={(v) => patch({ dipAzimuth: v })} placeholder="direction the plane dips towards" />
          <Text label="Pitch" id="estimate-pitch" value={form.pitch ?? ''} onChange={(v) => patch({ pitch: v })} placeholder="0 = major axis along strike" />
          <PlaneHint dip={form.dip ?? ''} dipAzimuth={form.dipAzimuth ?? ''} pitch={form.pitch ?? ''} id="estimate-plane-hint" />
        </>
      ) : (
        <Text label="Angles (deg)" id="estimate-angles" value={form.angles} onChange={(v) => patch({ angles: v })} placeholder="bearing, plunge, dip" />
      )}

      {form.method === 'ok' && (
        <>
          <h4 className="ops-heading">Variogram</h4>
          <Text label="Nugget" id="estimate-nugget" value={form.nugget} onChange={(v) => patch({ nugget: v })} />
          {form.structures.map((s, i) => (
            <div key={i} className="ops-block" data-testid={`estimate-structure-${i}`}>
              <Select label={`Structure ${i + 1}`} id={`estimate-structure-${i}-type`} value={s.type} onChange={(v) => patchStructure(i, { type: v })} options={opts(STRUCTURE_TYPES)} />
              <Text label="Sill" id={`estimate-structure-${i}-sill`} value={s.sill} onChange={(v) => patchStructure(i, { sill: v })} placeholder={i === 0 ? '' : '(blank = unused)'} />
              <Text label="Ranges (m)" id={`estimate-structure-${i}-ranges`} value={s.ranges ?? ''} onChange={(v) => patchStructure(i, { ranges: v })} placeholder="major, semi-major, minor (blank = search radii)" />
              <Select label="Orientation" id={`estimate-structure-${i}-orientation`} value={s.orientation ?? 'search'} onChange={(v) => patchStructure(i, { orientation: v as StructureForm['orientation'] })} options={[{ value: 'search', label: 'same as the search ellipsoid' }, { value: 'angles', label: 'bearing / plunge / dip' }, { value: 'plane', label: 'dip / dip azimuth / pitch' }]} />
              {s.orientation === 'angles' && <Text label="Angles (deg)" id={`estimate-structure-${i}-angles`} value={s.angles ?? ''} onChange={(v) => patchStructure(i, { angles: v })} placeholder="bearing, plunge, dip" />}
              {s.orientation === 'plane' && (
                <>
                  <Text label="Dip" id={`estimate-structure-${i}-dip`} value={s.dip ?? ''} onChange={(v) => patchStructure(i, { dip: v })} />
                  <Text label="Dip azimuth" id={`estimate-structure-${i}-dip-azimuth`} value={s.dipAzimuth ?? ''} onChange={(v) => patchStructure(i, { dipAzimuth: v })} />
                  <Text label="Pitch" id={`estimate-structure-${i}-pitch`} value={s.pitch ?? ''} onChange={(v) => patchStructure(i, { pitch: v })} placeholder="0 = major axis along strike" />
                  <PlaneHint dip={s.dip ?? ''} dipAzimuth={s.dipAzimuth ?? ''} pitch={s.pitch ?? ''} id={`estimate-structure-${i}-plane-hint`} />
                </>
              )}
              {form.structures.length > 1 && <div className="control"><button type="button" data-testid={`estimate-structure-${i}-remove`} onClick={() => patch({ structures: form.structures.filter((_, j) => j !== i) })}>Remove structure</button></div>}
            </div>
          ))}
          {form.structures.length < MAX_STRUCTURES && <div className="control"><button type="button" data-testid="estimate-add-structure" onClick={() => patch({ structures: [...form.structures, structure('')] })}>Add structure</button></div>}
          <p className="sidebar-note">Each structure has its own ranges and orientation; left blank they follow the search ellipsoid. Exponential and gaussian ranges are practical ranges (95 % of the sill).</p>
        </>
      )}

      <h4 className="ops-heading">Passes</h4>
      {form.passes.map((p, i) => (
        <div key={i} className="ops-block" data-testid={`estimate-pass-${i}`}>
          <Text label="Name" id={`estimate-pass-${i}-name`} value={p.name} onChange={(v) => patchPass(i, { name: v })} />
          <Text label="Radius factor" id={`estimate-pass-${i}-factor`} value={p.radiusFactor} onChange={(v) => patchPass(i, { radiusFactor: v })} />
          <Text label="Min samples" id={`estimate-pass-${i}-min`} value={p.minSamples} onChange={(v) => patchPass(i, { minSamples: v })} />
          <Text label="Max samples" id={`estimate-pass-${i}-max`} value={p.maxSamples} onChange={(v) => patchPass(i, { maxSamples: v })} />
          <Text label="Max per hole" id={`estimate-pass-${i}-perhole`} value={p.maxPerHole} onChange={(v) => patchPass(i, { maxPerHole: v })} placeholder="no limit" />
          {form.passes.length > 1 && <div className="control"><button type="button" onClick={() => patch({ passes: form.passes.filter((_, j) => j !== i) })}>Remove pass</button></div>}
        </div>
      ))}
      <div className="control"><button type="button" data-testid="estimate-add-pass" onClick={() => patch({ passes: [...form.passes, pass(`pass ${form.passes.length + 1}`, String(form.passes.length + 1), '1')] })}>Add pass</button></div>

      <div className="control">
        <button type="button" data-testid="estimate-run" disabled={running || !model || !comp} onClick={() => void run()}>{running ? 'Running…' : 'Run estimation'}</button>
        {running && <button type="button" data-testid="estimate-cancel" onClick={() => void cancel()}>Cancel</button>}
      </div>
      <Error message={error} id="estimate-error" />
      {job && (
        <div data-testid="estimate-job" data-status={job.status}>
          <progress max={1} value={job.progress} data-testid="estimate-progress" />
          <p className="sidebar-note">{job.status}{job.message ? `: ${job.message}` : ''}</p>
          {job.error && <p className="sidebar-error" role="alert" data-testid="estimate-job-error">{job.error}</p>}
          {job.summary && !job.summary.cancelled && (
            <>
              <p className="sidebar-note" data-testid="estimate-summary">
                {job.summary.attribute}: {job.summary.n_estimated} of {job.summary.n_targeted} blocks estimated, {job.summary.n_unestimated} not; estimation step {job.summary.seconds.toFixed(1)} s.
              </p>
            </>
          )}
        </div>
      )}
    </Panel>
  )
}

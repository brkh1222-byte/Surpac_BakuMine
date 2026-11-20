/**
 * Block model dialogs (P6.T4): create a model (explicit grid, or fitted to drillholes / a solid
 * / a DTM with padding) and code it with ordered steps (attribute, value, constraint picked
 * from the project's solids and DTMs, combined with and / and not; sub-blocking on or off).
 * Same code as `bakumine blockmodel create` and `blockmodel code`.
 */
import { useState } from 'react'
import { ofKind, opsApi, useOps } from '../../ops/api'
import {
  EMPTY_CONSTRAINT,
  buildCodeRequest,
  nums,
  type ConstraintForm,
  type StepForm,
} from '../../ops/builders'
import { Check, Error, Panel, Select, Text, message, opts } from './common'

function CreateModel() {
  const summary = useOps((s) => s.summary)
  const [name, setName] = useState('model')
  const [how, setHow] = useState<'grid' | 'fit'>('grid')
  const [origin, setOrigin] = useState('')
  const [size, setSize] = useState('10, 10, 5')
  const [shape, setShape] = useState('')
  const [fit, setFit] = useState<string[]>([])
  const [padding, setPadding] = useState('0')
  const [depth, setDepth] = useState('0')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const fitCandidates = summary?.objects.filter((o) => ['database', 'solid', 'dtm'].includes(o.kind)) ?? []

  const run = async () => {
    const sz = nums(size)
    if (!sz || sz.length !== 3) return setError('Block size: three numbers')
    const body: Record<string, unknown> = { name, size: sz, max_depth: Number(depth) || 0 }
    if (how === 'grid') {
      const o = nums(origin)
      const sh = nums(shape)
      if (!o || o.length !== 3) return setError('Origin: three numbers (x, y, z)')
      if (!sh || sh.length !== 3) return setError('Shape: three whole numbers (nx, ny, nz)')
      body.origin = o
      body.shape = sh
    } else {
      if (fit.length === 0) return setError('Tick what the model should cover')
      const p = nums(padding)
      if (!p || (p.length !== 1 && p.length !== 3)) return setError('Padding: one number or three')
      body.extent_from = fit
      body.padding = p.length === 1 ? p[0] : p
    }
    setError(null)
    setResult(null)
    try {
      const r = await opsApi.createModel(body)
      setResult(`Created “${String(r.name)}”: ${String(r.n_blocks)} parent blocks, shape ${(r.shape as number[]).join(' x ')}, origin ${(r.origin as number[]).join(', ')}.`)
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    }
  }

  return (
    <>
      <h3 className="ops-heading">Create model</h3>
      <Text label="Name" id="model-name" value={name} onChange={setName} />
      <Select label="Extent" id="model-how" value={how} onChange={(v) => setHow(v as 'grid' | 'fit')} options={[{ value: 'grid', label: 'origin and shape' }, { value: 'fit', label: 'fit to objects' }]} />
      {how === 'grid' ? (
        <>
          <Text label="Origin" id="model-origin" value={origin} onChange={setOrigin} placeholder="x, y, z (minimum corner)" />
          <Text label="Shape" id="model-shape" value={shape} onChange={setShape} placeholder="nx, ny, nz" />
        </>
      ) : (
        <>
          <fieldset className="control report-set">
            <legend>Cover</legend>
            {fitCandidates.map((o) => (
              <Check key={o.id} label={`${o.name} (${o.kind})`} id={`model-fit-${o.id}`} checked={fit.includes(o.id)} onChange={(c) => setFit(c ? [...fit, o.id] : fit.filter((f) => f !== o.id))} />
            ))}
          </fieldset>
          <Text label="Padding (m)" id="model-padding" value={padding} onChange={setPadding} />
        </>
      )}
      <Text label="Block size" id="model-size" value={size} onChange={setSize} placeholder="dx, dy, dz" />
      <Text label="Sub-block depth" id="model-depth" value={depth} onChange={setDepth} placeholder="0 = no sub-blocks" />
      <div className="control">
        <button type="button" data-testid="model-create" onClick={() => void run()}>Create model</button>
      </div>
      <Error message={error} id="model-error" />
      {result && <p className="sidebar-note" data-testid="model-result">{result}</p>}
    </>
  )
}

const blankStep = (): StepForm => ({ attribute: '', value: '', first: { ...EMPTY_CONSTRAINT }, combine: 'and', second: { ...EMPTY_CONSTRAINT } })

function ConstraintEditor({ id, value, onChange, solids, dtms }: { id: string; value: ConstraintForm; onChange: (c: ConstraintForm) => void; solids: { id: string; name: string }[]; dtms: { id: string; name: string }[] }) {
  const set = (p: Partial<ConstraintForm>) => onChange({ ...value, ...p })
  const refs = value.type === 'solid' ? solids : dtms
  return (
    <>
      <Select
        label="Where"
        id={`${id}-type`}
        value={value.type}
        onChange={(v) => onChange({ ...EMPTY_CONSTRAINT, type: v as ConstraintForm['type'], side: v === 'solid' ? 'inside' : v === 'dtm' ? 'above' : '' })}
        options={[{ value: 'none', label: '(none)' }, { value: 'solid', label: 'solid' }, { value: 'dtm', label: 'surface' }, { value: 'attribute', label: 'expression' }]}
      />
      {(value.type === 'solid' || value.type === 'dtm') && (
        <>
          <Select label={value.type === 'solid' ? 'Solid' : 'Surface'} id={`${id}-ref`} value={value.ref} onChange={(v) => set({ ref: v })} options={[{ value: '', label: '(pick)' }, ...refs.map((r) => ({ value: r.id, label: r.name }))]} />
          <Select
            label="Side"
            id={`${id}-side`}
            value={value.side}
            onChange={(v) => set({ side: v })}
            options={value.type === 'solid' ? opts(['inside', 'outside']) : opts(['above', 'below'])}
          />
          {value.type === 'dtm' && <Check label="Include blocks beyond the surface edge" id={`${id}-outside`} checked={value.outside} onChange={(c) => set({ outside: c })} />}
        </>
      )}
      {value.type === 'attribute' && <Text label="Expression" id={`${id}-expression`} value={value.expression} onChange={(v) => set({ expression: v })} placeholder='rock == "ORE"' />}
    </>
  )
}

function CodeModel() {
  const summary = useOps((s) => s.summary)
  const models = ofKind(summary, 'model')
  const solids = ofKind(summary, 'solid')
  const dtms = ofKind(summary, 'dtm')
  const [chosen, setChosen] = useState('')
  const [attributes, setAttributes] = useState([{ name: '', kind: 'category', background: '', unit: '' }])
  const [steps, setSteps] = useState<StepForm[]>([blankStep()])
  const [subblock, setSubblock] = useState(true)
  const [merge, setMerge] = useState(true)
  const [maxLevel, setMaxLevel] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<string[] | null>(null)
  const [busy, setBusy] = useState(false)
  const model = models.find((m) => m.id === chosen) ?? models[0]
  const patchStep = (i: number, p: Partial<StepForm>) => setSteps((s) => s.map((x, j) => (j === i ? { ...x, ...p } : x)))
  const patchAttr = (i: number, p: Partial<(typeof attributes)[number]>) => setAttributes((s) => s.map((x, j) => (j === i ? { ...x, ...p } : x)))
  const known = [...(model?.attributes?.map((a) => a.name) ?? []), ...attributes.map((a) => a.name).filter(Boolean)]

  const run = async () => {
    if (!model) return setError('Create a block model first')
    const body = buildCodeRequest(attributes, steps, { subblock, merge, maxLevel })
    if (typeof body === 'string') return setError(body)
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await opsApi.codeModel(model.id, body)
      setResult(r.steps.map((s) => `${String(s.attribute)} = ${String(s.value)}: ${String(s.n_assigned)} blocks, ${Math.round(Number(s.volume)).toLocaleString('en-US')} m3`).concat(`${r.n_blocks} blocks now`))
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h3 className="ops-heading">Code model</h3>
      {models.length === 0 && <p className="sidebar-note">Create a block model first.</p>}
      {models.length > 1 && <Select label="Model" id="code-model" value={model?.id ?? ''} onChange={setChosen} options={models.map((m) => ({ value: m.id, label: m.name }))} />}
      <h4 className="ops-heading">New attributes</h4>
      {attributes.map((a, i) => (
        <div key={i} className="ops-block">
          <Text label="Name" id={`code-attr-${i}-name`} value={a.name} onChange={(v) => patchAttr(i, { name: v })} />
          <Select label="Kind" id={`code-attr-${i}-kind`} value={a.kind} onChange={(v) => patchAttr(i, { kind: v })} options={opts(['category', 'float', 'int'])} />
          {a.kind === 'category' ? (
            <Text label="Background" id={`code-attr-${i}-background`} value={a.background} onChange={(v) => patchAttr(i, { background: v })} placeholder="value of uncoded blocks" />
          ) : (
            <Text label="Unit" id={`code-attr-${i}-unit`} value={a.unit} onChange={(v) => patchAttr(i, { unit: v })} />
          )}
        </div>
      ))}
      <div className="control"><button type="button" data-testid="code-add-attribute" onClick={() => setAttributes([...attributes, { name: '', kind: 'float', background: '', unit: '' }])}>Add attribute</button></div>
      <h4 className="ops-heading">Steps (applied in order)</h4>
      {steps.map((s, i) => (
        <div key={i} className="ops-block" data-testid={`code-step-${i}`}>
          <Text label="Attribute" id={`code-step-${i}-attribute`} value={s.attribute} onChange={(v) => patchStep(i, { attribute: v })} list="code-attributes" />
          <Text label="Value" id={`code-step-${i}-value`} value={s.value} onChange={(v) => patchStep(i, { value: v })} placeholder='ORE, 2.7 or iif(rock == "ORE", 2.8, 1.9)' />
          <ConstraintEditor id={`code-step-${i}-first`} value={s.first} onChange={(c) => patchStep(i, { first: c })} solids={solids} dtms={dtms} />
          {s.first.type !== 'none' && (
            <>
              <Select label="Combine" id={`code-step-${i}-combine`} value={s.combine} onChange={(v) => patchStep(i, { combine: v as StepForm['combine'] })} options={opts(['and', 'and not'])} />
              <ConstraintEditor id={`code-step-${i}-second`} value={s.second} onChange={(c) => patchStep(i, { second: c })} solids={solids} dtms={dtms} />
            </>
          )}
          {steps.length > 1 && <div className="control"><button type="button" onClick={() => setSteps(steps.filter((_, j) => j !== i))}>Remove step</button></div>}
        </div>
      ))}
      <datalist id="code-attributes">{known.map((k) => <option key={k} value={k} />)}</datalist>
      <div className="control"><button type="button" data-testid="code-add-step" onClick={() => setSteps([...steps, blankStep()])}>Add step</button></div>
      <Check label="Sub-block at the boundaries" id="code-subblock" checked={subblock} onChange={setSubblock} />
      <Check label="Merge uniform sub-blocks" id="code-merge" checked={merge} onChange={setMerge} />
      {subblock && <Text label="Max level" id="code-maxlevel" value={maxLevel} onChange={setMaxLevel} placeholder="model's depth" />}
      <div className="control">
        <button type="button" data-testid="code-run" disabled={busy || !model} onClick={() => void run()}>{busy ? 'Coding…' : 'Code model'}</button>
      </div>
      <Error message={error} id="code-error" />
      {result && <ul className="sidebar-note" data-testid="code-result">{result.map((r) => <li key={r}>{r}</li>)}</ul>}
    </>
  )
}

export function ModelPanel() {
  return (
    <Panel id="model" title="Block model">
      <CreateModel />
      <CodeModel />
    </Panel>
  )
}

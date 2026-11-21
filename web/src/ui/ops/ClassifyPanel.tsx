/**
 * Resource classification dialog: an ordered list of classes (free-text labels) with their
 * criteria, the optional drillhole support and the optional smoothing. Runs the same code
 * as `bakumine classify`; the class is a category attribute of the model, so the report
 * can be grouped by it. Undo is in the job log.
 */
import { useState } from 'react'
import { ofKind, useOps } from '../../ops/api'
import { CRITERIA, JORC_CLASSES, buildClassifyRequest, describeClasses, runClassify, type ClassRow, type ClassifyResult } from '../../ops/classify'
import { Check, Error, Panel, Select, Text, message, opts, useClearOnUndo } from './common'

const CIS_CLASSES: ClassRow[] = [
  { label: 'B', criteria: 'max_pass=1, min_samples=8, min_holes=3' },
  { label: 'C', criteria: 'max_pass=2, min_holes=2' },
  { label: 'P1', criteria: '' },
]

export function ClassifyPanel() {
  const summary = useOps((s) => s.summary)
  const models = ofKind(summary, 'model')
  const composites = ofKind(summary, 'composite')
  const [chosen, setChosen] = useState('')
  const [attribute, setAttribute] = useState('')
  const [output, setOutput] = useState('res_class')
  const [comps, setComps] = useState('')
  const [holeRadius, setHoleRadius] = useState('')
  const [holeNth, setHoleNth] = useState('3')
  const [holeFilter, setHoleFilter] = useState('')
  const [classes, setClasses] = useState<ClassRow[]>(JORC_CLASSES)
  const [smooth, setSmooth] = useState(false)
  const [overwriteOutput, setOverwriteOutput] = useState(false)
  const [minAgree, setMinAgree] = useState('4')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ClassifyResult | null>(null)
  const [busy, setBusy] = useState(false)
  useClearOnUndo(() => {
    setResult(null)
    setError(null)
  })

  const model = models.find((m) => m.id === chosen) ?? models[0]
  const names = new Set((model?.attributes ?? []).map((a) => a.name))
  // estimated grades: float attributes that have a pass number next to them
  const floats = (model?.attributes ?? []).filter((a) => a.kind === 'float').map((a) => a.name)
  const grades = floats.filter((n) => names.has(`${n}_pass`))
  const candidates = grades.length > 0 ? grades : floats
  const grade = candidates.includes(attribute) ? attribute : candidates[0] ?? ''
  const setRow = (i: number, patch: Partial<ClassRow>) => setClasses(classes.map((c, j) => (j === i ? { ...c, ...patch } : c)))

  const ownSupport = classes.some((c) => /hole_(radius|nth)/.test(c.criteria))

  const run = async () => {
    const body = buildClassifyRequest({
      model: model?.id ?? '', attribute: grade, output, composites: holeRadius.trim() || ownSupport ? comps || composites[0]?.id || '' : comps,
      holeRadius, holeNth, holeFilter, classes, smooth, minAgree, maxSame: '0', iterations: '1', overwriteOutput,
    })
    if (typeof body === 'string') return setError(body)
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await runClassify(model!.id, body))
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel id="classify" title="Classify">
      {models.length === 0 && <p className="sidebar-note">Create and estimate a block model first.</p>}
      {models.length > 1 && <Select label="Model" id="classify-model" value={model?.id ?? ''} onChange={setChosen} options={models.map((m) => ({ value: m.id, label: m.name }))} />}
      <Select label="Estimated grade" id="classify-attribute" value={grade} onChange={setAttribute} options={opts(candidates)} />
      <Text label="Class attribute" id="classify-output" value={output} onChange={setOutput} />
      {names.has(output.trim()) && (
        <Check label="Replace this attribute even if no classification wrote it" id="classify-overwrite-output" checked={overwriteOutput} onChange={setOverwriteOutput} />
      )}
      <p className="ops-heading">Classes, best first (first match wins)</p>
      {classes.map((c, i) => (
        <div className="ops-block" key={i}>
          <Text label="Label" id={`classify-label-${i}`} value={c.label} onChange={(v) => setRow(i, { label: v })} />
          <Text label="Criteria" id={`classify-criteria-${i}`} value={c.criteria} onChange={(v) => setRow(i, { criteria: v })} placeholder="the rest (estimated)" />
          <button type="button" data-testid={`classify-remove-${i}`} onClick={() => setClasses(classes.filter((_, j) => j !== i))}>Remove</button>
        </div>
      ))}
      <div className="control">
        <button type="button" data-testid="classify-add" onClick={() => setClasses([...classes, { label: '', criteria: '' }])}>Add class</button>
        <button type="button" data-testid="classify-preset-jorc" onClick={() => setClasses(JORC_CLASSES)}>Measured / Indicated / Inferred</button>
        <button type="button" data-testid="classify-preset-cis" onClick={() => setClasses(CIS_CLASSES)}>B / C / P1</button>
      </div>
      <p className="sidebar-note">Criteria as name=value, comma separated: {CRITERIA.join(', ')}. A class may have its own drillhole support: hole_radius=75, hole_nth=2.</p>
      <Text label="Hole radius (m)" id="classify-hole-radius" value={holeRadius} onChange={setHoleRadius} placeholder="use stored support" />
      {holeRadius.trim() && <Text label="Distance to hole no." id="classify-hole-nth" value={holeNth} onChange={setHoleNth} />}
      {(holeRadius.trim() || ownSupport) && (
        <>
          <Text label="Only composites with" id="classify-hole-filter" value={holeFilter} onChange={setHoleFilter} placeholder="all; e.g. domain=MIN" />
          <Select label="Composites" id="classify-composites" value={comps || composites[0]?.id || ''} onChange={setComps} options={composites.map((c) => ({ value: c.id, label: c.name }))} />
        </>
      )}
      <Check label="Smooth isolated blocks (parent grid)" id="classify-smooth" checked={smooth} onChange={setSmooth} />
      {smooth && <Text label="Neighbours agreeing (of 6)" id="classify-min-agree" value={minAgree} onChange={setMinAgree} />}
      <div className="control">
        <button type="button" data-testid="classify-run" disabled={busy || !model} onClick={() => void run()}>{busy ? 'Running…' : 'Classify'}</button>
      </div>
      <Error message={error} id="classify-error" />
      {result && (
        <div className="sidebar-note" data-testid="classify-result">
          <p>Attribute “{result.output}”: {result.n_classified} of {result.n_estimated} estimated blocks classified{result.n_smoothed > 0 ? `, ${result.n_smoothed} changed by smoothing` : ''}.</p>
          {describeClasses(result).map((line) => <p key={line}>{line}</p>)}
          <p>Group the report by “{result.output}” for tonnes and grade per class. Undo is in the job log.</p>
        </div>
      )}
    </Panel>
  )
}

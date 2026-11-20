/**
 * Mineralised intervals dialog: cut-off, minimum thickness, internal waste and the other
 * rules of `bakumine intercepts`. Lists the significant intercepts and can store the coded
 * intervals (MIN / WASTE) in the database as a table, to be used as compositing domains.
 */
import { useState } from 'react'
import { ofKind, useOps } from '../../ops/api'
import { buildInterceptsRequest, describeIntercept, runIntercepts, type InterceptsResult } from '../../ops/intercepts'
import { Check, Error, Panel, Select, Text, message, opts } from './common'

const SHOWN = 200

export function InterceptsPanel() {
  const summary = useOps((s) => s.summary)
  const databases = ofKind(summary, 'database')
  const [chosen, setChosen] = useState('')
  const [table, setTable] = useState('')
  const [grade, setGrade] = useState('')
  const [cutoff, setCutoff] = useState('0.2')
  const [minThickness, setMinThickness] = useState('2')
  const [maxInternalWaste, setMaxInternalWaste] = useState('2')
  const [maxTotalWaste, setMaxTotalWaste] = useState('')
  const [accumulationRule, setAccumulationRule] = useState(false)
  const [gapTolerance, setGapTolerance] = useState('0')
  const [topCut, setTopCut] = useState('')
  const [including, setIncluding] = useState('')
  const [includingMinThickness, setIncludingMinThickness] = useState('')
  const [bodyDip, setBodyDip] = useState('')
  const [bodyDipAzimuth, setBodyDipAzimuth] = useState('')
  const [saveAs, setSaveAs] = useState('mineralised')
  const [overwrite, setOverwrite] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<InterceptsResult | null>(null)
  const [busy, setBusy] = useState(false)

  const db = databases.find((d) => d.id === chosen) ?? databases[0]
  const tables = db?.tables ?? {}
  const tableNames = Object.keys(tables)
  const assayTable = tables[table] ? table : (tableNames.includes('assay') ? 'assay' : tableNames[0] ?? '')
  const grades = tables[assayTable]?.attributes ?? []
  const gradeColumn = grades.includes(grade) ? grade : grades[0] ?? ''

  const run = async () => {
    const body = buildInterceptsRequest({
      database: db?.id ?? '', table: assayTable, grade: gradeColumn, cutoff, minThickness, maxInternalWaste, maxTotalWaste,
      accumulationRule, gapTolerance, topCut, including, includingMinThickness, bodyDip, bodyDipAzimuth, saveAs, overwrite,
    })
    if (typeof body === 'string') return setError(body)
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await runIntercepts(body))
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel id="intercepts" title="Mineralised intervals">
      {databases.length === 0 && <p className="sidebar-note">Import a drillhole database first.</p>}
      {databases.length > 1 && <Select label="Database" id="intercepts-db" value={db?.id ?? ''} onChange={setChosen} options={databases.map((d) => ({ value: d.id, label: d.name }))} />}
      <Select label="Assays" id="intercepts-table" value={assayTable} onChange={setTable} options={opts(tableNames)} />
      <Select label="Grade" id="intercepts-grade" value={gradeColumn} onChange={setGrade} options={opts(grades)} />
      <Text label="Cut-off (≥)" id="intercepts-cutoff" value={cutoff} onChange={setCutoff} />
      <Text label="Min thickness (m)" id="intercepts-min-thickness" value={minThickness} onChange={setMinThickness} />
      <Text label="Internal waste (m)" id="intercepts-waste" value={maxInternalWaste} onChange={setMaxInternalWaste} />
      <Text label="Total waste (m)" id="intercepts-total-waste" value={maxTotalWaste} onChange={setMaxTotalWaste} placeholder="no limit" />
      <Check label="Accept thin runs by grade × thickness" id="intercepts-accumulation" checked={accumulationRule} onChange={setAccumulationRule} />
      <Text label="Bridge gaps ≤ (m)" id="intercepts-gap" value={gapTolerance} onChange={setGapTolerance} />
      <Text label="Top cut" id="intercepts-top-cut" value={topCut} onChange={setTopCut} placeholder="none" />
      <Text label="Including ≥" id="intercepts-including" value={including} onChange={setIncluding} placeholder="e.g. 1, 5" />
      {including.trim() && <Text label="Including: min (m)" id="intercepts-including-min" value={includingMinThickness} onChange={setIncludingMinThickness} placeholder="as above" />}
      <Text label="Body dip (°)" id="intercepts-dip" value={bodyDip} onChange={setBodyDip} placeholder="down-hole length" />
      <Text label="Dip azimuth (°)" id="intercepts-dip-azimuth" value={bodyDipAzimuth} onChange={setBodyDipAzimuth} />
      <Text label="Store as table" id="intercepts-save-as" value={saveAs} onChange={setSaveAs} placeholder="do not store" />
      {saveAs.trim() && <Check label="Replace an existing table" id="intercepts-overwrite" checked={overwrite} onChange={setOverwrite} />}
      <div className="control">
        <button type="button" data-testid="intercepts-run" disabled={busy || !db} onClick={() => void run()}>{busy ? 'Running…' : 'Find intercepts'}</button>
      </div>
      <Error message={error} id="intercepts-error" />
      {result && (
        <>
          <p className="sidebar-note" data-testid="intercepts-result">
            {result.n_intercepts} intercepts in {result.n_holes_mineralised} of {result.n_holes} holes: {result.mineralised_length.toFixed(1)} m
            {result.mean_grade != null && ` @ ${result.mean_grade.toFixed(2)}`}.
            {result.interval_table ? ` Stored as table “${result.interval_table}” (column “code”).` : ''} Table: {result.file}
          </p>
          <div className="report-table-wrap">
            <ul className="sidebar-note" data-testid="intercepts-list">
              {result.table.slice(0, SHOWN).map((row, i) => (
                <li key={i} style={{ marginLeft: row.level === 'including' ? 12 : 0, listStyle: 'none' }}>{describeIntercept(row)}</li>
              ))}
            </ul>
          </div>
          {(result.truncated || result.table.length > SHOWN) && <p className="sidebar-note">The first {Math.min(SHOWN, result.table.length)} rows are shown; the CSV has all of them.</p>}
        </>
      )}
    </Panel>
  )
}

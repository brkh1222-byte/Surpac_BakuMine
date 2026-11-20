/**
 * Compositing dialog (P6.T3): length, mode (downhole / bench / domain) and the domain
 * table and column. Runs the same code as `bakumine composite`; the result is added to the
 * project as a composites object.
 */
import { useState } from 'react'
import { ofKind, opsApi, useOps } from '../../ops/api'
import { Error, Panel, Select, Text, message, opts } from './common'

export function CompositePanel() {
  const summary = useOps((s) => s.summary)
  const databases = ofKind(summary, 'database')
  const [chosen, setChosen] = useState('')
  const [table, setTable] = useState('')
  const [mode, setMode] = useState('downhole')
  const [length, setLength] = useState('2')
  const [bench, setBench] = useState('')
  const [reference, setReference] = useState('0')
  const [domainTable, setDomainTable] = useState('')
  const [domainColumn, setDomainColumn] = useState('')
  const [name, setName] = useState('composites')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const db = databases.find((d) => d.id === chosen) ?? databases[0]
  const tables = db?.tables ?? {}
  const tableNames = Object.keys(tables)
  const sampleTable = tables[table] ? table : (tableNames.includes('assay') ? 'assay' : tableNames[0] ?? '')
  const domainSource = domainTable || sampleTable
  const domainColumns = tables[domainSource]?.columns.filter((c) => !['hole_id', 'depth_from', 'depth_to'].includes(c)) ?? []

  const run = async () => {
    if (!db) return setError('Import a drillhole database first')
    const body: Record<string, unknown> = { database: db.id, table: sampleTable, mode, name, domain_table: domainTable || undefined, domain_column: domainColumn || undefined }
    if (mode === 'bench') {
      if (!(Number(bench) > 0)) return setError('Bench height must be positive')
      body.bench_height = Number(bench)
      body.bench_reference = Number(reference) || 0
    } else {
      if (!(Number(length) > 0)) return setError('Length must be positive')
      body.length = Number(length)
    }
    if (mode === 'domain' && !domainColumn) return setError('Pick the domain column')
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await opsApi.composite(body)
      setResult(`Saved “${String(r.name)}”: ${String(r.rows)} composites in ${String(r.holes)} holes, mean length ${Number(r.length_mean).toFixed(2)} m.`)
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel id="composite" title="Composite">
      {databases.length === 0 && <p className="sidebar-note">Import a drillhole database first.</p>}
      {databases.length > 1 && <Select label="Database" id="composite-db" value={db?.id ?? ''} onChange={setChosen} options={databases.map((d) => ({ value: d.id, label: d.name }))} />}
      <Select label="Samples" id="composite-table" value={sampleTable} onChange={setTable} options={opts(tableNames)} />
      <Select label="Mode" id="composite-mode" value={mode} onChange={setMode} options={opts(['downhole', 'bench', 'domain'])} />
      {mode === 'bench' ? (
        <>
          <Text label="Bench (m)" id="composite-bench" value={bench} onChange={setBench} />
          <Text label="Reference RL" id="composite-reference" value={reference} onChange={setReference} />
        </>
      ) : (
        <Text label="Length (m)" id="composite-length" value={length} onChange={setLength} />
      )}
      <Select label="Domain table" id="composite-domain-table" value={domainTable} onChange={(v) => { setDomainTable(v); setDomainColumn('') }} options={opts(tableNames, '(samples table)')} />
      <Select label="Domain column" id="composite-domain-column" value={domainColumn} onChange={setDomainColumn} options={opts(domainColumns, '(none)')} />
      <Text label="Name" id="composite-name" value={name} onChange={setName} />
      <div className="control">
        <button type="button" data-testid="composite-run" disabled={busy || !db} onClick={() => void run()}>{busy ? 'Running…' : 'Composite'}</button>
      </div>
      <Error message={error} id="composite-error" />
      {result && <p className="sidebar-note" data-testid="composite-result">{result}</p>}
    </Panel>
  )
}

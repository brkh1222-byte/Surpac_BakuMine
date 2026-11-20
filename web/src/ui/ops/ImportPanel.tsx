/**
 * Import wizard (P6.T2): upload collar / survey / assay / geology CSVs, check the suggested
 * column map (editable), preview rows, import + validate (the same code as `bakumine import`
 * and `bakumine validate`), and read the validation report. The database is saved in the project.
 */
import { useState } from 'react'
import { opsApi, useOps, type ImportResult, type UploadPreview } from '../../ops/api'
import { Check, Error, Panel, Select, Text, message } from './common'

const TABLES = [
  { name: 'collar', required: true },
  { name: 'survey', required: true },
  { name: 'assay', required: false },
  { name: 'geology', required: false },
]
const CANONICAL: Record<string, string[]> = {
  collar: ['hole_id', 'x', 'y', 'z', 'max_depth'],
  survey: ['hole_id', 'depth', 'azimuth', 'dip'],
  interval: ['hole_id', 'depth_from', 'depth_to'],
}

function TableUpload({ name, required, upload, onUpload, onMap, onRemove }: {
  name: string
  required: boolean
  upload: UploadPreview | undefined
  onUpload: (file: File) => void
  onMap: (source: string, target: string) => void
  onRemove: () => void
}) {
  const known = CANONICAL[name] ?? CANONICAL.interval
  return (
    <fieldset className="control report-set" data-testid={`import-table-${name}`}>
      <legend>{name}{required ? ' *' : ''}</legend>
      <input type="file" accept=".csv,.txt,.xlsx" data-testid={`import-file-${name}`} onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
      {upload && (
        <>
          <span className="sidebar-note" data-testid={`import-rows-${name}`}>{upload.file}: {upload.rows} rows</span>
          <button type="button" onClick={onRemove}>Remove</button>
          {upload.missing.length > 0 && (
            <span className="sidebar-error" data-testid={`import-missing-${name}`}>No column for: {upload.missing.join(', ')}</span>
          )}
          <table className="report-table" data-testid={`import-map-${name}`}>
            <thead><tr><th>Source column</th><th>Maps to</th></tr></thead>
            <tbody>
              {upload.columns.map((c) => (
                <tr key={c}>
                  <td>{c}</td>
                  <td>
                    <input
                      type="text"
                      list={`canon-${name}`}
                      aria-label={`${name} ${c}`}
                      data-testid={`import-map-${name}-${c}`}
                      value={upload.mapping[c] ?? ''}
                      placeholder="(ignore)"
                      onChange={(e) => onMap(c, e.target.value)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <datalist id={`canon-${name}`}>{known.map((k) => <option key={k} value={k} />)}</datalist>
          <div className="report-table-wrap">
            <table className="report-table" data-testid={`import-preview-${name}`}>
              <thead><tr>{upload.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {upload.preview.map((row, i) => (
                  <tr key={i}>{upload.columns.map((c) => <td key={c}>{row[c]}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </fieldset>
  )
}

export function ImportPanel() {
  const [uploads, setUploads] = useState<Record<string, UploadPreview>>({})
  const [name, setName] = useState('drillholes')
  const [lengthUnit, setLengthUnit] = useState('m')
  const [dip, setDip] = useState('negative_down')
  const [order, setOrder] = useState('xyz')
  const [toMetres, setToMetres] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ImportResult | null>(null)

  const upload = async (table: string, file: File) => {
    setError(null)
    try {
      const preview = await opsApi.upload(table, file)
      setUploads((u) => ({ ...u, [table]: preview }))
    } catch (e) {
      setError(`${table}: ${message(e)}`)
    }
  }
  const remove = (table: string) => {
    void opsApi.removeUpload(table).catch(() => undefined)
    setUploads((u) => Object.fromEntries(Object.entries(u).filter(([k]) => k !== table)))
  }
  const map = (table: string, source: string, target: string) =>
    setUploads((u) => {
      const mapping = { ...u[table].mapping }
      if (target.trim()) mapping[source] = target.trim()
      else delete mapping[source]
      return { ...u, [table]: { ...u[table], mapping } }
    })

  const run = async () => {
    if (!uploads.collar || !uploads.survey) return setError('Upload the collar and survey tables first')
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const maps = Object.fromEntries(Object.entries(uploads).map(([t, u]) => [t, u.mapping]))
      const options = { length_unit: lengthUnit, convert_to_metres: toMetres, dip_convention: dip, coordinate_order: order }
      setResult(await opsApi.importDb({ name, maps, options }))
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel id="import" title="Import drillholes">
      {TABLES.map((t) => (
        <TableUpload
          key={t.name}
          name={t.name}
          required={t.required}
          upload={uploads[t.name]}
          onUpload={(f) => void upload(t.name, f)}
          onMap={(s, v) => map(t.name, s, v)}
          onRemove={() => remove(t.name)}
        />
      ))}
      <Text label="Name" id="import-name" value={name} onChange={setName} />
      <Select label="Length unit" id="import-unit" value={lengthUnit} onChange={setLengthUnit} options={[{ value: 'm', label: 'm' }, { value: 'ft', label: 'ft' }, { value: 'ftUS', label: 'ftUS' }]} />
      <Check label="Convert to metres" id="import-tometres" checked={toMetres} onChange={setToMetres} />
      <Select label="Dip" id="import-dip" value={dip} onChange={setDip} options={[{ value: 'negative_down', label: 'negative down' }, { value: 'positive_down', label: 'positive down' }]} />
      <Text label="Coord. order" id="import-order" value={order} onChange={setOrder} placeholder="xyz (yxz = northing first)" />
      <div className="control">
        <button type="button" data-testid="import-run" disabled={busy} onClick={() => void run()}>{busy ? 'Importing…' : 'Import and validate'}</button>
      </div>
      <Error message={error} id="import-error" />
      {result && (
        <div data-testid="import-result">
          <p className="sidebar-note" data-testid="import-summary">
            Saved “{result.name}”: {result.n_holes} holes, {result.errors} errors, {result.warnings} warnings, {result.infos} infos.
          </p>
          <div className="report-table-wrap">
            <table className="report-table" data-testid="import-counts">
              <thead><tr><th>Check</th><th>Severity</th><th>Count</th></tr></thead>
              <tbody>
                {result.counts.map((c) => (
                  <tr key={c.code} title={c.description}><td>{c.code}</td><td>{c.severity}</td><td>{c.count}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.issues.length > 0 && (
            <div className="report-table-wrap">
              <table className="report-table" data-testid="import-issues">
                <thead><tr><th>Hole</th><th>Table</th><th>Code</th><th>Message</th></tr></thead>
                <tbody>
                  {result.issues.map((x, i) => (
                    <tr key={i}><td>{x.hole_id}</td><td>{x.table}</td><td>{x.code}</td><td style={{ textAlign: 'left' }}>{x.message}</td></tr>
                  ))}
                </tbody>
              </table>
              {result.truncated && <p className="sidebar-note">Only the first {result.issues.length} issues are listed.</p>}
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}

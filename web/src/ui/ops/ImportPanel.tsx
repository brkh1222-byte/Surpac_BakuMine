/**
 * Import wizard (P6.T2): upload collar / survey / assay / geology tables (CSV, XLSX, XLS; also
 * the CSVs of `bakumine extract-tables`), say how each file is laid out, check the suggested
 * column map (editable), import + validate (the same code as `bakumine import` and
 * `bakumine validate`), and read the reports. The database is saved in the project together
 * with the import recipe.
 *
 * Per file: the sheets to stack (one sheet per hole), a preview of each sheet, the header row
 * (one or two rows), the hole id from the sheet name, depths from the sample number, QC rows.
 * Per table: several files are merged. For all: hole id spelling, CRS of longitude / latitude.
 */
import { useState } from 'react'
import { opsApi, useOps, type ImportResult, type UploadPreview } from '../../ops/api'
import { emptyForm, holeIdRule, matchSheets, sourceSpec, type SourceForm } from '../../ops/importSpec'
import { Check, Error, Panel, Select, Text, message } from './common'

const TABLES = [
  { name: 'collar', required: true, hint: '' },
  { name: 'survey', required: false, hint: 'without it: from the collar table’s azimuth and dip' },
  { name: 'assay', required: false, hint: '' },
  { name: 'geology', required: false, hint: '' },
]
const CANONICAL: Record<string, string[]> = {
  collar: ['hole_id', 'x', 'y', 'z', 'max_depth', 'azimuth', 'dip', 'lon', 'lat', 'lon_deg', 'lon_min', 'lon_sec', 'lat_deg', 'lat_min', 'lat_sec'],
  survey: ['hole_id', 'depth', 'azimuth', 'dip'],
  interval: ['hole_id', 'depth_from', 'depth_to', 'sample_id', 'sample_type', 'sample_length'],
}
const ACCEPT = '.csv,.txt,.tsv,.xlsx,.xlsm,.xls'
/** a row of its own inside a wrapping fieldset (the sidebar is narrow) */
const ROW = { flexBasis: '100%', width: '100%', margin: 0 } as const
/** a fieldset is as wide as its widest table unless told otherwise */
const FIT = { minWidth: 0, maxWidth: '100%' } as const

interface Source {
  preview: UploadPreview
  original: string
  form: SourceForm
}

function counts(values: Record<string, number>): string {
  return Object.entries(values).map(([k, n]) => `${k} ${n}`).join(', ')
}

function SourceCard({ table, index, source, onForm, onMap, onMapDone }: {
  table: string
  index: number
  source: Source
  /** a layout choice changed: `resetMap` drops an edited column map (the header changed) */
  onForm: (patch: Partial<SourceForm>, resetMap?: boolean) => void
  onMap: (column: string, target: string) => void
  onMapDone: () => void
}) {
  const { preview, form } = source
  const key = index === 0 ? table : `${table}-${index}`
  const interval = preview.kind === 'interval'
  const known = CANONICAL[preview.kind] ?? CANONICAL.interval
  const names = preview.sheets.map((s) => s.name)
  const selected = form.sheets ?? preview.selected.map((s) => s.name)
  const [pattern, setPattern] = useState('')
  const toggle = (name: string, on: boolean) => {
    const next = names.filter((n) => (n === name ? on : selected.includes(n)))
    if (next.length) onForm({ sheets: next }, true)
  }
  const check = preview.sample_depth
  const headerEnd = preview.grid_header_row + preview.grid_header_rows - 1
  return (
    <div data-testid={`import-source-${key}`} style={{ width: '100%', minWidth: 0 }}>
      <span className="sidebar-note" style={{ ...ROW, display: 'block' }} data-testid={`import-rows-${key}`}>{source.original || preview.file}: {preview.rows} rows{preview.selected.length > 1 ? ` from ${preview.selected.length} sheets` : ''}</span>
      {names.length > 1 && (
        <fieldset className="control report-set" style={FIT} data-testid={`import-sheets-${key}`}>
          <legend>Sheets to stack ({selected.length} of {names.length}); click a name to preview it</legend>
          <input type="text" aria-label={`${key} sheet pattern`} data-testid={`import-sheet-pattern-${key}`} value={pattern} placeholder="pattern, e.g. DH*" onChange={(e) => setPattern(e.target.value)} />
          <button type="button" data-testid={`import-sheet-apply-${key}`} onClick={() => { const found = matchSheets(names, pattern); if (found.length) onForm({ sheets: found, previewSheet: found[0] }, true) }}>Select matching</button>
          <div className="report-table-wrap" style={{ maxHeight: 96, width: '100%' }}>
            {names.map((n, i) => (
              <span key={n} className="control-inline" style={{ marginRight: 8 }}>
                <input type="checkbox" aria-label={`${key} sheet ${n}`} data-testid={`import-sheet-${key}-${n}`} checked={selected.includes(n)} onChange={(e) => toggle(n, e.target.checked)} />
                <button type="button" data-testid={`import-sheet-show-${key}-${n}`} style={{ fontWeight: n === preview.grid_sheet ? 700 : 400 }} onClick={() => onForm({ previewSheet: n })}>{n}</button>
                <small> {preview.sheets[i].rows}</small>
              </span>
            ))}
          </div>
          <label className="control-inline" style={ROW}>
            <input type="checkbox" data-testid={`import-hole-from-sheet-${key}`} checked={form.holeFromSheet} onChange={(e) => onForm({ holeFromSheet: e.target.checked }, true)} />
            Hole id = sheet name
          </label>
        </fieldset>
      )}
      <fieldset className="control report-set" style={FIT}>
        <legend>Header of {preview.grid_sheet ? `“${preview.grid_sheet}”` : 'the file'}: row {preview.grid_header_row}{preview.grid_header_rows === 2 ? `–${headerEnd}` : ''}{preview.grid_header_row === 0 ? ' (none)' : ''}; click a row number to change</legend>
        <label className="control-inline">Header row
          <input type="text" inputMode="numeric" size={3} aria-label={`${key} header row`} data-testid={`import-header-row-${key}`} value={form.headerRow} placeholder="auto" onChange={(e) => onForm({ headerRow: e.target.value.replace(/[^0-9]/g, '') }, true)} />
        </label>
        <label className="control-inline">rows
          <select aria-label={`${key} header rows`} data-testid={`import-header-rows-${key}`} value={form.headerRows} onChange={(e) => onForm({ headerRows: e.target.value }, true)}>
            <option value="">auto</option>
            <option value="1">1</option>
            <option value="2">2 (group over sub-headers)</option>
          </select>
        </label>
        <div className="report-table-wrap" style={{ maxHeight: 150, width: '100%' }}>
          <table className="report-table" data-testid={`import-grid-${key}`}>
            <tbody>
              {preview.grid.map((row, r) => {
                const isHeader = r + 1 >= preview.grid_header_row && r + 1 <= headerEnd
                return (
                  <tr key={r} style={isHeader ? { fontWeight: 700 } : undefined} data-header={isHeader ? 'true' : undefined}>
                    <td><button type="button" title="Use as the header row" data-testid={`import-grid-row-${key}-${r + 1}`} onClick={() => onForm({ headerRow: String(r + 1) }, true)}>{r + 1}</button></td>
                    {row.map((cell, c) => <td key={c} style={{ textAlign: 'left' }}>{cell}</td>)}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </fieldset>
      {interval && (
        <fieldset className="control report-set" style={FIT}>
          <legend>Rows</legend>
          <label className="control-inline" style={ROW}>
            <input type="checkbox" data-testid={`import-sample-depth-${key}`} checked={form.sampleDepth} onChange={(e) => onForm({ sampleDepth: e.target.checked })} />
            Depths from the sample number where from / to are empty (to = number after the hole id, from = to − length)
          </label>
          {form.sampleDepth && (
            <>
              <label className="control-inline" style={ROW}>Sample length
                <input type="text" inputMode="decimal" size={4} data-testid={`import-sample-length-${key}`} value={form.sampleLength} onChange={(e) => onForm({ sampleLength: e.target.value })} />
              </label>
              <textarea rows={2} style={{ ...ROW, boxSizing: 'border-box' }} aria-label={`${key} sample number patterns`} data-testid={`import-sample-patterns-${key}`} value={form.samplePatterns} placeholder={'optional patterns, one per line, e.g. DH0*{holenum}-{holenum}(?P<to>\\d+)'} onChange={(e) => onForm({ samplePatterns: e.target.value })} />
            </>
          )}
          <label className="control-inline" style={ROW}>QC rows (blank, standard, duplicate)
            <select style={{ maxWidth: '100%' }} data-testid={`import-qc-${key}`} value={form.qc} onChange={(e) => onForm({ qc: e.target.value as 'separate' | 'off' })}>
              <option value="separate">set aside in a QC table</option>
              <option value="off">leave among the samples</option>
            </select>
          </label>
          {Object.keys(preview.qc).length > 0 && <span className="sidebar-note" style={ROW} data-testid={`import-qc-summary-${key}`}>QC rows set aside: {counts(preview.qc)}</span>}
          {check && (
            <span className="sidebar-note" style={ROW} data-testid={`import-depth-check-${key}`}>
              Depths from sample numbers: {check.rows} rows; not understood {check.unmatched}{check.unmatched_examples.length ? ` (${check.unmatched_examples.slice(0, 3).join(', ')})` : ''}, duplicates {check.duplicates}, backwards {check.backwards}, gaps {check.gaps}, overlaps {check.overlaps}
            </span>
          )}
        </fieldset>
      )}
      {preview.missing.length > 0 && (
        <span className="sidebar-error" data-testid={`import-missing-${key}`}>No column for: {preview.missing.join(', ')}</span>
      )}
      <div style={{ overflowX: 'auto', width: '100%' }}>
      <table className="report-table" data-testid={`import-map-${key}`}>
        <thead><tr><th>Source column</th><th>Maps to</th></tr></thead>
        <tbody>
          {preview.columns.map((c) => (
            <tr key={c}>
              <td>{c}</td>
              <td>
                <input
                  type="text"
                  list={`canon-${key}`}
                  aria-label={`${key} ${c}`}
                  data-testid={`import-map-${key}-${c}`}
                  value={(form.map ?? preview.mapping)[c] ?? ''}
                  placeholder="(ignore)"
                  onChange={(e) => onMap(c, e.target.value)}
                  onBlur={() => form.map && onMapDone()}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <datalist id={`canon-${key}`}>{known.map((k) => <option key={k} value={k} />)}</datalist>
      <div className="report-table-wrap">
        <table className="report-table" data-testid={`import-preview-${key}`}>
          <thead><tr>{preview.prepared.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {preview.prepared.rows.map((row, i) => (
              <tr key={i}>{row.map((cell, c) => <td key={c}>{cell}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      {preview.notes.length > 0 && <span className="sidebar-note" style={ROW} data-testid={`import-source-notes-${key}`}>{preview.notes.join('; ')}</span>}
    </div>
  )
}

export function ImportPanel() {
  const [tables, setTables] = useState<Record<string, Source[]>>({})
  const [name, setName] = useState('drillholes')
  const [lengthUnit, setLengthUnit] = useState('m')
  const [dip, setDip] = useState('auto')
  const [order, setOrder] = useState('auto')
  const [toMetres, setToMetres] = useState(false)
  const [crs, setCrs] = useState('')
  const [holePattern, setHolePattern] = useState('')
  const [holeTemplate, setHoleTemplate] = useState('')
  const [precedence, setPrecedence] = useState('first')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ImportResult | null>(null)

  const put = (table: string, index: number, source: Source) =>
    setTables((t) => {
      const list = [...(t[table] ?? [])]
      list[index] = source
      return { ...t, [table]: list }
    })

  const upload = async (table: string, file: File, add: boolean) => {
    setError(null)
    try {
      const preview = await opsApi.upload(table, file, add)
      const source = { preview, original: file.name, form: emptyForm() }
      if (add) put(table, preview.index, source)
      else setTables((t) => ({ ...t, [table]: [source] }))
    } catch (e) {
      setError(`${table}: ${message(e)}`)
    }
  }
  const remove = (table: string) => {
    void opsApi.removeUpload(table).catch(() => undefined)
    setTables((t) => Object.fromEntries(Object.entries(t).filter(([k]) => k !== table)))
  }
  /** Ask the server what the file gives with this form; the form is kept when it fails. */
  const refresh = async (table: string, index: number, source: Source, form: SourceForm) => {
    put(table, index, { ...source, form })
    setError(null)
    try {
      const spec = sourceSpec(form, source.preview.kind === 'interval', true)
      const preview = await opsApi.previewUpload(table, index, spec, holeIdRule(holePattern, holeTemplate))
      put(table, index, { ...source, form, preview })
    } catch (e) {
      setError(`${table}: ${message(e)}`)
    }
  }
  const refreshAll = () => {
    for (const [table, sources] of Object.entries(tables)) sources.forEach((s, i) => void refresh(table, i, s, s.form))
  }

  const run = async () => {
    if (!tables.collar?.length) return setError('Upload the collar table first')
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const body = {
        name,
        tables: Object.fromEntries(Object.entries(tables).map(([t, sources]) => [t, sources.map((s) => sourceSpec(s.form, s.preview.kind === 'interval'))])),
        options: { length_unit: lengthUnit, convert_to_metres: toMetres, dip_convention: dip, coordinate_order: order, crs: crs.trim() || null },
        hole_id: holeIdRule(holePattern, holeTemplate),
        merge: { precedence },
      }
      setResult(await opsApi.importDb(body))
      await useOps.getState().changed()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel id="import" title="Import drillholes">
      {TABLES.map((t) => {
        const sources = tables[t.name] ?? []
        return (
          <fieldset className="control report-set" style={FIT} key={t.name} data-testid={`import-table-${t.name}`}>
            <legend>{t.name}{t.required ? ' *' : ''}{t.hint ? ` (${t.hint})` : ''}</legend>
            <input type="file" accept={ACCEPT} aria-label={`${t.name} file`} data-testid={`import-file-${t.name}`} onChange={(e) => e.target.files?.[0] && void upload(t.name, e.target.files[0], false)} />
            {sources.length > 0 && (
              <>
                <button type="button" onClick={() => remove(t.name)}>Remove</button>
                <label className="control-inline" style={ROW}>Add a file to merge
                  <input type="file" accept={ACCEPT} aria-label={`${t.name} more files`} data-testid={`import-add-${t.name}`} onChange={(e) => { if (e.target.files?.[0]) void upload(t.name, e.target.files[0], true); e.target.value = '' }} />
                </label>
              </>
            )}
            {sources.map((s, i) => (
              <SourceCard
                key={i}
                table={t.name}
                index={i}
                source={s}
                onForm={(patch, resetMap) => void refresh(t.name, i, s, { ...s.form, ...patch, ...(resetMap ? { map: null } : {}) })}
                onMap={(column, target) => {
                  const map = { ...(s.form.map ?? s.preview.mapping) }
                  if (target.trim()) map[column] = target.trim()
                  else delete map[column]
                  put(t.name, i, { ...s, form: { ...s.form, map } })
                }}
                onMapDone={() => void refresh(t.name, i, s, s.form)}
              />
            ))}
          </fieldset>
        )
      })}
      <Text label="Name" id="import-name" value={name} onChange={setName} />
      <Select label="Length unit" id="import-unit" value={lengthUnit} onChange={setLengthUnit} options={[{ value: 'm', label: 'm' }, { value: 'ft', label: 'ft' }, { value: 'ftUS', label: 'ftUS' }]} />
      <Check label="Convert to metres" id="import-tometres" checked={toMetres} onChange={setToMetres} />
      <Select label="Dip" id="import-dip" value={dip} onChange={setDip} options={[{ value: 'auto', label: 'auto (all positive = down)' }, { value: 'negative_down', label: 'negative down' }, { value: 'positive_down', label: 'positive down' }]} />
      <Select label="Coord. order" id="import-order" value={order} onChange={setOrder} options={[{ value: 'auto', label: 'auto (detect northing first)' }, { value: 'xyz', label: 'x = east, y = north' }, { value: 'yxz', label: 'x = north, y = east' }]} />
      <Text label="CRS of lon / lat" id="import-crs" value={crs} onChange={setCrs} placeholder="UTM zone of the data (or EPSG:32647)" />
      <Text label="Hole id pattern" id="import-hole-pattern" value={holePattern} onChange={setHolePattern} placeholder="e.g. DH[_-]?0*(\d+)" />
      <Text label="…written as" id="import-hole-template" value={holeTemplate} onChange={setHoleTemplate} placeholder="e.g. DH-{1:02d}" />
      <div className="control">
        <button type="button" data-testid="import-hole-apply" onClick={refreshAll}>Apply hole id pattern to the previews</button>
      </div>
      <Select label="Files disagree" id="import-precedence" value={precedence} onChange={setPrecedence} options={[{ value: 'first', label: 'first file wins' }, { value: 'last', label: 'last file wins' }]} />
      <div className="control">
        <button type="button" data-testid="import-run" disabled={busy} onClick={() => void run()}>{busy ? 'Importing…' : 'Import and validate'}</button>
      </div>
      <Error message={error} id="import-error" />
      {result && (
        <div data-testid="import-result">
          <p className="sidebar-note" data-testid="import-summary">
            Saved “{result.name}”: {result.n_holes} holes, {result.errors} errors, {result.warnings} warnings, {result.infos} infos.
          </p>
          <p className="sidebar-note" data-testid="import-tables">
            Rows: {counts(result.tables)}. Dip {result.dip_convention.replace('_', ' ')}, coordinate order {result.coordinate_order}.
          </p>
          {Object.entries(result.qc).map(([t, c]) => (
            <p className="sidebar-note" key={t} data-testid={`import-qc-result-${t}`}>{t}: QC rows kept apart: {counts(c)}</p>
          ))}
          {Object.entries(result.sample_depth).map(([t, checks]) => checks.map((c, i) => (
            <p className="sidebar-note" key={`${t}-${i}`} data-testid={`import-depth-result-${t}`}>{t}: {c.rows} depths from sample numbers; not understood {c.unmatched}, duplicates {c.duplicates}, backwards {c.backwards}</p>
          )))}
          {result.survey && <p className="sidebar-note" data-testid="import-survey-result">Survey made from the collar table: {result.survey.generated_from_collar} holes{result.survey.holes_without_azimuth_or_dip.length ? `; no azimuth / dip for ${result.survey.holes_without_azimuth_or_dip.join(', ')}` : ''}</p>}
          {result.coordinates && (
            <p className="sidebar-note" data-testid="import-coordinates">
              Longitude / latitude → {result.coordinates.crs}: {result.coordinates.converted} converted, {result.coordinates.checked} checked against the projected columns, {result.coordinates.disagree} differ by more than {result.coordinates.tolerance} m
              {result.coordinates.rows.length ? ` (${result.coordinates.rows.map((r) => `${r.hole_id}: ${r.difference} m`).join(', ')})` : ''}
            </p>
          )}
          {Object.entries(result.conflicts).map(([t, c]) => (
            <div key={t} data-testid={`import-conflicts-${t}`}>
              <p className="sidebar-note">{t}: files merged ({c.precedence} file wins): {c.merged} rows joined, {c.filled} cells filled, {c.conflicts} conflicts</p>
              {c.rows.length > 0 && (
                <div className="report-table-wrap">
                  <table className="report-table">
                    <thead><tr>{c.keys.map((k) => <th key={k}>{k}</th>)}<th>Column</th><th>Kept</th><th>Dropped</th><th>From</th></tr></thead>
                    <tbody>
                      {c.rows.map((r, i) => (
                        <tr key={i}>{c.keys.map((k) => <td key={k}>{String(r[k])}</td>)}<td>{String(r.column)}</td><td>{String(r.kept)}</td><td>{String(r.dropped)}</td><td>{String(r.dropped_source)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
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
          {result.notes.length > 0 && (
            <details data-testid="import-notes">
              <summary className="sidebar-note">What the importer changed or could not read ({result.notes.length})</summary>
              <ul className="sidebar-note">{result.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
            </details>
          )}
          <details data-testid="import-recipe">
            <summary className="sidebar-note">Import recipe (saved with the database; replay with <code>bakumine import --recipe</code>)</summary>
            <pre className="sidebar-note" style={{ maxHeight: 200, overflow: 'auto' }}>{JSON.stringify(result.recipe, null, 1)}</pre>
          </details>
        </div>
      )}
    </Panel>
  )
}

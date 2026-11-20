/**
 * Report panel: volume / tonnes / grade of the selected block model, with grade attribute,
 * density, group-by, cut-offs and bench height. The table is computed by the server with
 * `bakumine.core.report.report` (the same code as `bakumine blockmodel report`); Export CSV
 * downloads that CSV. `Where` takes the same expression as the CLI's `--where`.
 */
import { useState } from 'react'
import { buildReportRequest, fetchReport, fetchReportCsv, formatCell, type ReportSettings, type ReportTable } from '../report/report'
import { useBlocks } from '../state/blocks'

const EMPTY: ReportSettings = { grades: [], density: '', groupBy: [], cutoffs: '', benchHeight: '', where: '' }

export function ReportPanel() {
  const models = useBlocks((s) => s.models)
  const ids = Object.keys(models)
  const [chosen, setChosen] = useState<string | null>(null)
  const [settings, setSettings] = useState<ReportSettings>(EMPTY)
  const [table, setTable] = useState<ReportTable | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [open, setOpen] = useState(false)

  const id = chosen && models[chosen] ? chosen : ids[0]
  const model = id ? models[id] : undefined
  if (!model || !id) return null
  const numeric = model.attributes.filter((a) => a.type === 'numeric')
  const categorical = model.attributes.filter((a) => a.type === 'categorical')
  const units: Record<string, string | null> = {}
  for (const a of numeric) if (a.type === 'numeric') units[a.name] = a.unit
  const patch = (p: Partial<ReportSettings>) => setSettings((s) => ({ ...s, ...p }))
  const toggle = (list: string[], name: string) => (list.includes(name) ? list.filter((n) => n !== name) : [...list, name])

  const request = () => buildReportRequest(settings, units)

  const run = async () => {
    const body = request()
    if (typeof body === 'string') return setError(body)
    setBusy(true)
    setError(null)
    try {
      setTable(await fetchReport(id, body))
    } catch (e) {
      setTable(null)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const exportCsv = async () => {
    const body = request()
    if (typeof body === 'string') return setError(body)
    try {
      const text = await fetchReportCsv(id, body)
      const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }))
      const link = document.createElement('a')
      link.href = url
      link.download = `${id}-report.csv`
      link.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <section className="report-panel" data-testid="report-panel" aria-label="Report">
      <button type="button" className="report-toggle" aria-expanded={open} onClick={() => setOpen(!open)} data-testid="report-toggle">
        Report {open ? '▾' : '▸'}
      </button>
      {open && (
        <div className="object-controls">
          {ids.length > 1 && (
            <label className="control">
              <span>Model</span>
              <select value={id} onChange={(e) => { setChosen(e.target.value); setTable(null) }}>
                {ids.map((m) => <option key={m} value={m}>{models[m].name}</option>)}
              </select>
            </label>
          )}
          <fieldset className="control report-set">
            <legend>Grade</legend>
            {numeric.map((a) => (
              <label key={a.name} className="control-inline">
                <input
                  type="checkbox"
                  data-testid={`report-grade-${a.name}`}
                  checked={settings.grades.includes(a.name)}
                  onChange={() => patch({ grades: toggle(settings.grades, a.name) })}
                />
                {a.name}
              </label>
            ))}
          </fieldset>
          <label className="control">
            <span>Density</span>
            <input
              type="text"
              list="report-density"
              data-testid="report-density"
              placeholder="attribute or t/m3"
              value={settings.density}
              onChange={(e) => patch({ density: e.target.value })}
            />
            <datalist id="report-density">{numeric.map((a) => <option key={a.name} value={a.name} />)}</datalist>
          </label>
          <fieldset className="control report-set">
            <legend>Group by</legend>
            {categorical.map((a) => (
              <label key={a.name} className="control-inline">
                <input
                  type="checkbox"
                  data-testid={`report-group-${a.name}`}
                  checked={settings.groupBy.includes(a.name)}
                  onChange={() => patch({ groupBy: toggle(settings.groupBy, a.name) })}
                />
                {a.name}
              </label>
            ))}
          </fieldset>
          <label className="control">
            <span>Cut-offs</span>
            <input type="text" data-testid="report-cutoffs" placeholder="0.5, 1" value={settings.cutoffs} onChange={(e) => patch({ cutoffs: e.target.value })} />
          </label>
          <label className="control">
            <span>Bench (m)</span>
            <input type="text" data-testid="report-bench" placeholder="none" value={settings.benchHeight} onChange={(e) => patch({ benchHeight: e.target.value })} />
          </label>
          <label className="control">
            <span>Where</span>
            <input type="text" data-testid="report-where" placeholder="au > 0.3" value={settings.where} onChange={(e) => patch({ where: e.target.value })} />
          </label>
          <div className="control">
            <button type="button" data-testid="report-run" disabled={busy} onClick={run}>{busy ? 'Running…' : 'Run report'}</button>
            <button type="button" data-testid="report-export" onClick={exportCsv}>Export CSV</button>
          </div>
          {error && <p className="sidebar-error" role="alert" data-testid="report-error">{error}</p>}
          {table && (
            <div className="report-table-wrap">
              <table className="report-table" data-testid="report-table">
                <thead>
                  <tr>{table.columns.map((c) => <th key={c}>{c}</th>)}</tr>
                </thead>
                <tbody>
                  {table.rows.map((row, i) => (
                    <tr key={i} className={row.row === 'total' ? 'is-total' : undefined}>
                      {table.columns.map((c) => <td key={c}>{formatCell(c, row[c] ?? null)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
              {table.notes.map((n) => <p key={n} className="sidebar-note">{n}</p>)}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

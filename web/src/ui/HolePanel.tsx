import { formatCoord } from '../scene/format'
import { useDisplay } from '../state/display'

/** Details of the selected drillhole: collar, depth and its intervals in the active table. */
export function HolePanel() {
  const hole = useDisplay((s) => s.selection)
  const clear = useDisplay((s) => s.setSelection)
  if (!hole) return null
  return (
    <aside className="hole-panel" data-testid="hole-panel" aria-label={`Drillhole ${hole.holeId}`}>
      <header>
        <span className="hole-panel-title" data-testid="hole-panel-title">{hole.holeId}</span>
        <button type="button" onClick={() => clear(null)} aria-label="Close hole details" title="Close">
          ×
        </button>
      </header>
      <dl className="hole-facts">
        <div><dt>Collar E</dt><dd>{formatCoord(hole.collar[0], 2)}</dd></div>
        <div><dt>Collar N</dt><dd>{formatCoord(hole.collar[1], 2)}</dd></div>
        <div><dt>Collar Z</dt><dd>{formatCoord(hole.collar[2], 2)}</dd></div>
        <div><dt>Depth</dt><dd data-testid="hole-panel-depth">{hole.maxDepth.toFixed(2)} m</dd></div>
        <div><dt>Intervals</dt><dd>{hole.table ? `${hole.rows.length} in ${hole.table}` : '–'}</dd></div>
      </dl>
      {hole.rows.length > 0 && (
        <div className="hole-table">
          <table>
            <thead>
              <tr>
                <th>From</th>
                <th>To</th>
                {hole.columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {hole.rows.map((row, i) => (
                <tr key={i}>
                  <td>
                    {row.color && <i style={{ background: row.color }} />}
                    {row.from.toFixed(2)}
                  </td>
                  <td>{row.to.toFixed(2)}</td>
                  {row.values.map((v, k) => (
                    <td key={k} className={v === 'no value' ? 'is-missing' : undefined}>{v === 'no value' ? '–' : v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {hole.table && hole.rows.length === 0 && <p className="hole-empty">No {hole.table} intervals in this hole.</p>}
    </aside>
  )
}

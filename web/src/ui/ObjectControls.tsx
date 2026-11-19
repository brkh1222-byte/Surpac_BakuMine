/** Per-object display controls shown under an object's row in the sidebar tree. */
import { attributeKey, useDisplay } from '../state/display'

function DrillholeControls({ id }: { id: string }) {
  const display = useDisplay((s) => s.drillholes[id])
  const setColorBy = useDisplay((s) => s.setColorBy)
  const setShowLabels = useDisplay((s) => s.setShowLabels)
  if (!display) return null
  return (
    <div className="object-controls" data-testid={`controls-${id}`}>
      <label className="control">
        <span>Colour by</span>
        <select
          data-testid={`colour-by-${id}`}
          value={display.colorBy ?? ''}
          onChange={(e) => setColorBy(id, e.target.value || null)}
        >
          <option value="">None</option>
          {display.attributes.map((a) => {
            const key = attributeKey(a)
            const unit = a.type === 'numeric' && a.unit ? ` (${a.unit})` : ''
            return (
              <option key={key} value={key}>
                {a.name}
                {unit} · {a.table}
              </option>
            )
          })}
        </select>
      </label>
      <label className="control control-check">
        <input
          type="checkbox"
          data-testid={`labels-${id}`}
          checked={display.showLabels}
          onChange={(e) => setShowLabels(id, e.target.checked)}
        />
        <span>Hole labels</span>
      </label>
    </div>
  )
}

export function ObjectControls({ id }: { id: string }) {
  return <DrillholeControls id={id} />
}

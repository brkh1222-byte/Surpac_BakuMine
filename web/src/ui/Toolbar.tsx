import { useEffect, useState } from 'react'
import { view, type StandardView } from '../scene/view'
import { MAX_EXAGGERATION, MIN_EXAGGERATION, useViewer, type Projection } from '../state/store'

const VIEWS: { id: StandardView; label: string; title: string }[] = [
  { id: 'plan', label: 'Plan', title: 'Plan view: looking down, north up' },
  { id: 'north', label: 'North', title: 'Section view: looking north' },
  { id: 'east', label: 'East', title: 'Section view: looking east' },
  { id: 'iso', label: 'Iso', title: 'Isometric view from the south-west' },
]

const PROJECTIONS: { id: Projection; label: string }[] = [
  { id: 'perspective', label: 'Perspective' },
  { id: 'orthographic', label: 'Orthographic' },
]

/** Free-text number field that commits on Enter or blur. */
function ExaggerationInput() {
  const value = useViewer((s) => s.verticalExaggeration)
  const setValue = useViewer((s) => s.setVerticalExaggeration)
  const [text, setText] = useState(String(value))
  useEffect(() => setText(String(value)), [value])

  const commit = () => {
    const parsed = Number(text.replace(',', '.'))
    if (Number.isFinite(parsed) && parsed > 0) setValue(parsed)
    else setText(String(value))
  }

  return (
    <label className="field" title="Vertical exaggeration: stretches elevations in the view. Coordinates shown are always true.">
      <span>Vertical ×</span>
      <input
        data-testid="ve-input"
        type="number"
        min={MIN_EXAGGERATION}
        max={MAX_EXAGGERATION}
        step={0.5}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit()
        }}
      />
    </label>
  )
}

export function Toolbar() {
  const projection = useViewer((s) => s.projection)
  const setProjection = useViewer((s) => s.setProjection)
  const theme = useViewer((s) => s.theme)
  const toggleTheme = useViewer((s) => s.toggleTheme)

  return (
    <header className="toolbar" role="toolbar" aria-label="View">
      <div className="brand">
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M1.5 11.5 8 14.5l6.5-3V5L8 2 1.5 5z M1.5 5 8 8l6.5-3 M8 8v6.5" />
        </svg>
        <span>BakuMine</span>
      </div>

      <div className="group" role="group" aria-label="Standard views">
        {VIEWS.map((v) => (
          <button key={v.id} type="button" title={v.title} data-testid={`view-${v.id}`} onClick={() => view.rig?.setView(v.id)}>
            {v.label}
          </button>
        ))}
      </div>

      <button
        type="button"
        className="standalone"
        title="Zoom to the extents of all visible objects"
        data-testid="zoom-extents"
        onClick={() => view.rig?.zoomToExtents()}
      >
        Zoom to extents
      </button>

      <div className="group" role="group" aria-label="Projection">
        {PROJECTIONS.map((p) => (
          <button
            key={p.id}
            type="button"
            aria-pressed={projection === p.id}
            data-testid={`proj-${p.id}`}
            onClick={() => setProjection(p.id)}
          >
            {p.label}
          </button>
        ))}
      </div>

      <ExaggerationInput />

      <div className="spacer" />

      <button
        type="button"
        className="standalone"
        data-testid="theme-toggle"
        title="Switch between dark and light"
        onClick={toggleTheme}
      >
        {theme === 'dark' ? 'Light theme' : 'Dark theme'}
      </button>
    </header>
  )
}

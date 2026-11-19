/**
 * Legend of one colour mapping. Give it the `LegendSpec` the layer coloured its buffers
 * with (colour/legend.ts); pass `onChange` to let the user edit the range and switch
 * between a continuous ramp and classes. Reused by every layer that colours by value.
 */
import { useEffect, useState } from 'react'
import {
  continuousTicks,
  legendRows,
  legendTitle,
  type LegendSpec,
  type NumericLegendSpec,
  type NumericSettings,
} from '../colour/legend'
import { MISSING_COLOR, rampGradient } from '../colour/ramps'
import { useDisplay } from '../state/display'
import { useViewer } from '../state/store'

/** Number field that commits on Enter or blur and snaps back when the input is unusable. */
function RangeInput({ label, value, testId, onCommit }: { label: string; value: number; testId: string; onCommit: (value: number) => void }) {
  const [text, setText] = useState(String(value))
  useEffect(() => setText(String(value)), [value])
  const commit = () => {
    const parsed = Number(text.replace(',', '.'))
    if (text.trim() !== '' && Number.isFinite(parsed) && parsed !== value) onCommit(parsed)
    else setText(String(value))
  }
  return (
    <label className="legend-field">
      <span>{label}</span>
      <input
        data-testid={testId}
        type="text"
        inputMode="decimal"
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

function NumericControls({ spec, onChange, onReset }: { spec: NumericLegendSpec; onChange: (patch: Partial<NumericSettings>) => void; onReset?: () => void }) {
  return (
    <div className="legend-controls">
      <RangeInput label="Min" value={spec.min} testId="legend-min" onCommit={(min) => onChange({ min })} />
      <RangeInput label="Max" value={spec.max} testId="legend-max" onCommit={(max) => onChange({ max })} />
      <div className="group legend-mode" role="group" aria-label="Legend type">
        <button type="button" data-testid="legend-continuous" aria-pressed={spec.mode === 'continuous'} onClick={() => onChange({ mode: 'continuous' })}>
          Ramp
        </button>
        <button type="button" data-testid="legend-stepped" aria-pressed={spec.mode === 'stepped'} onClick={() => onChange({ mode: 'stepped' })}>
          Classes
        </button>
      </div>
      {onReset && (
        <button type="button" className="legend-reset" onClick={onReset} title="Back to the default range">
          Reset
        </button>
      )}
    </div>
  )
}

export interface LegendProps {
  spec: LegendSpec
  /** Shown small next to the title: the object the legend belongs to. */
  subject?: string
  onChange?: (patch: Partial<NumericSettings>) => void
  onReset?: () => void
}

export function Legend({ spec, subject, onChange, onReset }: LegendProps) {
  const theme = useViewer((s) => s.theme)
  const continuous = spec.kind === 'numeric' && spec.mode === 'continuous'
  const rows = continuous ? [] : legendRows(spec, theme)

  return (
    <section className="legend" data-testid="legend" data-kind={spec.kind === 'numeric' ? spec.mode : 'categorical'} aria-label={`Legend: ${legendTitle(spec)}`}>
      <header>
        <span className="legend-title" data-testid="legend-title">{legendTitle(spec)}</span>
        {subject && <span className="legend-subject">{subject}</span>}
      </header>
      {continuous && spec.kind === 'numeric' && (
        <>
          <div className="legend-ramp" style={{ background: rampGradient(spec.ramp, theme) }} />
          <div className="legend-ticks" data-testid="legend-ticks">
            {continuousTicks(spec).map((tick, i) => (
              <span key={i}>{tick}</span>
            ))}
          </div>
          {spec.hasMissing && (
            <div className="legend-row">
              <i style={{ background: MISSING_COLOR[theme] }} />
              <span>No value</span>
            </div>
          )}
        </>
      )}
      {rows.length > 0 && (
        <div className="legend-rows" data-testid="legend-rows">
          {rows.map((row) => (
            <div className="legend-row" key={row.label}>
              <i style={{ background: row.color }} />
              <span>{row.label}</span>
            </div>
          ))}
        </div>
      )}
      {spec.kind === 'numeric' && onChange && <NumericControls spec={spec} onChange={onChange} onReset={onReset} />}
    </section>
  )
}

/** Every active legend, stacked in the corner of the viewport. */
export function LegendStack() {
  const legends = useDisplay((s) => s.legends)
  const ids = Object.keys(legends)
  if (ids.length === 0) return null
  return (
    <div className="legends" data-testid="legends">
      {ids.map((id) => (
        <Legend key={id} {...legends[id]} />
      ))}
    </div>
  )
}

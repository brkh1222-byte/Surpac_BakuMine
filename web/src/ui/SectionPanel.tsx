/**
 * The Section panel at the foot of the sidebar. Off: one row with the ways to start a
 * section (presets, draw in plan, through the selected hole). On: where the section is,
 * step buttons and a slider across the data, the numbers, clip / ghost, and the section
 * view. Shortcuts: ] or PageUp steps forward, [ or PageDown back, Esc stops drawing.
 */
import { useEffect, useRef, useState } from 'react'
import { describeSection } from '../section/describe'
import { view } from '../scene/view'
import { useDisplay } from '../state/display'
import { sectionTravel, sectionTravelRange, useSection, type SectionPreset } from '../state/section'
import { allBounds, useViewer } from '../state/store'

const PRESETS: { id: SectionPreset; label: string; title: string }[] = [
  { id: 'we', label: 'W–E', title: 'West–east section through the middle of the data, looking north' },
  { id: 'sn', label: 'S–N', title: 'South–north section through the middle of the data, looking west' },
  { id: 'plan', label: 'Plan', title: 'Horizontal section (bench) at the middle elevation, looking down' },
]

/** Number field that commits on Enter or blur and snaps back when the text is not a number. */
function NumberField({ label, title, value, onCommit, testId, decimals = 2 }: {
  label: string
  title?: string
  value: number
  onCommit: (value: number) => void
  testId: string
  decimals?: number
}) {
  const shown = String(Math.round(value * 10 ** decimals) / 10 ** decimals)
  const [text, setText] = useState(shown)
  useEffect(() => setText(shown), [shown])
  const latest = useRef(shown)
  latest.current = shown
  const commit = () => {
    const parsed = Number(text.replace(',', '.').replace(/\s/g, ''))
    if (text.trim() !== '' && Number.isFinite(parsed)) onCommit(parsed)
    // Show what the store made of it (clamped, refused, or unchanged), once it has.
    queueMicrotask(() => setText(latest.current))
  }
  return (
    <label className="section-field" title={title}>
      <span>{label}</span>
      <input
        type="text"
        inputMode="decimal"
        data-testid={testId}
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

function typing(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (!el || !el.tagName) return false
  return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable
}

/** Step and cancel shortcuts, active while the pointer is not in a text field. */
function useSectionKeys(): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const section = useSection.getState()
      if (event.key === 'Escape' && section.draft) {
        section.cancelDrawing()
        return
      }
      if (!section.enabled || event.ctrlKey || event.metaKey || event.altKey) return
      // Range inputs use PageUp / PageDown themselves; text fields take [ and ].
      if (typing(event.target) && (event.target as HTMLInputElement).type !== 'checkbox') return
      if (event.key === ']' || event.key === 'PageUp') section.stepBy(1)
      else if (event.key === '[' || event.key === 'PageDown') section.stepBy(-1)
      else return
      event.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}

export function SectionPanel() {
  useSectionKeys()
  const s = useSection()
  const selection = useDisplay((d) => d.selection)
  const objects = useViewer((v) => v.objects)
  const bounds = allBounds({ objects } as Parameters<typeof allBounds>[0])
  const [details, setDetails] = useState(false)

  const draw = () => {
    if (s.draft) {
      s.cancelDrawing()
      return
    }
    // The line is drawn on the plan.
    useViewer.getState().setProjection('orthographic')
    view.rig?.setView('plan')
    s.startDrawing()
  }
  const throughHole = () => {
    if (!selection) return
    const trace = view.holeTraces.get(selection.objectId)?.(selection.hole)
    if (trace) s.throughHole(trace)
  }

  const range = bounds ? sectionTravelRange(s.plane, bounds) : null
  const travel = sectionTravel(s.plane)
  const [e, n, z] = s.plane.centre

  return (
    <section className={'section-panel' + (s.enabled ? ' is-on' : '')} aria-label="Section" data-testid="section-panel">
      <header>
        <label className="section-toggle" title="Cut the scene with a section plane">
          <input
            type="checkbox"
            data-testid="section-enabled"
            checked={s.enabled}
            onChange={(event) => s.setEnabled(event.target.checked)}
          />
          <span>Section</span>
        </label>
        <div className="group" role="group" aria-label="New section">
          {PRESETS.map((p) => (
            <button key={p.id} type="button" title={p.title} data-testid={`section-preset-${p.id}`} onClick={() => s.preset(p.id)}>
              {p.label}
            </button>
          ))}
          <button
            type="button"
            title="Draw the section line: click its two ends in plan view"
            data-testid="section-draw"
            aria-pressed={s.draft !== null}
            onClick={draw}
          >
            Draw
          </button>
          <button
            type="button"
            title={selection ? `Vertical section through hole ${selection.holeId}` : 'Select a drillhole first: a vertical section through it'}
            data-testid="section-through-hole"
            disabled={!selection}
            onClick={throughHole}
          >
            Hole
          </button>
        </div>
      </header>

      {s.enabled && (
        <div className="section-body">
          <p className="section-summary" data-testid="section-summary">
            {describeSection(s.plane, s)}
          </p>

          <div className="section-step">
            <button type="button" title="Step back ( [ or PageDown )" data-testid="section-back" onClick={() => s.stepBy(-1)}>
              ◀
            </button>
            <input
              type="range"
              aria-label="Section position across the data"
              data-testid="section-slider"
              min={range ? Math.min(range[0], travel) : travel - 1}
              max={range ? Math.max(range[1], travel) : travel + 1}
              step="any"
              value={travel}
              disabled={!range}
              onChange={(event) => s.setTravel(Number(event.target.value))}
            />
            <button type="button" title="Step forward ( ] or PageUp )" data-testid="section-forward" onClick={() => s.stepBy(1)}>
              ▶
            </button>
          </div>

          <div className="section-grid">
            <NumberField label="Step" title="Distance of one step, m" value={s.step} onCommit={s.setStep} testId="section-step" />
            <NumberField
              label="Front"
              title="Slab thickness towards the viewer, m"
              value={s.front}
              onCommit={(front) => s.setThickness({ front })}
              testId="section-front"
            />
            <NumberField
              label="Back"
              title="Slab thickness away from the viewer, m"
              value={s.back}
              onCommit={(back) => s.setThickness({ back })}
              testId="section-back-thickness"
            />
          </div>

          <div className="section-row">
            <div className="group" role="group" aria-label="Outside the slab">
              <button
                type="button"
                title="Hide everything outside the slab"
                aria-pressed={s.mode === 'clip'}
                data-testid="section-mode-clip"
                onClick={() => s.setMode('clip')}
              >
                Clip
              </button>
              <button
                type="button"
                title="Show the whole scene, dimmed outside the slab"
                aria-pressed={s.mode === 'ghost'}
                data-testid="section-mode-ghost"
                onClick={() => s.setMode('ghost')}
              >
                Ghost
              </button>
            </div>
            {s.locked ? (
              <button
                type="button"
                className="section-view is-locked"
                title="Leave the section view: rotation is free again"
                data-testid="section-unlock"
                onClick={() => s.setLocked(false)}
              >
                Unlock view
              </button>
            ) : (
              <button
                type="button"
                className="section-view"
                title="Look square-on at the section (orthographic, rotation locked)"
                data-testid="section-look"
                onClick={s.lookAt}
              >
                Look at section
              </button>
            )}
          </div>

          <button type="button" className="section-more" aria-expanded={details} data-testid="section-details" onClick={() => setDetails(!details)}>
            {details ? '▾' : '▸'} Position and orientation
          </button>
          {details && (
            <div className="section-grid section-grid-wide">
              <NumberField label="E" title="Easting of a point on the plane" value={e} onCommit={(v) => s.setPlane({ centre: [v, n, z] })} testId="section-e" />
              <NumberField label="N" title="Northing of a point on the plane" value={n} onCommit={(v) => s.setPlane({ centre: [e, v, z] })} testId="section-n" />
              <NumberField label="Z" title="Elevation of a point on the plane" value={z} onCommit={(v) => s.setPlane({ centre: [e, n, v] })} testId="section-z" />
              <NumberField
                label="Azimuth"
                title="Azimuth of the section line, degrees clockwise from north (90 = west to east, looking north)"
                value={s.plane.azimuth}
                onCommit={(azimuth) => s.setPlane({ azimuth })}
                testId="section-azimuth"
              />
              <NumberField
                label="Dip"
                title="Dip of the plane, degrees: 90 = vertical section, 0 = plan"
                value={s.plane.dip}
                onCommit={(dip) => s.setPlane({ dip })}
                testId="section-dip"
              />
            </div>
          )}
        </div>
      )}
    </section>
  )
}

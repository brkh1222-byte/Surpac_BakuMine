/**
 * Design form: the body of `POST /api/v1/blast/simulate`. Empty optional fields are sent
 * as null ("let the rules of thumb decide"). "Simulate" asks the API; without it the demo
 * bench is regenerated from the same parameters (and labelled as demo data).
 */
import { useEffect, useState } from 'react'
import { blastCentre } from '../scene/Rings'
import { useBlast } from '../state'
import type { DesignParams, ExplosiveName, PatternType, TimingType, Xy } from '../types'

function Num({
  label, value, onChange, unit, step = 'any', min, nullable = false, testId,
}: {
  label: string
  value: number | null
  onChange: (v: number | null) => void
  unit?: string
  step?: number | 'any'
  min?: number
  nullable?: boolean
  testId?: string
}) {
  return (
    <label className="blast-field">
      <span>
        {label}
        {unit && <i> {unit}</i>}
      </span>
      <input
        type="number"
        step={step}
        min={min}
        data-testid={testId}
        value={value ?? ''}
        placeholder={nullable ? 'auto' : undefined}
        onChange={(e) => {
          const text = e.target.value.trim()
          if (text === '') {
            if (nullable) onChange(null)
            return
          }
          const v = Number(text)
          if (Number.isFinite(v)) onChange(v)
        }}
      />
    </label>
  )
}

function Choice<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: readonly (readonly [T, string])[]; onChange: (v: T) => void }) {
  return (
    <label className="blast-field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map(([id, text]) => (
          <option key={id} value={id}>
            {text}
          </option>
        ))}
      </select>
    </label>
  )
}

const PATTERNS: readonly (readonly [PatternType, string])[] = [['staggered', 'Staggered'], ['square', 'Square']]
const EXPLOSIVES: readonly (readonly [ExplosiveName, string])[] = [['ANFO', 'ANFO'], ['Emulsion', 'Emulsion']]
const TIMINGS: readonly (readonly [TimingType, string])[] = [
  ['row_by_row', 'Row by row'], ['echelon', 'Echelon'], ['v_cut', 'V-cut'], ['hole_by_hole', 'Hole by hole'],
]

/** "x, y" per line -> polygon; null when empty, 'error' when it cannot be read. */
export function parsePolygon(text: string): Xy[] | null | 'error' {
  const lines = text.split('\n').map((l) => l.trim()).filter((l) => l !== '')
  if (lines.length === 0) return null
  const out: Xy[] = []
  for (const line of lines) {
    const parts = line.split(/[\s,;]+/).map(Number)
    if (parts.length < 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return 'error'
    out.push([parts[0], parts[1]])
  }
  return out.length >= 3 ? out : 'error'
}

export function DesignPanel() {
  const design = useBlast((s) => s.design)
  const status = useBlast((s) => s.status)
  const bundle = useBlast((s) => s.bundle)
  const error = useBlast((s) => s.error)
  const { patchDesign, simulate } = useBlast.getState()
  const [polygonText, setPolygonText] = useState(() => (design.polygon ?? []).map((p) => p.join(', ')).join('\n'))
  // The API's defaults (and a submitted polygon) arrive after the first render.
  useEffect(() => {
    setPolygonText((design.polygon ?? []).map((p) => p.join(', ')).join('\n'))
  }, [design.polygon])
  const polygon = parsePolygon(polygonText)

  const rock = (patch: Partial<DesignParams['rock']>) => patchDesign({ rock: { ...design.rock, ...patch } })
  const timing = (patch: Partial<DesignParams['timing']>) => patchDesign({ timing: { ...design.timing, ...patch } })
  const monitor = (i: number, patch: Partial<DesignParams['monitors'][number]>) =>
    patchDesign({ monitors: design.monitors.map((m, k) => (k === i ? { ...m, ...patch } : m)) })
  const addMonitor = () => {
    const c = bundle ? blastCentre(bundle) : [0, 0, 0]
    patchDesign({
      monitors: [...design.monitors, { name: `M${design.monitors.length + 1}`, xyz: [Math.round(c[0] + 150), Math.round(c[1] + 150), Math.round(c[2])], limit_mm_s: 10 }],
    })
  }

  return (
    <form
      className="blast-section blast-design"
      data-testid="blast-design"
      onSubmit={(e) => {
        e.preventDefault()
        if (polygon === 'error') return
        patchDesign({ polygon })
        void simulate()
      }}
    >
      <h3>Design</h3>
      <label className="blast-field blast-field-wide">
        <span>Name</span>
        <input type="text" value={design.name} onChange={(e) => patchDesign({ name: e.target.value })} />
      </label>
      <div className="blast-grid">
        <Num label="Bench height" unit="m" value={design.bench_height} min={1} onChange={(v) => v !== null && v > 0 && patchDesign({ bench_height: v })} testId="blast-bench-height" />
        <Num label="Diameter" unit="mm" value={Math.round(design.diameter * 1e4) / 10} min={30} onChange={(v) => v !== null && v > 0 && patchDesign({ diameter: v / 1000 })} />
        <Num label="Burden" unit="m" value={design.burden} nullable onChange={(v) => patchDesign({ burden: v })} />
        <Num label="Spacing" unit="m" value={design.spacing} nullable onChange={(v) => patchDesign({ spacing: v })} />
        <Num label="Stemming" unit="m" value={design.stemming} nullable onChange={(v) => patchDesign({ stemming: v })} />
        <Num label="Subdrill" unit="m" value={design.subdrill} nullable onChange={(v) => patchDesign({ subdrill: v })} />
        <Choice label="Pattern" value={design.pattern} options={PATTERNS} onChange={(v) => patchDesign({ pattern: v })} />
        <Choice label="Explosive" value={design.explosive} options={EXPLOSIVES} onChange={(v) => patchDesign({ explosive: v })} />
        <Num label="Face azimuth" unit="°" value={design.face_azimuth} onChange={(v) => v !== null && patchDesign({ face_azimuth: v })} />
      </div>

      <h4>Timing</h4>
      <div className="blast-grid">
        <Choice label="Sequence" value={design.timing.type} options={TIMINGS} onChange={(v) => timing({ type: v })} />
        <Num label="Inter-hole" unit="ms" value={design.timing.inter_hole_ms} min={0} onChange={(v) => v !== null && v >= 0 && timing({ inter_hole_ms: v })} />
        <Num label="Inter-row" unit="ms" value={design.timing.inter_row_ms} min={0} onChange={(v) => v !== null && v >= 0 && timing({ inter_row_ms: v })} />
      </div>

      <h4>Rock</h4>
      <div className="blast-grid">
        <Num label="Density" unit="kg/m³" value={design.rock.density} onChange={(v) => v !== null && v > 0 && rock({ density: v })} />
        <Num label="UCS" unit="MPa" value={design.rock.ucs_mpa} onChange={(v) => v !== null && v > 0 && rock({ ucs_mpa: v })} />
        <Num label="Tensile" unit="MPa" value={design.rock.tensile_mpa} onChange={(v) => v !== null && v > 0 && rock({ tensile_mpa: v })} />
        <Num label="Young's" unit="GPa" value={design.rock.youngs_gpa} onChange={(v) => v !== null && v > 0 && rock({ youngs_gpa: v })} />
        <Num label="Poisson" value={design.rock.poisson} step={0.01} onChange={(v) => v !== null && v > 0 && v < 0.5 && rock({ poisson: v })} />
        <Num label="Rock factor A" value={design.rock.rock_factor} nullable onChange={(v) => rock({ rock_factor: v })} />
      </div>

      <h4>Bench polygon</h4>
      <textarea
        className={'blast-polygon' + (polygon === 'error' ? ' is-bad' : '')}
        rows={3}
        placeholder="east, north per line — empty = demo bench"
        value={polygonText}
        onChange={(e) => setPolygonText(e.target.value)}
      />
      {polygon === 'error' && <p className="blast-note blast-note-warn">At least three "east, north" lines are needed.</p>}

      <h4>
        Monitors
        <button type="button" className="blast-mini" onClick={addMonitor}>
          Add
        </button>
      </h4>
      {design.monitors.length === 0 && <p className="blast-note">None: the simulation uses its own default points.</p>}
      {design.monitors.map((m, i) => (
        <div key={i} className="blast-monitor">
          <input type="text" aria-label="Monitor name" value={m.name} onChange={(e) => monitor(i, { name: e.target.value })} />
          {([0, 1, 2] as const).map((k) => (
            <input
              key={k}
              type="number"
              step="any"
              aria-label={['East', 'North', 'Elevation'][k]}
              title={['East', 'North', 'Elevation'][k]}
              value={m.xyz[k]}
              onChange={(e) => {
                const v = Number(e.target.value)
                if (e.target.value === '' || !Number.isFinite(v)) return
                const xyz = [...m.xyz] as typeof m.xyz
                xyz[k] = v
                monitor(i, { xyz })
              }}
            />
          ))}
          <input
            type="number"
            step="any"
            aria-label="PPV limit, mm/s"
            title="PPV limit, mm/s"
            value={m.limit_mm_s}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (e.target.value !== '' && v > 0) monitor(i, { limit_mm_s: v })
            }}
          />
          <button type="button" className="blast-mini" title="Remove" onClick={() => patchDesign({ monitors: design.monitors.filter((_, k) => k !== i) })}>
            ×
          </button>
        </div>
      ))}

      <button type="submit" className="blast-simulate" data-testid="blast-simulate" disabled={status === 'loading' || polygon === 'error'}>
        {status === 'loading' && <i className="blast-spinner" aria-hidden="true" />}
        {status === 'loading' ? 'Simulating…' : 'Simulate'}
      </button>
      {error && (
        <p className="blast-error" role="alert" data-testid="blast-error">
          {error}
        </p>
      )}
    </form>
  )
}

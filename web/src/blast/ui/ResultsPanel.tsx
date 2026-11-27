/** Results of the loaded bundle: summary cards, fragmentation, monitors, model notes. */
import { useBlast } from '../state'
import type { BlastBundle, ModelInfo } from '../types'
import { FragCurve } from './FragCurve'

function fmt(v: number | null | undefined, digits = 0): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '–'
  return v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

const has = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

const size = (m: number) => (Number.isFinite(m) ? (m >= 1 ? `${fmt(m, 2)} m` : `${fmt(m * 1000)} mm`) : '–')

function Card({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: 'ok' | 'bad' }) {
  return (
    <div className={'blast-card' + (tone ? ` is-${tone}` : '')}>
      <span>{label}</span>
      <b>
        {value}
        {unit && <i> {unit}</i>}
      </b>
    </div>
  )
}

function constants(info: ModelInfo): string {
  return Object.entries(info)
    .filter(([key]) => key !== 'calibrated')
    .map(([key, value]) => `${key} = ${typeof value === 'number' ? Number(value.toPrecision(4)) : String(value)}`)
    .join(', ')
}

function Cards({ bundle }: { bundle: BlastBundle }) {
  const s = bundle.summary
  const worst = bundle.monitors.reduce<(typeof bundle.monitors)[number] | null>((w, m) => {
    if (m.limit_mm_s == null) return w
    return !w || m.ppv_mm_s / m.limit_mm_s > w.ppv_mm_s / (w.limit_mm_s ?? 1) ? m : w
  }, null)
  return (
    <div className="blast-cards" data-testid="blast-cards">
      <Card label="Volume" value={fmt(s.volume_m3)} unit="m³" />
      <Card label="Tonnes" value={fmt(s.tonnes)} unit="t" />
      <Card label="Explosive" value={fmt(s.explosive_kg)} unit="kg" />
      <Card label="Powder factor" value={fmt(s.powder_factor_kg_m3, 2)} unit={`kg/m³ · ${fmt(s.powder_factor_kg_t, 2)} kg/t`} />
      <Card label="Holes" value={fmt(s.holes)} />
      <Card label="Drilled" value={fmt(s.drilled_m)} unit="m" />
      <Card label="x50" value={size(s.x50_m)} />
      <Card label="P80" value={size(s.p80_m)} />
      <Card label="Uniformity n" value={fmt(s.uniformity_n, 2)} />
      <Card label="Oversize" value={fmt(s.oversize_fraction * 100, 1)} unit="%" />
      <Card label="MIC (8 ms)" value={fmt(s.mic_kg)} unit="kg" />
      <Card label="Duration" value={fmt(s.duration_ms)} unit="ms" />
      {worst && (
        <Card
          label={`Max PPV · ${worst.name}`}
          value={fmt(worst.ppv_mm_s, 1)}
          unit={`/ ${fmt(worst.limit_mm_s, 1)} mm/s`}
          tone={worst.ok === false ? 'bad' : 'ok'}
        />
      )}
      <Card label="Flyrock max" value={fmt(s.flyrock_max_m)} unit="m" />
      <Card label="Clearance" value={fmt(s.clearance_m)} unit="m" />
      <Card label="Back-break" value={fmt(s.backbreak_m, 1)} unit="m" />
      {has(s.burden_m) && has(s.spacing_m) && <Card label="Burden × spacing" value={`${fmt(s.burden_m, 2)} × ${fmt(s.spacing_m, 2)}`} unit="m" />}
      {has(s.rows) && <Card label="Rows" value={fmt(s.rows)} />}
      {has(s.stemming_m) && <Card label="Stemming" value={fmt(s.stemming_m, 2)} unit="m" />}
      {has(s.subdrill_m) && <Card label="Subdrill" value={fmt(s.subdrill_m, 2)} unit="m" />}
      {has(s.burden_relief_ms_m) && <Card label="Burden relief" value={fmt(s.burden_relief_ms_m, 1)} unit="ms/m" />}
      {has(s.throw_com_m) && <Card label="Throw (centre of mass)" value={fmt(s.throw_com_m, 1)} unit="m" />}
      {has(s.throw_max_m) && <Card label="Throw max" value={fmt(s.throw_max_m, 1)} unit="m" />}
      {has(s.muckpile_max_height_m) && <Card label="Muckpile height" value={fmt(s.muckpile_max_height_m, 1)} unit="m" />}
    </div>
  )
}

export function ResultsPanel() {
  const bundle = useBlast((s) => s.bundle)
  const source = useBlast((s) => s.source)
  const reason = useBlast((s) => s.reason)
  if (!bundle) return <p className="blast-note">No result loaded.</p>
  const models = Object.entries(bundle.summary.models ?? {})
  const notes = Array.isArray(bundle.summary.notes) ? bundle.summary.notes.filter((n): n is string => typeof n === 'string') : []
  return (
    <>
      <section className="blast-section">
        <h3>Results · {bundle.design.name}</h3>
        {source === 'demo' && (
          <p className="blast-note blast-note-warn" data-testid="blast-demo-note">
            Demo data made in the browser, not the engineering model. {reason}
          </p>
        )}
        <Cards bundle={bundle} />
      </section>

      <section className="blast-section">
        <h3>Fragmentation (cumulative passing, %)</h3>
        <FragCurve frag={bundle.fragmentation} p80={bundle.summary.p80_m} />
      </section>

      {bundle.monitors.length > 0 && (
        <section className="blast-section">
          <h3>Monitors</h3>
          <table className="blast-table">
            <thead>
              <tr>
                <th>Point</th>
                <th>Dist, m</th>
                <th>PPV / limit, mm/s</th>
                <th>Air, dBL</th>
              </tr>
            </thead>
            <tbody>
              {bundle.monitors.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td>{fmt(m.distance_m)}</td>
                  <td className={m.ok === false ? 'is-bad' : m.ok === true ? 'is-ok' : undefined}>
                    {fmt(m.ppv_mm_s, 1)} / {fmt(m.limit_mm_s, 1)}
                  </td>
                  <td className={m.airblast_ok === false ? 'is-bad' : m.airblast_ok === true ? 'is-ok' : undefined}>
                    {fmt(m.airblast_dbl)}
                    {m.airblast_limit_dbl != null && ` / ${fmt(m.airblast_limit_dbl)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="blast-section">
        <h3>Flyrock (range by mechanism)</h3>
        <p className="blast-note">
          Face burst {fmt(bundle.flyrock.face_burst_m)} m · cratering {fmt(bundle.flyrock.cratering_m)} m · stemming ejection{' '}
          {fmt(bundle.flyrock.stemming_ejection_m)} m · safety factor {fmt(bundle.flyrock.safety_factor, 1)}
        </p>
      </section>

      <section className="blast-section">
        <h3>Model notes</h3>
        {notes.length > 0 && (
          <ul className="blast-notes" data-testid="blast-notes">
            {notes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        )}
        <p className="blast-note">
          Empirical engineering models, not a first-principles simulation. Constants marked uncalibrated are published typical values until a
          site calibration is supplied.
        </p>
        <ul className="blast-models" data-testid="blast-models">
          {models.map(([name, info]) => (
            <li key={name}>
              <b>{name.replace(/_/g, ' ')}</b>
              <span className={info.calibrated ? 'is-ok' : 'is-warn'}>{info.calibrated ? 'calibrated' : 'uncalibrated site constants'}</span>
              <i>{constants(info)}</i>
            </li>
          ))}
        </ul>
      </section>
    </>
  )
}

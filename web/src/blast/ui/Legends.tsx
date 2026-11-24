/** Legends of the 3D view: firing time, PPV (log scale) and damage classes. */
import { DAMAGE_COLORS, PPV_RANGE, PPV_STOPS, TIMING_STOPS } from '../math'
import { useBlast } from '../state'
import type { BlastBundle } from '../types'

function Ramp({ title, stops, ticks }: { title: string; stops: readonly string[]; ticks: string[] }) {
  return (
    <div className="blast-legend">
      <div className="blast-legend-title">{title}</div>
      <div className="blast-legend-bar" style={{ background: `linear-gradient(to right, ${stops.join(', ')})` }} />
      <div className="blast-legend-ticks">
        {ticks.map((t, i) => (
          <span key={i}>{t}</span>
        ))}
      </div>
    </div>
  )
}

export function Legends({ bundle }: { bundle: BlastBundle }) {
  const layers = useBlast((s) => s.layers)
  const duration = Math.max(0, ...bundle.timeline.holes.map((h) => h.t_fire_ms))
  const { classes, thresholds_mm_s: thresholds } = bundle.damage
  return (
    <div className="blast-legends">
      {layers.holes && <Ramp title="Firing time, ms" stops={TIMING_STOPS} ticks={['0', `${Math.round(duration / 2)}`, `${Math.round(duration)}`]} />}
      {(layers.waves || layers.ppv) && (
        <Ramp title="PPV, mm/s (log)" stops={PPV_STOPS} ticks={[`${PPV_RANGE.min}`, '10', '100', `${PPV_RANGE.max}`]} />
      )}
      {layers.damage && (
        <div className="blast-legend">
          <div className="blast-legend-title">Damage (near-field PPV)</div>
          {classes.slice(1).map((name, i) => (
            <div key={name} className="blast-legend-row">
              <i style={{ background: DAMAGE_COLORS[Math.min(i + 1, DAMAGE_COLORS.length - 1)] }} />
              {name}
              {thresholds[i] !== undefined && <span>≥ {thresholds[i]} mm/s</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

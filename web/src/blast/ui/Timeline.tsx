/**
 * The player under the 3D view: play / pause, scrub, speed and loop, the time in ms, and
 * the charge fired per delay window (tallest bar = MIC) above the detonation half of the
 * slider.
 */
import { useEffect } from 'react'
import { detonationSplit, formatMs, sliderToTime, timeToSlider } from '../math'
import { SPEEDS, useBlast } from '../state'
import type { BlastBundle } from '../types'

const W = 1000
const H = 34

function Envelope({ bundle, split, end }: { bundle: BlastBundle; split: number; end: number }) {
  const envelope = bundle.envelope
  const x = (t: number) => timeToSlider(t, split, end) * W
  if (!envelope || envelope.bins.length === 0) {
    // No envelope in the bundle: one tick per hole at its firing time.
    return (
      <svg className="blast-envelope" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-label="Firing times">
        {bundle.timeline.holes.map((h) => (
          <line key={h.id} x1={x(h.t_fire_ms)} x2={x(h.t_fire_ms)} y1={H * 0.35} y2={H} className="blast-envelope-tick" />
        ))}
      </svg>
    )
  }
  const top = Math.max(envelope.mic_kg, ...envelope.bins.map((b) => b.kg), 1e-9)
  const tallest = envelope.bins.reduce((best, b) => (b.kg > best.kg ? b : best), envelope.bins[0])
  return (
    <svg
      className="blast-envelope"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`Charge per ${envelope.window_ms} ms window, MIC ${envelope.mic_kg} kg`}
      data-testid="blast-envelope"
    >
      {envelope.bins.map((b) => {
        const x0 = x(b.t_ms)
        const width = Math.max(2, x(b.t_ms + envelope.window_ms) - x0 - 1)
        const height = Math.max(1, (b.kg / top) * (H - 2))
        return (
          <rect key={b.t_ms} x={x0} y={H - height} width={width} height={height} className={b === tallest ? 'is-mic' : undefined}>
            <title>{`${b.t_ms}–${b.t_ms + envelope.window_ms} ms: ${b.kg.toFixed(0)} kg, ${b.holes.length} hole${b.holes.length === 1 ? '' : 's'}`}</title>
          </rect>
        )
      })}
    </svg>
  )
}

export function Timeline() {
  const bundle = useBlast((s) => s.bundle)
  const t = useBlast((s) => s.t)
  const tEnd = useBlast((s) => s.tEnd)
  const last = useBlast((s) => s.lastFire)
  const playing = useBlast((s) => s.playing)
  const speed = useBlast((s) => s.speed)
  const auto = useBlast((s) => s.autoSpeed)
  const loop = useBlast((s) => s.loop)
  const split = detonationSplit(last, tEnd)

  // Space plays / pauses, unless a field has the focus.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (event.code !== 'Space' || (target && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(target.tagName))) return
      event.preventDefault()
      useBlast.getState().togglePlay()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const s = useBlast.getState()
  return (
    <footer className="blast-timeline" data-testid="blast-timeline">
      <div className="blast-transport">
        <button type="button" className="blast-play" data-testid="blast-play" disabled={!bundle} onClick={s.togglePlay} title="Play / pause (Space)">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button type="button" disabled={!bundle} title="Back to the first initiation" onClick={() => { s.setPlaying(false); s.setTime(0) }}>
          Reset
        </button>
        <span className="blast-time" data-testid="blast-time">
          {formatMs(t)} <i>/ {formatMs(tEnd)}</i>
        </span>
      </div>
      <div className="blast-track">
        {bundle && <Envelope bundle={bundle} split={split} end={tEnd} />}
        <input
          type="range"
          min={0}
          max={1}
          step={0.0005}
          value={timeToSlider(t, split, tEnd)}
          disabled={!bundle}
          aria-label="Time"
          data-testid="blast-scrub"
          onChange={(e) => {
            s.setPlaying(false)
            s.setTime(sliderToTime(Number(e.target.value), split, tEnd))
          }}
        />
        <div className="blast-track-scale">
          <span>0</span>
          <span>detonation · last hole {formatMs(last)}</span>
          <span className="blast-track-mid">{formatMs(split)}</span>
          <span>rock flight</span>
          <span>{formatMs(tEnd)}</span>
        </div>
      </div>
      <div className="blast-speed">
        <label title="Slow motion while the holes fire, real time for the rock flight">
          <input type="checkbox" checked={auto} onChange={(e) => s.setAutoSpeed(e.target.checked)} /> Auto
        </label>
        <select value={speed} aria-label="Playback speed" data-testid="blast-speed" onChange={(e) => s.setSpeed(Number(e.target.value))}>
          {SPEEDS.map((v) => (
            <option key={v} value={v}>
              {v}×
            </option>
          ))}
        </select>
        <label>
          <input type="checkbox" checked={loop} onChange={(e) => s.setLoop(e.target.checked)} /> Loop
        </label>
        <span className="blast-hint" title="A P-wave crosses the bench in a few milliseconds: use 0.01×–0.05× (or scrub) to see the wavefronts.">
          waves: 0.01×
        </span>
      </div>
    </footer>
  )
}

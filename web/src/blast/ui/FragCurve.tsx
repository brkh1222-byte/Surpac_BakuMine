/** Fragmentation curve: cumulative passing against size on a log axis. Plain SVG. */
import type { Fragmentation } from '../types'

const W = 280
const H = 150
const PAD = { left: 30, right: 8, top: 8, bottom: 22 }

export function FragCurve({ frag, p80 }: { frag: Fragmentation; p80: number }) {
  const pts = frag.sizes_m.map((x, i) => [x, frag.passing[i]] as const).filter(([x, p]) => x > 0 && Number.isFinite(p))
  if (pts.length < 2) return <p className="blast-note">No fragmentation curve in this result.</p>
  const lo = Math.log10(pts[0][0])
  const hi = Math.log10(pts[pts.length - 1][0])
  const x = (size: number) => PAD.left + ((Math.log10(size) - lo) / (hi - lo || 1)) * (W - PAD.left - PAD.right)
  const y = (p: number) => PAD.top + (1 - p) * (H - PAD.top - PAD.bottom)
  const path = pts.map(([s, p], i) => `${i ? 'L' : 'M'}${x(s).toFixed(1)} ${y(p).toFixed(1)}`).join(' ')
  const decades: number[] = []
  for (let d = Math.ceil(lo - 1e-9); d <= Math.floor(hi + 1e-9); d++) decades.push(d)
  const inside = (size: number) => size > 0 && Math.log10(size) >= lo && Math.log10(size) <= hi
  const mark = (size: number, p: number, label: string) =>
    inside(size) && (
      <g className="blast-frag-mark">
        <line x1={x(size)} x2={x(size)} y1={y(p)} y2={y(0)} />
        <line x1={x(Math.pow(10, lo))} x2={x(size)} y1={y(p)} y2={y(p)} />
        <circle cx={x(size)} cy={y(p)} r={2.6} />
        <text x={x(size) + 4} y={y(p) + 11}>{label}</text>
      </g>
    )
  const label = (size: number) => (size >= 1 ? `${size} m` : `${Math.round(size * 1000)} mm`)
  return (
    <svg className="blast-frag" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Fragmentation: cumulative passing against size" data-testid="blast-frag">
      {[0, 0.25, 0.5, 0.75, 1].map((p) => (
        <g key={p} className="blast-frag-grid">
          <line x1={PAD.left} x2={W - PAD.right} y1={y(p)} y2={y(p)} />
          <text x={PAD.left - 4} y={y(p) + 3} textAnchor="end">{p * 100}</text>
        </g>
      ))}
      {decades.map((d) => (
        <g key={d} className="blast-frag-grid">
          <line x1={x(10 ** d)} x2={x(10 ** d)} y1={PAD.top} y2={H - PAD.bottom} />
          <text x={x(10 ** d)} y={H - 8} textAnchor="middle">{label(10 ** d)}</text>
        </g>
      ))}
      {mark(frag.x50_m, 0.5, `x50 ${label(Number(frag.x50_m.toPrecision(2)))}`)}
      {mark(p80, 0.8, `P80 ${label(Number(p80.toPrecision(2)))}`)}
      <path d={path} className="blast-frag-line" />
    </svg>
  )
}

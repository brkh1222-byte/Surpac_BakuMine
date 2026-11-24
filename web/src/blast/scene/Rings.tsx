/** Circles on the ground (flyrock range, clearance zone, airblast levels) and monitor pins. */
import { useEffect, useMemo } from 'react'
import { BufferGeometry, Float32BufferAttribute } from 'three'
import { useViewer } from '../../state/store'
import { sampleGrid } from '../math'
import type { BlastBundle } from '../types'
import { BLAST_COLORS, type LabelItem, type LayerProps } from './common'
import { Labels } from './Labels'

/** Plan centre of the blast (mean collar) in world coordinates. */
export function blastCentre(bundle: BlastBundle): [number, number, number] {
  const holes = bundle.timeline.holes
  if (holes.length === 0) return [bundle.muckpile.origin[0], bundle.muckpile.origin[1], 0]
  const sum = holes.reduce((s, h) => [s[0] + h.collar[0], s[1] + h.collar[1], s[2] + h.collar[2]], [0, 0, 0])
  return [sum[0] / holes.length, sum[1] / holes.length, sum[2] / holes.length]
}

const SEGMENTS = 128

/** A circle draped on the PPV surface grid (flat at the collar level where the grid has no data). */
function useCircle(bundle: BlastBundle, origin: readonly number[], radius: number): BufferGeometry {
  const geometry = useMemo(() => {
    const [cx, cy, cz] = blastCentre(bundle)
    const pts: number[] = []
    for (let i = 0; i < SEGMENTS; i++) {
      const a = (i / SEGMENTS) * 2 * Math.PI
      const x = cx + radius * Math.cos(a)
      const y = cy + radius * Math.sin(a)
      const z = sampleGrid(bundle.ppv_surface, bundle.ppv_surface.z, x, y)
      pts.push(x - origin[0], y - origin[1], (Number.isFinite(z) ? z : cz) - origin[2] + 0.4)
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new Float32BufferAttribute(pts, 3))
    return g
  }, [bundle, origin, radius])
  useEffect(() => () => geometry.dispose(), [geometry])
  return geometry
}

function Circle({ bundle, origin, radius, color, opacity = 1 }: LayerProps & { radius: number; color: string; opacity?: number }) {
  const geometry = useCircle(bundle, origin, radius)
  return (
    <lineLoop geometry={geometry} renderOrder={5}>
      <lineBasicMaterial color={color} transparent opacity={opacity} depthTest={false} />
    </lineLoop>
  )
}

function northOf(bundle: BlastBundle, origin: readonly number[], radius: number): [number, number, number] {
  const [cx, cy, cz] = blastCentre(bundle)
  const z = sampleGrid(bundle.ppv_surface, bundle.ppv_surface.z, cx, cy + radius)
  return [cx - origin[0], cy + radius - origin[1], (Number.isFinite(z) ? z : cz) - origin[2] + 0.4]
}

export function FlyrockRings({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const { max_range_m: range, clearance_m: clearance } = bundle.flyrock
  const labels = useMemo<LabelItem[]>(
    () => [
      { position: northOf(bundle, origin, range), text: `Flyrock max ${range.toFixed(0)} m`, className: 'blast-label-ring' },
      { position: northOf(bundle, origin, clearance), text: `Clearance ${clearance.toFixed(0)} m`, className: 'blast-label-ring blast-label-danger' },
    ],
    [bundle, origin, range, clearance],
  )
  return (
    <>
      {range > 0 && <Circle bundle={bundle} origin={origin} radius={range} color={BLAST_COLORS[theme].ring} />}
      {clearance > 0 && <Circle bundle={bundle} origin={origin} radius={clearance} color={BLAST_COLORS[theme].clearance} />}
      <Labels items={labels} />
    </>
  )
}

export function AirblastRings({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const rings = bundle.airblast.rings
  const labels = useMemo<LabelItem[]>(
    () => rings.map((r) => ({ position: northOf(bundle, origin, -r.radius_m), text: `${r.dbl.toFixed(0)} dBL`, className: 'blast-label-ring blast-label-air' })),
    [bundle, origin, rings],
  )
  return (
    <>
      {rings.map((r, i) => (
        <Circle key={i} bundle={bundle} origin={origin} radius={r.radius_m} color={BLAST_COLORS[theme].air} opacity={0.8} />
      ))}
      <Labels items={labels} />
    </>
  )
}

/** Height of a monitor pin, m. */
const PIN = 14

export function Monitors({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const monitors = bundle.monitors
  const geometry = useMemo(() => {
    const pts: number[] = []
    for (const m of monitors) {
      const x = m.xyz[0] - origin[0]
      const y = m.xyz[1] - origin[1]
      const z = m.xyz[2] - origin[2]
      pts.push(x, y, z, x, y, z + PIN)
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new Float32BufferAttribute(pts, 3))
    return g
  }, [monitors, origin])
  useEffect(() => () => geometry.dispose(), [geometry])
  const labels = useMemo<LabelItem[]>(
    () =>
      monitors.map((m) => ({
        position: [m.xyz[0] - origin[0], m.xyz[1] - origin[1], m.xyz[2] - origin[2] + PIN],
        text: `${m.name}: ${m.ppv_mm_s.toFixed(m.ppv_mm_s < 10 ? 1 : 0)}${m.limit_mm_s != null ? ` / ${m.limit_mm_s}` : ''} mm/s`,
        className: 'blast-label-monitor ' + (m.ok === false ? 'is-bad' : m.ok === true ? 'is-ok' : ''),
      })),
    [monitors, origin],
  )
  return (
    <>
      <lineSegments geometry={geometry} renderOrder={5}>
        <lineBasicMaterial color={BLAST_COLORS[theme].ink} depthTest={false} />
      </lineSegments>
      {monitors.map((m, i) => (
        <mesh key={i} position={[m.xyz[0] - origin[0], m.xyz[1] - origin[1], m.xyz[2] - origin[2] + PIN]} renderOrder={5}>
          <sphereGeometry args={[1.2, 12, 8]} />
          <meshBasicMaterial color={m.ok === false ? '#e04b3f' : '#3fbf6f'} depthTest={false} />
        </mesh>
      ))}
      <Labels items={labels} />
    </>
  )
}

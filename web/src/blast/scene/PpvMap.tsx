/**
 * Peak particle velocity on the surface: the ppv_surface grid as a coloured drape (log
 * scale, the legend's range), revealed node by node at its arrival time, with iso-lines
 * at the monitors' limits.
 */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BufferAttribute, BufferGeometry, DoubleSide, Float32BufferAttribute } from 'three'
import { useViewer } from '../../state/store'
import { contourSegments, ppvColor } from '../math'
import { useBlast } from '../state'
import { BLAST_COLORS, linear, type LayerProps } from './common'

/** A node fades in over this long after the wave reaches it. */
const REVEAL_MS = 4
const ALPHA = 0.72

export function PpvMap({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const grid = bundle.ppv_surface

  const built = useMemo(() => {
    const [nx, ny] = grid.shape
    const n = nx * ny
    const position = new Float32Array(n * 3)
    const color = new Float32Array(n * 4)
    const ok = (i: number) => Number.isFinite(grid.z[i] ?? NaN) && Number.isFinite(grid.ppv_mm_s[i] ?? NaN)
    let last = 0
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const i = ix * ny + iy
        position[i * 3] = grid.origin[0] + ix * grid.spacing - origin[0]
        position[i * 3 + 1] = grid.origin[1] + iy * grid.spacing - origin[1]
        position[i * 3 + 2] = (grid.z[i] ?? 0) - origin[2] + 0.2
        const c = linear(ppvColor(grid.ppv_mm_s[i] ?? 0))
        color.set([c[0], c[1], c[2], 0], i * 4)
        const arrival = grid.arrival_ms[i]
        if (typeof arrival === 'number' && Number.isFinite(arrival) && arrival > last) last = arrival
      }
    }
    const index: number[] = []
    for (let ix = 0; ix + 1 < nx; ix++) {
      for (let iy = 0; iy + 1 < ny; iy++) {
        const a = ix * ny + iy
        const b = a + ny
        if (ok(a) && ok(b) && ok(b + 1) && ok(a + 1)) index.push(a, b, b + 1, a, b + 1, a + 1)
      }
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(position, 3))
    g.setAttribute('color', new BufferAttribute(color, 4))
    g.setIndex(index)

    const limits = [...new Set(bundle.monitors.map((m) => m.limit_mm_s).filter((v): v is number => typeof v === 'number' && v > 0))]
    const lines: number[] = []
    for (const limit of limits) {
      const world = contourSegments(grid, grid.ppv_mm_s, grid.z, limit)
      for (let i = 0; i < world.length; i += 3) lines.push(world[i] - origin[0], world[i + 1] - origin[1], world[i + 2] - origin[2] + 0.35)
    }
    const iso = new BufferGeometry()
    iso.setAttribute('position', new Float32BufferAttribute(lines, 3))
    return { g, iso, last: last + REVEAL_MS, state: { t: NaN } }
  }, [bundle, grid, origin])
  useEffect(
    () => () => {
      built.g.dispose()
      built.iso.dispose()
    },
    [built],
  )

  useFrame(() => {
    const t = Math.min(useBlast.getState().t, built.last)
    if (t === built.state.t) return
    built.state.t = t
    const color = built.g.getAttribute('color') as BufferAttribute
    const array = color.array as Float32Array
    for (let i = 0; i < color.count; i++) {
      const arrival = grid.arrival_ms[i]
      const f = typeof arrival === 'number' ? (t - arrival) / REVEAL_MS : 1
      array[i * 4 + 3] = f <= 0 ? 0 : f >= 1 ? ALPHA : ALPHA * f
    }
    color.needsUpdate = true
  })

  return (
    <>
      <mesh geometry={built.g} renderOrder={3}>
        <meshBasicMaterial vertexColors transparent depthWrite={false} side={DoubleSide} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
      </mesh>
      <lineSegments geometry={built.iso} renderOrder={4}>
        <lineBasicMaterial color={BLAST_COLORS[theme].ink} transparent opacity={0.85} depthTest={false} />
      </lineSegments>
    </>
  )
}

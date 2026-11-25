/**
 * Thrown rock and soil. The simulation moves parcels (throw.py); each parcel is drawn as a
 * cluster of fragments that follows the parcel's path (math.particlePosition) and spreads
 * by a small velocity difference of its own (realism.fragmentOffset). A fragment is an
 * irregular faceted block that tumbles while it flies and lies still once landed.
 *
 * - Sizes are drawn from the predicted Rosin-Rammler curve (realism.fragmentSize); a
 *   fragment is never drawn smaller than about a pixel (math.fragmentDrawSize).
 * - Parcels that start within realism.SOIL_THICKNESS_M of the bench top are soil: small
 *   brown clods with a wider spread. The thickness is assumed.
 * - With the satellite picture on, fragments have rock and soil colours; without it they
 *   keep the diagram colours (warm while flying, flyrock in the warning colour).
 */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BufferAttribute, IcosahedronGeometry, InstancedBufferAttribute, InstancedMesh, MeshStandardMaterial, type BufferGeometry } from 'three'
import { useViewer } from '../../state/store'
import { fragmentDrawSize, particlePosition, Phase, SETTLE_MS, smoothstep } from '../math'
import { useBlast } from '../state'
import { BLAST_COLORS, hexLinear, type LayerProps } from './common'
import { depthBelowCollars, fragmentOffset, fragmentSize, hash01, SOIL_SIZE_M, SOIL_THICKNESS_M } from './realism'

/** At most this many fragments are drawn. */
export const FRAGMENT_CAP = 26_000
/** At most this many fragments per parcel. */
export const CLUSTER_MAX = 10
/** Flyrock is drawn this many times larger than muck of the same size. */
const FLYROCK_BOOST = 2.5
/** Smallest drawn fragment as a fraction of its distance to the camera in the site view (about a pixel across). */
const SITE_FLOOR = 0.0006

/** Rock and soil tones of the site view (sRGB hex): fresh grey rock, weathered brown rock, pale dusty rock; soil. */
const ROCK_TONES = ['#8d8880', '#7b7368', '#a59f93', '#6a655f', '#928a7c'] as const
const SOIL_TONES = ['#6f5c48', '#7d6850', '#5f4f3e'] as const

/** An irregular block: an icosphere with every vertex pushed in or out, flat faces. */
export function rockGeometry(): BufferGeometry {
  const base = new IcosahedronGeometry(1, 1)
  const position = base.getAttribute('position') as BufferAttribute
  // The icosphere is not indexed: the same corner appears in several faces and must move the same way.
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i)
    const y = position.getY(i)
    const z = position.getZ(i)
    const key = Math.round(x * 97) * 31 + Math.round(y * 97) * 57 + Math.round(z * 97) * 131
    const r = 0.72 + 0.5 * hash01(key, 1)
    position.setXYZ(i, x * r, y * r, z * r)
  }
  base.computeVertexNormals() // not indexed: one normal per face
  return base
}

export function Throw({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const site = useBlast((s) => s.layers.satellite && s.source === 'api' && s.imagery !== null)
  const particles = bundle.throw.particles

  const built = useMemo(() => {
    const total = particles.t0_ms.length
    const cluster = Math.max(1, Math.min(CLUSTER_MAX, Math.floor(FRAGMENT_CAP / Math.max(1, total))))
    const step = Math.max(1, Math.ceil(total / FRAGMENT_CAP))
    const parents: number[] = []
    for (let i = 0; i < total; i++) if (i % step === 0 || particles.kind[i] === 1) parents.push(i)
    const count = parents.length * cluster
    const collars = bundle.timeline.holes.map((h) => h.collar)

    const geometry = rockGeometry()
    const mesh = new InstancedMesh(geometry, new MeshStandardMaterial({ roughness: 0.95, metalness: 0, flatShading: true }), Math.max(1, count))
    mesh.count = count
    mesh.frustumCulled = false
    mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3)
    mesh.instanceMatrix.array.fill(0)

    const parent = new Int32Array(count)
    const soil = new Uint8Array(count)
    const size = new Float32Array(count)
    const d0 = new Float32Array(count * 3)
    const dv = new Float32Array(count * 3)
    const scale = new Float32Array(count * 3) // shape of the block along its own axes
    const axis = new Float32Array(count * 3) // spin axis (unit)
    const spin = new Float32Array(count) // rad / s
    const turn = new Float32Array(count) // start angle
    const tone = new Float32Array(count * 3) // site colour, linear
    const shade = new Float32Array(count)
    const rock = ROCK_TONES.map(hexLinear)
    const earth = SOIL_TONES.map(hexLinear)
    parents.forEach((i, a) => {
      const isSoil = particles.kind[i] !== 1 && depthBelowCollars(particles.p0[i], collars) < SOIL_THICKNESS_M
      const speed = Math.hypot(particles.v0[i][0], particles.v0[i][1], particles.v0[i][2])
      for (let c = 0; c < cluster; c++) {
        const k = a * cluster + c
        const seed = i * 16 + c
        parent[k] = i
        soil[k] = isSoil ? 1 : 0
        // The first fragment of a cluster is the parcel itself: its size and its path.
        const own = c === 0
        size[k] = isSoil
          ? SOIL_SIZE_M[0] + (SOIL_SIZE_M[1] - SOIL_SIZE_M[0]) * hash01(seed, 2) ** 2
          : own
            ? particles.size_m[i]
            : Math.min(1.6, Math.max(0.02, fragmentSize(bundle.fragmentation, hash01(seed, 2))))
        const reach = own ? 0 : isSoil ? 1.4 : 0.9
        const drift = own ? 0 : speed * (isSoil ? 0.22 : 0.1)
        for (let j = 0; j < 3; j++) {
          d0[k * 3 + j] = (hash01(seed, 3 + j) - 0.5) * 2 * reach
          dv[k * 3 + j] = (hash01(seed, 6 + j) - 0.5) * 2 * drift
        }
        if (isSoil && !own) dv[k * 3 + 2] += 0.15 * speed * hash01(seed, 9) // loose soil is lifted more
        // Blocks are longer than wide; soil clods are rounder.
        const flat = isSoil ? 0.25 : 0.55
        scale[k * 3] = 1 + flat * (hash01(seed, 10) - 0.3)
        scale[k * 3 + 1] = 1 - flat * 0.6 * hash01(seed, 11)
        scale[k * 3 + 2] = 1 - flat * hash01(seed, 12)
        const u = hash01(seed, 13) * 2 - 1
        const phi = hash01(seed, 14) * Math.PI * 2
        const r = Math.sqrt(1 - u * u)
        axis[k * 3] = r * Math.cos(phi)
        axis[k * 3 + 1] = r * Math.sin(phi)
        axis[k * 3 + 2] = u
        // Small pieces spin faster (assumed: 2 to 14 rad/s).
        spin[k] = (2 + 12 * hash01(seed, 15)) / (0.6 + 2.5 * size[k])
        turn[k] = hash01(seed, 16) * Math.PI * 2
        const tones = isSoil ? earth : rock
        const pick = tones[Math.floor(hash01(seed, 17) * tones.length) % tones.length]
        tone[k * 3] = pick[0]
        tone[k * 3 + 1] = pick[1]
        tone[k * 3 + 2] = pick[2]
        shade[k] = 0.82 + 0.36 * hash01(seed, 18)
      }
    })
    return { geometry, mesh, count, parent, soil, size, d0, dv, scale, axis, spin, turn, tone, shade }
  }, [particles, bundle])
  useEffect(
    () => () => {
      built.geometry.dispose()
      ;(built.mesh.material as MeshStandardMaterial).dispose()
    },
    [built],
  )

  const scratch = useMemo(() => ({ p: new Float64Array(3), o: new Float64Array(3) }), [])
  useFrame(({ camera }) => {
    const { t, fragmentScale } = useBlast.getState()
    const g = bundle.throw.g
    const matrix = built.mesh.instanceMatrix.array as Float32Array
    const color = (built.mesh.instanceColor as InstancedBufferAttribute).array as Float32Array
    const rockTheme = hexLinear(BLAST_COLORS[theme].fragment)
    const warm = hexLinear(BLAST_COLORS[theme].flying)
    const fly = hexLinear(BLAST_COLORS[theme].clearance)
    const cx = camera.position.x
    const cy = camera.position.y
    const cz = camera.position.z
    const { p, o: off } = scratch
    const { parent, soil, size, d0, dv, scale, axis, spin, turn, tone, shade } = built
    let last = -1
    let phase: Phase = Phase.InPlace
    for (let k = 0; k < built.count; k++) {
      const i = parent[k]
      if (i !== last) {
        phase = particlePosition(particles, i, t, g, p)
        last = i
      }
      const m = k * 16
      if (phase === Phase.InPlace) {
        // Still rock in the bench: nothing to draw.
        for (let j = 0; j < 11; j++) matrix[m + j] = 0
        matrix[m + 15] = 1
        continue
      }
      const flight = (particles.t_land_ms[i] - particles.t0_ms[i]) / 1000
      const dt = Math.min((t - particles.t0_ms[i]) / 1000, flight)
      fragmentOffset([d0[k * 3], d0[k * 3 + 1], d0[k * 3 + 2]], [dv[k * 3], dv[k * 3 + 1], dv[k * 3 + 2]], dt, flight, off)
      const x = p[0] + off[0] - origin[0]
      const y = p[1] + off[1] - origin[1]
      const flyrock = particles.kind[i] === 1
      const distance = Math.hypot(x - cx, y - cy, p[2] - origin[2] - cz)
      const half =
        fragmentDrawSize(size[k], fragmentScale * (flyrock ? FLYROCK_BOOST : 1), distance, site ? SITE_FLOOR : undefined) * (flyrock && !site ? 1.6 : 1)
      // A landed fragment lies on the surface, not half inside it.
      const landed = phase === Phase.Flying ? 0 : 1
      const z = p[2] + off[2] - origin[2] + landed * half * 0.35 * scale[k * 3 + 2]

      // Rotation about the spin axis by the angle reached (Rodrigues), stopped at landing.
      const angle = turn[k] + spin[k] * dt
      const c = Math.cos(angle)
      const s = Math.sin(angle)
      const q = 1 - c
      const ax = axis[k * 3]
      const ay = axis[k * 3 + 1]
      const az = axis[k * 3 + 2]
      const sx = scale[k * 3] * half
      const sy = scale[k * 3 + 1] * half
      const sz = scale[k * 3 + 2] * half
      matrix[m] = (c + ax * ax * q) * sx
      matrix[m + 1] = (ay * ax * q + az * s) * sx
      matrix[m + 2] = (az * ax * q - ay * s) * sx
      matrix[m + 4] = (ax * ay * q - az * s) * sy
      matrix[m + 5] = (c + ay * ay * q) * sy
      matrix[m + 6] = (az * ay * q + ax * s) * sy
      matrix[m + 8] = (ax * az * q + ay * s) * sz
      matrix[m + 9] = (ay * az * q - ax * s) * sz
      matrix[m + 10] = (c + az * az * q) * sz
      matrix[m + 12] = x
      matrix[m + 13] = y
      matrix[m + 14] = z
      matrix[m + 15] = 1

      const f = shade[k]
      if (site) {
        color[k * 3] = tone[k * 3] * f
        color[k * 3 + 1] = tone[k * 3 + 1] * f
        color[k * 3 + 2] = tone[k * 3 + 2] * f
      } else {
        // Diagram colours: warm in the air, cooling to rock-grey while it settles.
        const cool = flyrock ? 0 : phase === Phase.Flying ? 0 : smoothstep((t - particles.t_land_ms[i]) / SETTLE_MS)
        const hot = flyrock ? fly : warm
        const cold = soil[k] ? [tone[k * 3], tone[k * 3 + 1], tone[k * 3 + 2]] : rockTheme
        color[k * 3] = (hot[0] + (cold[0] - hot[0]) * cool) * f
        color[k * 3 + 1] = (hot[1] + (cold[1] - hot[1]) * cool) * f
        color[k * 3 + 2] = (hot[2] + (cold[2] - hot[2]) * cool) * f
      }
    }
    built.mesh.instanceMatrix.needsUpdate = true
    ;(built.mesh.instanceColor as InstancedBufferAttribute).needsUpdate = true
  })

  return <primitive object={built.mesh} />
}

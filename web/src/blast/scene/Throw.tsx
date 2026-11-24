/**
 * Thrown fragments as instanced tetrahedra: in the bench (hidden) until launch, ballistic
 * until t_land_ms, then eased to p_rest (math.particlePosition). A fragment is drawn at
 * size_m times the user's "fragment size" factor, but never smaller than a few pixels
 * (math.fragmentDrawSize), so a real-size pattern still shows where the rock goes. Warm
 * while flying, rock-grey once landed; flyrock (kind 1) stays in the warning colour.
 */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { Color, InstancedBufferAttribute, InstancedMesh, Matrix4, MeshStandardMaterial, Quaternion, TetrahedronGeometry, Vector3 } from 'three'
import { useViewer } from '../../state/store'
import { fragmentDrawSize, particlePosition, Phase, SETTLE_MS, smoothstep } from '../math'
import { useBlast } from '../state'
import { BLAST_COLORS, hexLinear, type LayerProps } from './common'

/** More particles than this are thinned evenly (the bundle may carry many). */
export const PARTICLE_CAP = 40_000
/** Flyrock is drawn this many times larger than muck of the same size. */
const FLYROCK_BOOST = 2.5

export function Throw({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const particles = bundle.throw.particles

  const built = useMemo(() => {
    const total = particles.t0_ms.length
    const step = Math.max(1, Math.ceil(total / PARTICLE_CAP))
    const drawn: number[] = []
    for (let i = 0; i < total; i++) if (i % step === 0 || particles.kind[i] === 1) drawn.push(i)
    const geometry = new TetrahedronGeometry(1)
    const mesh = new InstancedMesh(geometry, new MeshStandardMaterial({ roughness: 0.9, metalness: 0, flatShading: true }), Math.max(1, drawn.length))
    mesh.count = drawn.length
    mesh.frustumCulled = false
    mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(Math.max(1, drawn.length) * 3), 3)
    // Per particle: a fixed orientation (the 3x3 block of its instance matrix) and a shade.
    const basis = new Float32Array(drawn.length * 9)
    const shade = new Float32Array(drawn.length)
    const q = new Quaternion()
    const axis = new Vector3()
    const m = new Matrix4()
    drawn.forEach((i, k) => {
      const a = Math.sin(i * 12.9898) * 43758.5453
      const b = Math.sin(i * 78.233) * 12543.123
      axis.set(Math.sin(a), Math.cos(a), Math.sin(b)).normalize()
      q.setFromAxisAngle(axis, b)
      m.makeRotationFromQuaternion(q)
      for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) basis[k * 9 + c * 3 + r] = m.elements[c * 4 + r]
      shade[k] = 0.8 + 0.4 * Math.abs(Math.sin(i * 3.7))
    })
    mesh.instanceMatrix.array.fill(0)
    return { drawn, geometry, mesh, basis, shade }
  }, [particles])
  useEffect(
    () => () => {
      built.geometry.dispose()
      ;(built.mesh.material as MeshStandardMaterial).dispose()
    },
    [built],
  )

  const scratch = useMemo(() => ({ p: new Float64Array(3), c: new Color() }), [])
  useFrame(({ camera }) => {
    const { t, fragmentScale } = useBlast.getState()
    const g = bundle.throw.g
    const matrix = built.mesh.instanceMatrix.array as Float32Array
    const color = (built.mesh.instanceColor as InstancedBufferAttribute).array as Float32Array
    const rock = hexLinear(BLAST_COLORS[theme].fragment)
    const warm = hexLinear(BLAST_COLORS[theme].flying)
    const fly = hexLinear(BLAST_COLORS[theme].clearance)
    const cx = camera.position.x
    const cy = camera.position.y
    const cz = camera.position.z
    const { drawn, basis, shade } = built
    const p = scratch.p
    for (let k = 0; k < drawn.length; k++) {
      const i = drawn[k]
      const phase = particlePosition(particles, i, t, g, p)
      const o = k * 16
      if (phase === Phase.InPlace) {
        // Still rock in the bench: nothing to draw.
        for (let j = 0; j < 11; j++) matrix[o + j] = 0
        matrix[o + 15] = 1
        continue
      }
      const x = p[0] - origin[0]
      const y = p[1] - origin[1]
      const z = p[2] - origin[2]
      const flyrock = particles.kind[i] === 1
      const distance = Math.hypot(x - cx, y - cy, z - cz)
      const s = fragmentDrawSize(particles.size_m[i], fragmentScale * (flyrock ? FLYROCK_BOOST : 1), distance) * (flyrock ? 1.6 : 1)
      const b = k * 9
      matrix[o] = basis[b] * s
      matrix[o + 1] = basis[b + 1] * s
      matrix[o + 2] = basis[b + 2] * s
      matrix[o + 4] = basis[b + 3] * s
      matrix[o + 5] = basis[b + 4] * s
      matrix[o + 6] = basis[b + 5] * s
      matrix[o + 8] = basis[b + 6] * s
      matrix[o + 9] = basis[b + 7] * s
      matrix[o + 10] = basis[b + 8] * s
      matrix[o + 12] = x
      matrix[o + 13] = y
      matrix[o + 14] = z
      matrix[o + 15] = 1
      // Warm in the air, cooling to rock-grey while it settles.
      const cool = flyrock ? 0 : phase === Phase.Flying ? 0 : smoothstep((t - particles.t_land_ms[i]) / SETTLE_MS)
      const hot = flyrock ? fly : warm
      const f = shade[k]
      color[k * 3] = (hot[0] + (rock[0] - hot[0]) * cool) * f
      color[k * 3 + 1] = (hot[1] + (rock[1] - hot[1]) * cool) * f
      color[k * 3 + 2] = (hot[2] + (rock[2] - hot[2]) * cool) * f
    }
    built.mesh.instanceMatrix.needsUpdate = true
    ;(built.mesh.instanceColor as InstancedBufferAttribute).needsUpdate = true
  })

  return <primitive object={built.mesh} />
}

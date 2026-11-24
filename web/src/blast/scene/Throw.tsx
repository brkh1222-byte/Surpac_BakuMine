/**
 * Thrown fragments as instanced tetrahedra: in the bench (hidden) until launch, ballistic
 * until t_land_ms, then eased to p_rest (math.particlePosition). Sizes come from size_m
 * with a floor so that fines stay visible; flyrock (kind 1) is drawn larger and in the
 * warning colour.
 */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { Color, InstancedMesh, Matrix4, MeshStandardMaterial, Quaternion, TetrahedronGeometry, Vector3 } from 'three'
import { useViewer } from '../../state/store'
import { particlePosition, Phase } from '../math'
import { useBlast } from '../state'
import { BLAST_COLORS, type LayerProps } from './common'

/** More particles than this are thinned evenly (the bundle may carry many). */
export const PARTICLE_CAP = 40_000
/** Smallest drawn half-size, m. */
const FLOOR = 0.14
const FLYROCK_FLOOR = 0.45

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
    // Per particle: a fixed orientation times its size, as the 3x3 block of the instance matrix.
    const basis = new Float32Array(drawn.length * 9)
    const q = new Quaternion()
    const axis = new Vector3()
    const m = new Matrix4()
    drawn.forEach((i, k) => {
      const a = Math.sin(i * 12.9898) * 43758.5453
      const b = Math.sin(i * 78.233) * 12543.123
      axis.set(Math.sin(a), Math.cos(a), Math.sin(b)).normalize()
      q.setFromAxisAngle(axis, b)
      const fly = particles.kind[i] === 1
      const size = Math.max(fly ? FLYROCK_FLOOR : FLOOR, particles.size_m[i] / 2)
      m.makeRotationFromQuaternion(q)
      for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) basis[k * 9 + c * 3 + r] = m.elements[c * 4 + r] * size
    })
    mesh.instanceMatrix.array.fill(0)
    return { drawn, geometry, mesh, basis }
  }, [particles])
  useEffect(
    () => () => {
      built.geometry.dispose()
      ;(built.mesh.material as MeshStandardMaterial).dispose()
    },
    [built],
  )

  useEffect(() => {
    const c = new Color()
    const rock = new Color(BLAST_COLORS[theme].fragment)
    const fly = new Color(BLAST_COLORS[theme].clearance)
    built.drawn.forEach((i, k) => {
      if (particles.kind[i] === 1) c.copy(fly)
      else c.copy(rock).multiplyScalar(0.75 + 0.5 * Math.abs(Math.sin(i * 3.7)))
      built.mesh.setColorAt(k, c)
    })
    if (built.mesh.instanceColor) built.mesh.instanceColor.needsUpdate = true
  }, [built, particles, theme])

  const p = useMemo(() => new Float64Array(3), [])
  useFrame(() => {
    const t = useBlast.getState().t
    const g = bundle.throw.g
    const array = built.mesh.instanceMatrix.array as Float32Array
    const { drawn, basis } = built
    for (let k = 0; k < drawn.length; k++) {
      const phase = particlePosition(particles, drawn[k], t, g, p)
      const o = k * 16
      if (phase === Phase.InPlace) {
        // Still rock in the bench: nothing to draw.
        for (let j = 0; j < 11; j++) array[o + j] = 0
        array[o + 15] = 1
        continue
      }
      const b = k * 9
      array[o] = basis[b]
      array[o + 1] = basis[b + 1]
      array[o + 2] = basis[b + 2]
      array[o + 4] = basis[b + 3]
      array[o + 5] = basis[b + 4]
      array[o + 6] = basis[b + 5]
      array[o + 8] = basis[b + 6]
      array[o + 9] = basis[b + 7]
      array[o + 10] = basis[b + 8]
      array[o + 12] = p[0] - origin[0]
      array[o + 13] = p[1] - origin[1]
      array[o + 14] = p[2] - origin[2]
      array[o + 15] = 1
    }
    built.mesh.instanceMatrix.needsUpdate = true
  })

  return <primitive object={built.mesh} />
}

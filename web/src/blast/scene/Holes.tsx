/**
 * Blast holes: stemming and charge column as instanced cylinders, coloured by firing time
 * before the shot; a hole flashes when it fires (pulse decaying over ~50 ms) and is then
 * drawn spent. Delay labels sit on the collars.
 */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { AdditiveBlending, Color, CylinderGeometry, InstancedMesh, Matrix4, MeshBasicMaterial, MeshStandardMaterial, Quaternion, SphereGeometry, Vector3 } from 'three'
import { useViewer } from '../../state/store'
import { firePulse, timingColor } from '../math'
import { useBlast } from '../state'
import { BLAST_COLORS, hexLinear, linear, type LabelItem, type LayerProps } from './common'
import { Labels } from './Labels'

const UP = new Vector3(0, 1, 0)

/** Radius the holes are drawn with: true, or wide enough to see from across the bench. */
export function drawnRadius(diameter: number, exaggerate: boolean): number {
  return exaggerate ? Math.max(0.3, 2.5 * diameter) : diameter / 2
}

export function Holes({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const exaggerate = useBlast((s) => s.exaggerateHoles)
  const showLabels = useBlast((s) => s.layers.labels)
  const holes = bundle.timeline.holes
  const duration = useMemo(() => Math.max(1, ...holes.map((h) => h.t_fire_ms)), [holes])

  const built = useMemo(() => {
    const n = holes.length
    const cylinder = new CylinderGeometry(1, 1, 1, 12)
    const sphere = new SphereGeometry(1, 12, 8)
    const stemming = new InstancedMesh(cylinder, new MeshStandardMaterial({ roughness: 0.9, metalness: 0 }), n)
    const charge = new InstancedMesh(cylinder, new MeshStandardMaterial({ roughness: 0.7, metalness: 0 }), n)
    const flash = new InstancedMesh(sphere, new MeshBasicMaterial({ transparent: true, blending: AdditiveBlending, depthWrite: false, depthTest: false }), n)
    for (const mesh of [stemming, charge, flash]) mesh.frustumCulled = false
    flash.renderOrder = 6
    return { cylinder, sphere, stemming, charge, flash }
  }, [holes])
  useEffect(
    () => () => {
      built.cylinder.dispose()
      built.sphere.dispose()
      for (const mesh of [built.stemming, built.charge, built.flash]) (mesh.material as MeshBasicMaterial).dispose()
    },
    [built],
  )

  // Static part: the cylinders.
  useEffect(() => {
    const m = new Matrix4()
    const q = new Quaternion()
    const a = new Vector3()
    const b = new Vector3()
    const mid = new Vector3()
    const dir = new Vector3()
    const scale = new Vector3()
    const stem = new Color(BLAST_COLORS[theme].stemming)
    const place = (mesh: InstancedMesh, i: number, p: readonly number[], r: readonly number[], radius: number) => {
      a.set(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
      b.set(r[0] - origin[0], r[1] - origin[1], r[2] - origin[2])
      const length = a.distanceTo(b)
      dir.subVectors(b, a)
      if (length > 1e-9) dir.divideScalar(length)
      else dir.set(0, 0, -1)
      q.setFromUnitVectors(UP, dir)
      mid.addVectors(a, b).multiplyScalar(0.5)
      mesh.setMatrixAt(i, m.compose(mid, q, scale.set(radius, Math.max(length, 1e-6), radius)))
    }
    holes.forEach((h, i) => {
      const radius = drawnRadius(h.diameter_m, exaggerate)
      place(built.stemming, i, h.collar, h.charge_top, radius * 0.8)
      place(built.charge, i, h.charge_top, h.charge_bottom, radius)
      built.stemming.setColorAt(i, stem)
    })
    built.stemming.instanceMatrix.needsUpdate = true
    built.charge.instanceMatrix.needsUpdate = true
    if (built.stemming.instanceColor) built.stemming.instanceColor.needsUpdate = true
  }, [built, holes, origin, exaggerate, theme])

  // Per frame: colour and flash.
  const frame = useMemo(() => ({ m: new Matrix4(), c: new Color(), q: new Quaternion(), p: new Vector3(), s: new Vector3() }), [])
  useFrame(() => {
    const t = useBlast.getState().t
    const spent = hexLinear(BLAST_COLORS[theme].spent)
    for (let i = 0; i < holes.length; i++) {
      const h = holes[i]
      const fired = t >= h.t_fire_ms
      const pulse = firePulse(t, h.t_fire_ms)
      const base = fired ? spent : linear(timingColor(h.t_fire_ms, duration))
      frame.c.setRGB(base[0] + (1 - base[0]) * pulse, base[1] + (0.9 - base[1]) * pulse, base[2] + (0.6 - base[2]) * pulse)
      built.charge.setColorAt(i, frame.c)
      const radius = drawnRadius(h.diameter_m, true)
      const size = pulse > 0.02 ? radius * (1.5 + 5 * pulse) : 0
      frame.p.set(h.collar[0] - origin[0], h.collar[1] - origin[1], h.collar[2] - origin[2])
      built.flash.setMatrixAt(i, frame.m.compose(frame.p, frame.q, frame.s.setScalar(size)))
      built.flash.setColorAt(i, frame.c.setRGB(pulse, 0.75 * pulse, 0.3 * pulse))
    }
    if (built.charge.instanceColor) built.charge.instanceColor.needsUpdate = true
    if (built.flash.instanceColor) built.flash.instanceColor.needsUpdate = true
    built.flash.instanceMatrix.needsUpdate = true
  })

  const labels = useMemo<LabelItem[]>(
    () =>
      holes.map((h) => ({
        position: [h.collar[0] - origin[0], h.collar[1] - origin[1], h.collar[2] - origin[2] + 0.4],
        text: `${Math.round(h.t_fire_ms)}`,
        className: 'blast-label-delay',
      })),
    [holes, origin],
  )

  return (
    <>
      <primitive object={built.stemming} />
      <primitive object={built.charge} />
      <primitive object={built.flash} />
      {showLabels && <Labels items={labels} />}
    </>
  )
}

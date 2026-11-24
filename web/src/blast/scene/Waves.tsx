/**
 * Stress wavefronts: per source a translucent sphere of radius cp * (t - t_ms), cut off at
 * the collar elevation (the wave runs in the rock), plus the ring where it meets the
 * surface. Colour and opacity follow PPV(R) on the log scale of the legend; a front is
 * dropped below the cut-off, and at most `WAVE_CAP` (the newest) are drawn.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BackSide, BufferGeometry, Float32BufferAttribute, Group, LineBasicMaterial, LineLoop, Mesh, MeshBasicMaterial, Plane, SphereGeometry, Vector3 } from 'three'
import { activeWaves, logScale, ppvColor, PPV_RANGE } from '../math'
import { useBlast } from '../state'
import { linear, type LayerProps } from './common'

export const WAVE_CAP = 16
/** Fronts weaker than this are not drawn, mm/s. */
export const WAVE_CUTOFF_MM_S = 10

export function Waves({ bundle, origin }: LayerProps) {
  const gl = useThree((s) => s.gl)
  useEffect(() => {
    const before = gl.localClippingEnabled
    gl.localClippingEnabled = true
    return () => {
      gl.localClippingEnabled = before
    }
  }, [gl])

  const built = useMemo(() => {
    const group = new Group()
    const sphere = new SphereGeometry(1, 40, 20)
    const circle = new BufferGeometry()
    const pts: number[] = []
    for (let i = 0; i < 96; i++) pts.push(Math.cos((i / 96) * 2 * Math.PI), Math.sin((i / 96) * 2 * Math.PI), 0)
    circle.setAttribute('position', new Float32BufferAttribute(pts, 3))
    const slots = Array.from({ length: WAVE_CAP }, () => {
      const plane = new Plane(new Vector3(0, 0, -1), 0)
      const shell = new Mesh(sphere, new MeshBasicMaterial({ transparent: true, depthWrite: false, depthTest: false, side: BackSide, clippingPlanes: [plane] }))
      const ring = new LineLoop(circle, new LineBasicMaterial({ transparent: true, depthTest: false }))
      shell.renderOrder = 4
      ring.renderOrder = 5
      shell.frustumCulled = false
      ring.frustumCulled = false
      group.add(shell, ring)
      return { plane, shell, ring }
    })
    const collarZ = new Map(bundle.timeline.holes.map((h) => [h.id, h.collar[2]]))
    return { group, sphere, circle, slots, collarZ }
  }, [bundle])
  useEffect(
    () => () => {
      built.sphere.dispose()
      built.circle.dispose()
      for (const s of built.slots) {
        s.shell.material.dispose()
        s.ring.material.dispose()
      }
    },
    [built],
  )

  useFrame(() => {
    const t = useBlast.getState().t
    const { sources } = bundle.waves
    const active = activeWaves(sources, t, bundle.waves, WAVE_CUTOFF_MM_S, WAVE_CAP)
    for (let i = 0; i < built.slots.length; i++) {
      const slot = built.slots[i]
      const wave = active[i]
      if (!wave) {
        slot.shell.visible = false
        slot.ring.visible = false
        continue
      }
      const s = sources[wave.index]
      const cx = s.centre[0] - origin[0]
      const cy = s.centre[1] - origin[1]
      const cz = s.centre[2] - origin[2]
      const top = (built.collarZ.get(s.hole) ?? s.centre[2]) - origin[2]
      const strength = logScale(wave.ppv, WAVE_CUTOFF_MM_S, PPV_RANGE.max)
      const color = linear(ppvColor(wave.ppv))
      slot.shell.visible = true
      slot.shell.position.set(cx, cy, cz)
      slot.shell.scale.setScalar(wave.radius)
      slot.shell.material.color.setRGB(color[0], color[1], color[2])
      slot.shell.material.opacity = 0.03 + 0.22 * strength
      slot.plane.constant = top
      const dz = top - cz
      const r2 = wave.radius * wave.radius - dz * dz
      slot.ring.visible = r2 > 0
      if (r2 > 0) {
        const r = Math.sqrt(r2)
        slot.ring.position.set(cx, cy, top + 0.15)
        slot.ring.scale.set(r, r, 1)
        slot.ring.material.color.setRGB(color[0], color[1], color[2])
        slot.ring.material.opacity = 0.25 + 0.75 * strength
      }
    }
  })

  return <primitive object={built.group} />
}

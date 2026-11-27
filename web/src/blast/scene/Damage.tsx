/**
 * Damage blocks as instanced boxes. A blasted block is drawn as rock until its t_break,
 * then it is gone (it has become fragments); a block left in the wall switches on at its
 * t_break in the colour of its damage class. Intact blocks are never drawn.
 */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BoxGeometry, Color, InstancedMesh, Matrix4, MeshStandardMaterial } from 'three'
import { useViewer } from '../../state/store'
import { countAtOrBefore, DAMAGE_COLORS, sortedFinite } from '../math'
import { useBlast } from '../state'
import { BLAST_COLORS, type LayerProps } from './common'

/** Boxes are drawn slightly small, so that the block structure reads. */
const SHRINK = 0.9

export function Damage({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const blocks = bundle.damage.blocks

  const built = useMemo(() => {
    const drawn: number[] = []
    for (let i = 0; i < blocks.cls.length; i++) if (blocks.blasted[i] || blocks.cls[i] > 0) drawn.push(i)
    const box = new BoxGeometry(1, 1, 1)
    const mesh = new InstancedMesh(box, new MeshStandardMaterial({ roughness: 0.95, metalness: 0 }), Math.max(1, drawn.length))
    mesh.count = drawn.length
    mesh.frustumCulled = false
    return { drawn, box, mesh, breaks: sortedFinite(blocks.t_break_ms), state: { key: -1 } }
  }, [blocks])
  useEffect(
    () => () => {
      built.box.dispose()
      ;(built.mesh.material as MeshStandardMaterial).dispose()
    },
    [built],
  )

  useEffect(() => {
    const c = new Color()
    built.drawn.forEach((i, k) => {
      c.set(blocks.blasted[i] ? BLAST_COLORS[theme].rock : DAMAGE_COLORS[Math.min(DAMAGE_COLORS.length - 1, Math.max(0, blocks.cls[i]))])
      built.mesh.setColorAt(k, c)
    })
    if (built.mesh.instanceColor) built.mesh.instanceColor.needsUpdate = true
  }, [built, blocks, theme])

  const m = useMemo(() => new Matrix4(), [])
  useFrame(() => {
    const t = useBlast.getState().t
    const key = countAtOrBefore(built.breaks, t)
    if (key === built.state.key) return
    built.state.key = key
    built.drawn.forEach((i, k) => {
      // null: no break time given. A damaged block then shows from the start, a blasted one goes at t = 0.
      const broken = t >= (blocks.t_break_ms[i] ?? 0)
      const visible = blocks.blasted[i] ? !broken : broken
      const c = blocks.centre[i]
      const s = blocks.size[i]
      const f = visible ? SHRINK : 0
      m.makeScale(s[0] * f, s[1] * f, s[2] * f).setPosition(c[0] - origin[0], c[1] - origin[1], c[2] - origin[2])
      built.mesh.setMatrixAt(k, m)
    })
    built.mesh.instanceMatrix.needsUpdate = true
  })

  return <primitive object={built.mesh} />
}

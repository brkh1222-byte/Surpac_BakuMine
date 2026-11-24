/**
 * Text labels pinned to scene points: plain DOM nodes over the canvas, positioned every
 * rendered frame (the same approach as the modeller's grid and hole labels).
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import { Vector3 } from 'three'
import { blastView, type LabelItem } from './common'

const v = new Vector3()

export function Labels({ items }: { items: readonly LabelItem[] }) {
  const nodes = useRef<HTMLSpanElement[]>([])
  const size = useThree((s) => s.size)
  const invalidate = useThree((s) => s.invalidate)

  useEffect(() => {
    const host = blastView.labels
    if (!host) return
    const made = items.map((item) => {
      const el = document.createElement('span')
      el.className = 'blast-label' + (item.className ? ' ' + item.className : '')
      el.textContent = item.text
      el.style.display = 'none'
      host.appendChild(el)
      return el
    })
    nodes.current = made
    invalidate()
    return () => {
      for (const el of made) el.remove()
      nodes.current = []
    }
  }, [items, invalidate])

  useFrame(({ camera }) => {
    const list = nodes.current
    for (let i = 0; i < list.length && i < items.length; i++) {
      const p = items[i].position
      v.set(p[0], p[1], p[2]).project(camera)
      const el = list[i]
      if (v.z < -1 || v.z > 1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05) {
        el.style.display = 'none'
        continue
      }
      el.style.display = ''
      el.style.transform = `translate(${((v.x * 0.5 + 0.5) * size.width).toFixed(1)}px, ${((-v.y * 0.5 + 0.5) * size.height).toFixed(1)}px) translate(-50%, -100%)`
    }
  })
  return null
}

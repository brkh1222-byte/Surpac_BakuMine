/**
 * Section digitising in the canvas: draws the digitised strings, their vertices and the
 * snap marker, and turns pointer input on the section plane into edits while the tool is
 * on (state/digitise.ts, edits in digitise/edit.ts, snapping in digitise/snap.ts).
 *
 *   click                add a vertex to the string being drawn (or start one)
 *   click first vertex   close the string (three or more vertices)
 *   drag a vertex        move it
 *   shift + click edge   insert a vertex on that edge
 *   right-click vertex   delete it
 *   Enter / Esc / N      close / finish / new string;  Delete removes the selected vertex
 *   Ctrl+Z               undo the last edit
 *
 * Vertices go on the section plane (ray x plane), snapped to the nearest drillhole
 * crossing or interval boundary within SNAP_PIXELS. Strings are kept in world
 * coordinates; they are drawn in render space here, outside the clipped content group,
 * without a depth test, so they always show over the section.
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BufferGeometry, CanvasTexture, Float32BufferAttribute, Group, Points, PointsMaterial, Raycaster, Vector2 } from 'three'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { canClose, findString, type DigitString } from '../digitise/edit'
import { hitEdge, hitVertex, inSlabString, nearestSnap, onPlane, sectionId, snapTargets, type SnapTarget } from '../digitise/snap'
import { rayPlane, type SectionPlane } from '../section/plane'
import { useDigitise } from '../state/digitise'
import { useSection } from '../state/section'
import { useViewer, type Theme } from '../state/store'
import { getOrigin, renderToWorld, useSceneOrigin, worldToRender, type Vec3 } from './origin'
import { view } from './view'

/** Pointer tolerances, CSS pixels. */
export const SNAP_PIXELS = 10
const VERTEX_PIXELS = 8
const EDGE_PIXELS = 6
const CLICK_PIXELS = 4

export const DIGITISE_COLORS: Record<Theme, { open: string; active: string; closed: string; snap: string; vertex: string }> = {
  dark: { open: '#9aa7b4', active: '#ffd166', closed: '#4ade80', snap: '#22d3ee', vertex: '#ffffff' },
  light: { open: '#5b6672', active: '#b7791f', closed: '#15803d', snap: '#0e7490', vertex: '#111111' },
}

function discTexture(ring: boolean): CanvasTexture {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const c = canvas.getContext('2d')
  if (c) {
    c.beginPath()
    c.arc(size / 2, size / 2, size / 2 - 6, 0, Math.PI * 2)
    if (ring) {
      c.lineWidth = 8
      c.strokeStyle = '#ffffff'
      c.stroke()
      c.beginPath()
      c.arc(size / 2, size / 2, 5, 0, Math.PI * 2)
      c.fillStyle = '#ffffff'
      c.fill()
    } else {
      c.fillStyle = '#ffffff'
      c.fill()
      c.lineWidth = 6
      c.strokeStyle = 'rgba(0, 0, 0, 0.6)'
      c.stroke()
    }
  }
  return new CanvasTexture(canvas)
}

function points(size: number, texture: CanvasTexture, order: number, vertexColors: boolean): Points<BufferGeometry, PointsMaterial> {
  const p = new Points(
    new BufferGeometry(),
    new PointsMaterial({ size, sizeAttenuation: false, map: texture, transparent: true, alphaTest: 0.3, depthTest: false, depthWrite: false, vertexColors }),
  )
  p.renderOrder = order
  p.frustumCulled = false
  p.userData.noRaycast = true
  return p
}

/** Snap targets of every drawn drillhole layer, cached per plane and slab. */
let targetCache: { key: string; targets: SnapTarget[] } | null = null
export function currentSnapTargets(plane: SectionPlane, front: number, back: number): SnapTarget[] {
  const key = `${plane.centre.join(',')}|${plane.azimuth}|${plane.dip}|${front}|${back}|${[...view.holePieces.values()].map((p) => p.count).join(',')}`
  if (targetCache?.key === key) return targetCache.targets
  const targets: SnapTarget[] = []
  for (const [id, pieces] of view.holePieces) targets.push(...snapTargets(plane, { front, back }, pieces, id))
  targetCache = { key, targets }
  return targets
}

/** Strings that can be edited on the current section: those lying on its plane. */
export function editableStrings(strings: readonly DigitString[], plane: SectionPlane): DigitString[] {
  return strings.filter((s) => onPlane(s, plane))
}

function typing(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (!el || !el.tagName) return false
  return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable
}

export function DigitiseLayer() {
  const active = useDigitise((s) => s.active)
  const edit = useDigitise((s) => s.edit)
  const hover = useDigitise((s) => s.hover)
  const snap = useDigitise((s) => s.snap)
  const enabled = useSection((s) => s.enabled)
  const plane = useSection((s) => s.plane)
  const front = useSection((s) => s.front)
  const back = useSection((s) => s.back)
  const theme = useViewer((s) => s.theme)
  const exaggeration = useViewer((s) => s.verticalExaggeration)
  const origin = useSceneOrigin()
  const gl = useThree((s) => s.gl)
  const invalidate = useThree((s) => s.invalidate)

  const parts = useMemo(() => {
    const group = new Group()
    group.name = 'digitise'
    group.userData.noClip = true
    const line = new LineSegments2(
      new LineSegmentsGeometry(),
      new LineMaterial({ linewidth: 2.5, vertexColors: true, depthTest: false, depthWrite: false, transparent: true }),
    )
    line.renderOrder = 40
    line.frustumCulled = false
    line.userData.noRaycast = true
    line.visible = false
    const disc = discTexture(false)
    const ring = discTexture(true)
    const vertices = points(8, disc, 41, true)
    const marked = points(13, disc, 42, true)
    const marker = points(22, ring, 43, false)
    group.add(line, vertices, marked, marker)
    return { group, line, vertices, marked, marker, textures: [disc, ring] }
  }, [])
  useEffect(
    () => () => {
      parts.line.geometry.dispose()
      parts.line.material.dispose()
      for (const p of [parts.vertices, parts.marked, parts.marker]) {
        p.geometry.dispose()
        p.material.dispose()
      }
      for (const t of parts.textures) t.dispose()
    },
    [parts],
  )

  // ---- drawing
  useEffect(() => {
    const colors = DIGITISE_COLORS[theme]
    const shown = enabled ? edit.strings.filter((s) => inSlabString(s, plane, { front, back })) : edit.strings
    const colourOf = (s: DigitString) => (s.closed ? colors.closed : s.id === edit.active ? colors.active : colors.open)
    const rgb = (hex: string) => {
      const n = parseInt(hex.slice(1), 16)
      return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
    }
    const r = (p: Vec3) => worldToRender(p, exaggeration, origin)

    const segments: number[] = []
    const segmentColors: number[] = []
    const vertexPositions: number[] = []
    const vertexColors: number[] = []
    for (const s of shown) {
      const c = rgb(colourOf(s))
      const m = s.points.length
      const edges = s.closed ? m : m - 1
      for (let i = 0; i < edges; i++) {
        segments.push(...r(s.points[i]), ...r(s.points[(i + 1) % m]))
        segmentColors.push(...c, ...c)
      }
      for (const p of s.points) {
        vertexPositions.push(...r(p))
        vertexColors.push(...c)
      }
    }
    parts.line.visible = segments.length > 0
    if (segments.length > 0) {
      const geometry = new LineSegmentsGeometry()
      geometry.setPositions(segments)
      geometry.setColors(segmentColors)
      parts.line.geometry.dispose()
      parts.line.geometry = geometry
    }
    const setPoints = (p: Points<BufferGeometry, PointsMaterial>, positions: number[], cols?: number[]) => {
      const geometry = new BufferGeometry()
      geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
      if (cols) geometry.setAttribute('color', new Float32BufferAttribute(cols, 3))
      p.geometry.dispose()
      p.geometry = geometry
      p.visible = positions.length > 0
    }
    setPoints(parts.vertices, vertexPositions, vertexColors)

    // Larger: the hovered and the selected vertex, and the first vertex of the string
    // being drawn when a click on it would close it.
    const marks: number[] = []
    const markColors: number[] = []
    const mark = (id: string, index: number, hex: string) => {
      const s = findString(edit, id)
      if (!s || !s.points[index] || !shown.includes(s)) return
      marks.push(...r(s.points[index]))
      markColors.push(...rgb(hex))
    }
    if (active && edit.active && canClose(edit)) mark(edit.active, 0, colors.closed)
    if (edit.selected) mark(edit.selected.string, edit.selected.index, colors.vertex)
    if (active && hover) mark(hover.string, hover.index, colors.snap)
    setPoints(parts.marked, marks, markColors)

    parts.marker.material.color.set(colors.snap)
    setPoints(parts.marker, active && snap ? r(snap.point) : [])
    invalidate()
  }, [parts, edit, hover, snap, active, enabled, plane, front, back, theme, exaggeration, origin, invalidate])

  // ---- pointer input
  useEffect(() => {
    if (!active) return
    const dom = gl.domElement
    const host = dom.parentElement ?? dom
    const raycaster = new Raycaster()
    const ndc = new Vector2()
    let down: { x: number; y: number } | null = null
    let drag: { string: string; index: number } | null = null
    let scheduled = 0
    let pointer: { x: number; y: number } | null = null

    /** World point of the section plane under the pointer. */
    const worldAt = (x: number, y: number): Vec3 | null => {
      const camera = view.camera
      const section = useSection.getState()
      if (!camera || !section.enabled) return null
      const rect = dom.getBoundingClientRect()
      ndc.set(((x - rect.left) / rect.width) * 2 - 1, -((y - rect.top) / rect.height) * 2 + 1)
      raycaster.setFromCamera(ndc, camera)
      const e = useViewer.getState().verticalExaggeration
      const o = raycaster.ray.origin
      const d = raycaster.ray.direction
      return rayPlane(section.plane, renderToWorld([o.x, o.y, o.z], e, getOrigin()), [d.x, d.y, d.z / e], true)
    }
    const metres = (pixels: number) => pixels * view.metresPerPixel
    const context = () => {
      const section = useSection.getState()
      const state = useDigitise.getState()
      return {
        section,
        state,
        exaggeration: useViewer.getState().verticalExaggeration,
        editable: editableStrings(state.edit.strings, section.plane),
      }
    }
    const snapped = (p: Vec3): { point: Vec3; target: SnapTarget | null } => {
      const { section, state, exaggeration: e } = context()
      if (!state.snapping) return { point: p, target: null }
      const target = nearestSnap(currentSnapTargets(section.plane, section.front, section.back), p, metres(SNAP_PIXELS), e)
      return { point: target ? target.point : p, target }
    }

    const hoverUpdate = () => {
      scheduled = 0
      if (!pointer || drag) return
      const p = worldAt(pointer.x, pointer.y)
      const { state, editable, exaggeration: e } = context()
      if (!p) return state.setPointer(null, null)
      const hit = hitVertex(editable, p, metres(VERTEX_PIXELS), e)
      state.setPointer(hit, hit ? null : snapped(p).target)
    }

    // The listeners sit on the viewport (capture) to come before the camera controls;
    // panels inside the viewport are not the canvas.
    const onCanvas = (event: Event) => event.target === dom
    const onDown = (event: PointerEvent) => {
      down = null
      if (event.button !== 0 || !onCanvas(event)) return
      const p = worldAt(event.clientX, event.clientY)
      if (!p) return
      const { state, editable, exaggeration: e } = context()
      const hit = hitVertex(editable, p, metres(VERTEX_PIXELS), e)
      down = { x: event.clientX, y: event.clientY }
      if (!hit) return
      // On a vertex: this press is ours (no panning); it becomes a drag once it moves.
      event.stopPropagation()
      event.preventDefault()
      drag = hit
      state.select(hit)
    }
    let dragged = false
    const onMove = (event: PointerEvent) => {
      pointer = { x: event.clientX, y: event.clientY }
      if (drag) {
        event.stopPropagation()
        const moved = down ? Math.hypot(event.clientX - down.x, event.clientY - down.y) : Infinity
        if (!dragged && moved <= CLICK_PIXELS) return
        const p = worldAt(event.clientX, event.clientY)
        if (!p) return
        const state = useDigitise.getState()
        if (!dragged) state.beginDrag()
        dragged = true
        const s = snapped(p)
        state.dragVertex(drag, s.point)
        state.setPointer(drag, s.target)
        return
      }
      if (onCanvas(event) && !scheduled) scheduled = requestAnimationFrame(hoverUpdate)
    }
    const onUp = (event: PointerEvent) => {
      const start = down
      down = null
      if (drag) {
        const ref = drag
        const wasDrag = dragged
        drag = null
        dragged = false
        event.stopPropagation()
        if (!wasDrag) clickVertex(ref)
        return
      }
      if (event.button !== 0 || !start || !onCanvas(event) || Math.hypot(event.clientX - start.x, event.clientY - start.y) > CLICK_PIXELS) return
      const p = worldAt(event.clientX, event.clientY)
      if (!p) return
      const { section, state, editable, exaggeration: e } = context()
      const s = snapped(p)
      if (event.shiftKey) {
        const edge = hitEdge(editable, p, metres(EDGE_PIXELS), e)
        if (edge) state.insertVertex(edge, s.point)
        return
      }
      state.addVertex(sectionId(section.plane), s.point)
    }
    /** A click (no drag) on a vertex: close on the first vertex, else select it. */
    const clickVertex = (ref: { string: string; index: number }) => {
      const state = useDigitise.getState()
      const { edit } = state
      if (ref.string === edit.active && ref.index === 0 && canClose(edit)) {
        state.closeString(ref.string)
        return
      }
      const target = findString(edit, ref.string)
      // Clicking the last vertex of an open string carries on drawing it.
      if (target && !target.closed && ref.index === target.points.length - 1) state.activate(ref.string)
      state.select(ref)
    }
    const onContext = (event: MouseEvent) => {
      if (!onCanvas(event)) return
      const p = worldAt(event.clientX, event.clientY)
      if (!p) return
      const { state, editable, exaggeration: e } = context()
      const hit = hitVertex(editable, p, metres(VERTEX_PIXELS), e)
      if (!hit) return
      event.preventDefault()
      event.stopPropagation()
      state.deleteVertex(hit)
      state.setPointer(null, null)
    }
    const onLeave = () => {
      pointer = null
      if (!drag) useDigitise.getState().setPointer(null, null)
    }
    const onKey = (event: KeyboardEvent) => {
      if (typing(event.target)) return
      const state = useDigitise.getState()
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') state.undo()
      else if (event.ctrlKey || event.metaKey || event.altKey) return
      else if (event.key === 'Enter') state.closeString()
      else if (event.key === 'Escape' || event.key.toLowerCase() === 'n') state.newString()
      else if (event.key === 'Delete' || event.key === 'Backspace') state.deleteVertex()
      else return
      event.preventDefault()
    }

    host.addEventListener('pointerdown', onDown, true)
    host.addEventListener('pointermove', onMove, true)
    host.addEventListener('pointerup', onUp, true)
    host.addEventListener('contextmenu', onContext, true)
    dom.addEventListener('pointerleave', onLeave)
    window.addEventListener('keydown', onKey)
    return () => {
      host.removeEventListener('pointerdown', onDown, true)
      host.removeEventListener('pointermove', onMove, true)
      host.removeEventListener('pointerup', onUp, true)
      host.removeEventListener('contextmenu', onContext, true)
      dom.removeEventListener('pointerleave', onLeave)
      window.removeEventListener('keydown', onKey)
      if (scheduled) cancelAnimationFrame(scheduled)
      useDigitise.getState().setPointer(null, null)
    }
  }, [active, gl])

  // The objects join the scene graph only while there is something to draw.
  return edit.strings.length > 0 || (active && snap) ? <primitive object={parts.group} /> : null
}

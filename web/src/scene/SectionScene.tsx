/**
 * The section tools inside the canvas:
 *
 *   <SectionCamera/>     the section view: orthographic, square-on, rotation locked,
 *                        following the plane as it steps
 *   <SectionIndicator/>  where the section is, in the free 3D view: a translucent
 *                        rectangle on the plane, the slab faces, and the line being drawn
 *   <GhostRenderer/>     ghost mode: the whole scene dimmed, the slab drawn over it
 *   <SectionOutline/>    one mesh cut by the plane, as a crisp line (used by MeshLayer)
 *
 * Clipping itself is in scene/clipping.ts and is applied by <SceneContent>.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  BufferGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  OrthographicCamera,
  PlaneGeometry,
  Scene,
} from 'three'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { outlineSegments, type Outline } from '../section/outline'
import { outlineClient } from '../section/outlineClient'
import { boxInPlane, frameOf, fromPlane, type SectionPlane } from '../section/plane'
import { sectionExtent, useSection, type SectionState } from '../state/section'
import { allBounds, isShown, useViewer, type Bounds, type Theme } from '../state/store'
import { applySectionClipping, updateSectionPlanes } from './clipping'
import { useSceneOrigin, worldToRender, type Vec3 } from './origin'
import { SCENE_COLORS } from './theme'
import { view, type SectionViewSpec } from './view'

/** Shown when nothing is loaded: a 1 km square around the plane centre. */
function fallbackBounds(centre: Vec3): Bounds {
  return { min: [centre[0] - 500, centre[1] - 500, centre[2] - 500], max: [centre[0] + 500, centre[1] + 500, centre[2] + 500] }
}

/** What the section view frames: the visible objects that reach into the slab. */
export function sectionViewSpec(section: Pick<SectionState, 'plane' | 'front' | 'back'>, exaggeration: number): SectionViewSpec {
  const { plane } = section
  const frame = frameOf(plane)
  const state = useViewer.getState()
  const visible = Object.values(state.objects).filter((o) => isShown(state, o))
  const s: [number, number] = [Infinity, -Infinity]
  const t: [number, number] = [Infinity, -Infinity]
  const add = (bounds: Bounds, onlyInSlab: boolean) => {
    const box = boxInPlane(plane, bounds, frame)
    if (onlyInSlab && (box.min[2] > section.front || box.max[2] < -section.back)) return
    s[0] = Math.min(s[0], box.min[0])
    s[1] = Math.max(s[1], box.max[0])
    t[0] = Math.min(t[0], box.min[1])
    t[1] = Math.max(t[1], box.max[1])
  }
  for (const object of visible) add(object.bounds, true)
  if (!(s[1] > s[0])) add(allBounds(state) ?? fallbackBounds(plane.centre), false)

  const { v, n } = frame
  const centre = worldToRender(fromPlane(plane, (s[0] + s[1]) / 2, (t[0] + t[1]) / 2, 0, frame), exaggeration)
  // Render space: z is stretched, so in-plane "up" is longer and the normal tilts.
  const vScale = Math.hypot(v[0], v[1], v[2] * exaggeration)
  const direction: [number, number, number] = [n[0], n[1], n[2] / exaggeration]
  if (Math.hypot(direction[0], direction[1]) < 1e-4 * Math.abs(direction[2])) {
    // Looking straight down: lean a hair so that the plane's "up" (v) is up on screen
    // and the camera's vertical up vector stays usable.
    const h = Math.hypot(v[0], v[1]) || 1
    direction[0] -= (1e-5 * v[0] * Math.abs(direction[2])) / h
    direction[1] -= (1e-5 * v[1] * Math.abs(direction[2])) / h
  }
  return {
    centre,
    direction,
    halfWidth: Math.max((s[1] - s[0]) / 2, 1),
    halfHeight: Math.max(((t[1] - t[0]) / 2) * vScale, 1),
  }
}

/** Keeps the camera square-on to the section while `locked`; frees it otherwise. */
export function SectionCamera() {
  const locked = useSection((s) => s.locked)
  const viewRequest = useSection((s) => s.viewRequest)
  const plane = useSection((s) => s.plane)
  const projection = useViewer((s) => s.projection)
  const exaggeration = useViewer((s) => s.verticalExaggeration)
  const last = useRef<{ plane: SectionPlane; request: number; exaggeration: number } | null>(null)

  useEffect(() => {
    view.rig?.setRotationLocked(locked)
    if (!locked) last.current = null
    return () => view.rig?.setRotationLocked(false)
  }, [locked])

  useEffect(() => {
    const rig = view.rig
    if (!locked || projection !== 'orthographic' || !rig) return
    const previous = last.current
    if (
      previous &&
      previous.request === viewRequest &&
      previous.exaggeration === exaggeration &&
      previous.plane.azimuth === plane.azimuth &&
      previous.plane.dip === plane.dip
    ) {
      // Same orientation, the plane moved: slide along the normal, keep pan and zoom.
      const { n } = frameOf(plane)
      const d =
        (plane.centre[0] - previous.plane.centre[0]) * n[0] +
        (plane.centre[1] - previous.plane.centre[1]) * n[1] +
        (plane.centre[2] - previous.plane.centre[2]) * n[2]
      if (d !== 0) rig.shiftBy([n[0] * d, n[1] * d, n[2] * d * exaggeration])
    } else if (!rig.sectionView(sectionViewSpec(useSection.getState(), exaggeration))) {
      return
    }
    last.current = { plane, request: viewRequest, exaggeration }
  }, [locked, viewRequest, plane, projection, exaggeration])

  return null
}

const ACCENT: Record<Theme, string> = { dark: '#e2a03a', light: '#a45f00' }

function rectangle(plane: SectionPlane, s: [number, number], t: [number, number], w: number, exaggeration: number, origin: Vec3): number[][] {
  return [
    [s[0], t[0]],
    [s[1], t[0]],
    [s[1], t[1]],
    [s[0], t[1]],
  ].map(([a, b]) => worldToRender(fromPlane(plane, a, b, w), exaggeration, origin))
}

function loopSegments(corners: number[][]): number[] {
  const out: number[] = []
  for (let i = 0; i < corners.length; i++) out.push(...corners[i], ...corners[(i + 1) % corners.length])
  return out
}

/** The section in the free 3D view, and the line being drawn in plan. Render space. */
export function SectionIndicator() {
  const enabled = useSection((s) => s.enabled)
  const locked = useSection((s) => s.locked)
  const plane = useSection((s) => s.plane)
  const front = useSection((s) => s.front)
  const back = useSection((s) => s.back)
  const draft = useSection((s) => s.draft)
  const theme = useViewer((s) => s.theme)
  const exaggeration = useViewer((s) => s.verticalExaggeration)
  const objects = useViewer((s) => s.objects)
  const origin = useSceneOrigin()
  const invalidate = useThree((s) => s.invalidate)

  const parts = useMemo(() => {
    const group = new Group()
    group.name = 'section-indicator'
    const fill = new Mesh(
      new BufferGeometry(),
      new MeshBasicMaterial({ transparent: true, opacity: 0.1, side: DoubleSide, depthWrite: false }),
    )
    const border = new LineSegments(new BufferGeometry(), new LineBasicMaterial({ transparent: true, opacity: 0.95 }))
    const faces = new LineSegments(new BufferGeometry(), new LineBasicMaterial({ transparent: true, opacity: 0.4 }))
    const line = new LineSegments(new BufferGeometry(), new LineBasicMaterial({ depthTest: false, transparent: true }))
    fill.renderOrder = 8
    border.renderOrder = 9
    faces.renderOrder = 9
    line.renderOrder = 30
    for (const object of [fill, border, faces, line]) {
      object.frustumCulled = false
      group.add(object)
    }
    return { group, fill, border, faces, line }
  }, [])
  useEffect(
    () => () => {
      for (const object of [parts.fill, parts.border, parts.faces, parts.line]) {
        object.geometry.dispose()
        object.material.dispose()
      }
    },
    [parts],
  )

  useEffect(() => {
    const colour = ACCENT[theme]
    for (const object of [parts.fill, parts.border, parts.faces, parts.line]) object.material.color.set(colour)

    const show = enabled && !locked
    parts.fill.visible = parts.border.visible = parts.faces.visible = show
    if (show) {
      const state = useViewer.getState()
      const extent = sectionExtent(plane, allBounds(state) ?? fallbackBounds(plane.centre))
      const mid = rectangle(plane, extent.s, extent.t, 0, exaggeration, origin)
      const fill = new BufferGeometry()
      fill.setAttribute('position', new Float32BufferAttribute(mid.flat(), 3))
      fill.setIndex([0, 1, 2, 0, 2, 3])
      parts.fill.geometry.dispose()
      parts.fill.geometry = fill
      const border = new BufferGeometry()
      border.setAttribute('position', new Float32BufferAttribute(loopSegments(mid), 3))
      parts.border.geometry.dispose()
      parts.border.geometry = border
      const faces = new BufferGeometry()
      faces.setAttribute(
        'position',
        new Float32BufferAttribute(
          [
            ...loopSegments(rectangle(plane, extent.s, extent.t, front, exaggeration, origin)),
            ...loopSegments(rectangle(plane, extent.s, extent.t, -back, exaggeration, origin)),
          ],
          3,
        ),
      )
      parts.faces.geometry.dispose()
      parts.faces.geometry = faces
    }

    const drawn = draft?.a && draft.b ? [draft.a, draft.b] : null
    parts.line.visible = drawn !== null
    if (drawn) {
      const geometry = new BufferGeometry()
      geometry.setAttribute(
        'position',
        new Float32BufferAttribute(drawn.flatMap((p) => worldToRender(p, exaggeration, origin)), 3),
      )
      parts.line.geometry.dispose()
      parts.line.geometry = geometry
    }
    invalidate()
  }, [parts, enabled, locked, plane, front, back, draft, theme, exaggeration, objects, origin, invalidate])

  return <primitive object={parts.group} />
}

/**
 * Ghost mode: pass 1 draws the whole scene unclipped, a veil in the canvas colour dims
 * it, and pass 2 draws the slab alone on top (depth cleared, so the slab is never hidden
 * by the ghost in front of it). Mounted only while ghost mode is on: a `useFrame` with a
 * priority takes the render loop over from react-three-fiber.
 */
export function GhostRenderer() {
  const theme = useViewer((s) => s.theme)
  const veil = useMemo(() => {
    const scene = new Scene()
    const material = new MeshBasicMaterial({ transparent: true, opacity: 0.78, depthTest: false, depthWrite: false })
    const quad = new Mesh(new PlaneGeometry(2, 2), material)
    quad.frustumCulled = false
    scene.add(quad)
    return { scene, quad, material, camera: new OrthographicCamera(-1, 1, 1, -1, 0, 1) }
  }, [])
  useEffect(() => {
    veil.material.color.set(SCENE_COLORS[theme].background)
  }, [veil, theme])
  useEffect(
    () => () => {
      veil.quad.geometry.dispose()
      veil.material.dispose()
    },
    [veil],
  )

  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  const draw = useMemo(
    () => () => {
      const camera = view.camera
      const content = view.content
      if (!camera) return
      if (!content || !updateSectionPlanes()) {
        gl.render(scene, camera)
        return
      }
      applySectionClipping(content, false)
      gl.render(scene, camera)

      const autoClear = gl.autoClear
      const background = scene.background
      gl.autoClear = false
      gl.render(veil.scene, veil.camera)
      gl.clearDepth()
      // Pass 2: the content only, clipped to the slab.
      const hidden = scene.children.filter((child) => child !== content && child.visible && !('isLight' in child))
      for (const child of hidden) child.visible = false
      scene.background = null
      applySectionClipping(content, true)
      gl.render(scene, camera)
      applySectionClipping(content, false)
      scene.background = background
      for (const child of hidden) child.visible = true
      gl.autoClear = autoClear
    },
    [gl, scene, veil],
  )
  useEffect(() => {
    view.renderFrame = draw
    return () => {
      if (view.renderFrame === draw) view.renderFrame = null
    }
  }, [draw])
  useFrame(draw, 1)
  return null
}

const OUTLINE_WIDTH = 2.25
const OUTLINE_HALO_WIDTH = 5

export interface SectionOutlineProps {
  id: string
  vertices: Float64Array
  faces: Uint32Array
  color: string
  origin: Vec3
}

/**
 * The exact intersection of one mesh with the section plane: a fat line in the object
 * colour over a thin halo in the canvas colour. It lies ON the mesh surface, so it is
 * drawn without a depth test (it would z-fight with the clipped surface) and after
 * everything else; it is exempt from clipping (`noClip`): it is the cut itself.
 * Mount it inside the mesh's group, in scene space.
 */
export function SectionOutline({ id, vertices, faces, color, origin }: SectionOutlineProps) {
  const enabled = useSection((s) => s.enabled)
  const plane = useSection((s) => s.plane)
  const theme = useViewer((s) => s.theme)
  const invalidate = useThree((s) => s.invalidate)
  const [outline, setOutline] = useState<Outline | null>(null)
  // The line objects join the scene graph only while there is something to draw.
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    outlineClient.register(id, vertices, faces)
    return () => outlineClient.unregister(id)
  }, [id, vertices, faces])

  useEffect(() => {
    if (!enabled) {
      setOutline(null)
      return
    }
    let cancelled = false
    void outlineClient.cut(id, plane.centre, frameOf(plane).n).then((result) => {
      // null: a newer plane replaced this request; its result is on the way.
      if (!cancelled && result) setOutline(result)
    })
    return () => {
      cancelled = true
    }
  }, [enabled, plane, id, vertices, faces])

  const lines = useMemo(() => {
    const make = (width: number, order: number) => {
      const material = new LineMaterial({ linewidth: width, depthTest: false, depthWrite: false, transparent: true })
      const line = new LineSegments2(new LineSegmentsGeometry(), material)
      line.renderOrder = order
      line.frustumCulled = false
      line.visible = false
      line.userData.noRaycast = true
      return line
    }
    const group = new Group()
    group.name = `section-outline:${id}`
    group.userData.noClip = true
    const halo = make(OUTLINE_HALO_WIDTH, 20)
    halo.name = `section-outline-halo:${id}`
    const line = make(OUTLINE_WIDTH, 21)
    line.name = `section-outline-line:${id}`
    group.add(halo, line)
    return { group, halo, line }
  }, [id])
  useEffect(
    () => () => {
      for (const line of [lines.halo, lines.line]) {
        line.geometry.dispose()
        line.material.dispose()
      }
    },
    [lines],
  )

  useEffect(() => {
    const segments = outline ? outlineSegments(outline, origin) : null
    const has = segments !== null && segments.length > 0
    if (has) {
      // The instance count of a fat-line geometry is fixed by its first buffer: new geometry.
      const geometry = new LineSegmentsGeometry()
      geometry.setPositions(segments)
      lines.halo.geometry.dispose()
      lines.halo.geometry = lines.line.geometry = geometry
    }
    lines.halo.visible = lines.line.visible = has
    setMounted(has)
    if (outline) view.section.outlines.set(id, outline)
    else view.section.outlines.delete(id)
    invalidate()
  }, [lines, outline, origin, id, invalidate])
  useEffect(() => () => void view.section.outlines.delete(id), [id])

  useEffect(() => {
    lines.line.material.color.set(color)
    lines.halo.material.color.set(SCENE_COLORS[theme].background)
    invalidate()
  }, [lines, color, theme, invalidate])

  return mounted ? <primitive object={lines.group} /> : null
}

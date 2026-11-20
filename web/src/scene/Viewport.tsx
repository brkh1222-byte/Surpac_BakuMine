import { Canvas, useThree } from '@react-three/fiber'
import { useEffect } from 'react'
import { Object3D, PerspectiveCamera, Plane, Raycaster, Vector2, Vector3 } from 'three'
import { rayPlane, toPlane } from '../section/plane'
import { useBlocks } from '../state/blocks'
import { useDisplay } from '../state/display'
import { sectionExtent, useSection } from '../state/section'
import { allBounds, useViewer } from '../state/store'
import { BlockPanel } from '../ui/BlockPanel'
import { HolePanel } from '../ui/HolePanel'
import { HoverTooltip } from '../ui/HoverTooltip'
import { LegendStack } from '../ui/Legend'
import { CameraRig } from './CameraRig'
import { insideSectionSlab, sectionClipActive, updateSectionPlanes } from './clipping'
import { installDebugHandle } from './debug'
import { SceneContent } from './layers'
import { MockLayer } from './MockLayer'
import { getOrigin, renderToWorld } from './origin'
import { PerfScene } from './PerfScene'
import { ProjectScene } from './ProjectScene'
import { gridElevation, ReferenceGrid } from './ReferenceGrid'
import { GhostRenderer, SectionCamera, SectionIndicator } from './SectionScene'
import { SCENE_COLORS } from './theme'
import { overlay, view, type PickResult } from './view'

// World convention: x = east, y = north, z = up.
Object3D.DEFAULT_UP.set(0, 0, 1)

/**
 * Visible leaf objects under `root` for three's raycaster (which does not skip hidden
 * ones). Objects flagged `userData.noRaycast` are left out: their layer picks them
 * itself (view.pickers) or they are too heavy to raycast on every pointer move.
 */
function pickable(root: Object3D, out: Object3D[] = []): Object3D[] {
  if (!root.visible) return out
  if ((root as Object3D & { geometry?: unknown }).geometry && !root.userData.noRaycast) out.push(root)
  for (const child of root.children) pickable(child, out)
  return out
}

/**
 * Reports the true world coordinate under the cursor: picked object, else grid plane.
 * In the section view it is always the point of the section plane under the cursor
 * (ray x plane), which is what a section is read at. While a section line is being
 * drawn, clicks place its two ends instead of selecting.
 */
function CursorProbe() {
  const gl = useThree((s) => s.gl)

  useEffect(() => {
    const dom = gl.domElement
    const raycaster = new Raycaster()
    const ndc = new Vector2()
    const plane = new Plane(new Vector3(0, 0, 1), 0)
    const hit = new Vector3()
    let pointer: { x: number; y: number } | null = null
    let scheduled = 0
    let down: { x: number; y: number } | null = null

    /** Casts the pointer ray: nearest raycast hit and nearest layer-picker result. */
    const cast = (x: number, y: number) => {
      const camera = view.camera
      if (!camera) return null
      const rect = dom.getBoundingClientRect()
      ndc.set(((x - rect.left) / rect.width) * 2 - 1, -((y - rect.top) / rect.height) * 2 + 1)
      raycaster.setFromCamera(ndc, camera)
      raycaster.params.Line.threshold = view.metresPerPixel * 4
      let hits = view.content ? raycaster.intersectObjects(pickable(view.content), false) : []
      // What the section clips away is not there to be hit.
      if (sectionClipActive() && updateSectionPlanes()) {
        hits = hits.filter((h) => insideSectionSlab(h.point.x, h.point.y, h.point.z, view.metresPerPixel * 0.5))
      }

      const perspective = camera instanceof PerspectiveCamera
      const ray = {
        origin: raycaster.ray.origin,
        direction: raycaster.ray.direction,
        metresPerPixel0: perspective ? 0 : view.metresPerPixel,
        metresPerPixelSlope: perspective ? (2 * Math.tan((camera.fov * Math.PI) / 360)) / rect.height : 0,
      }
      const results = [...view.pickers].map((picker) => picker(ray)).filter((r): r is PickResult => r !== null)
      const nearest = results.reduce<PickResult | null>((best, r) => (!best || r.distance < best.distance ? r : best), null)
      // An opaque surface in front hides what is behind it; a see-through one does not.
      const blocker = hits.find((h) => !h.object.userData.seeThrough)
      const picked = nearest && blocker && blocker.distance < nearest.distance ? null : nearest
      return { hits, picked }
    }

    /** The last cast ray meets the grid plane here (world), or null. */
    const ground = (): [number, number, number] | null => {
      const origin = getOrigin()
      const { verticalExaggeration } = useViewer.getState()
      const elevation = gridElevation()
      plane.constant = -(elevation - origin[2]) * verticalExaggeration
      if (!raycaster.ray.intersectPlane(plane, hit)) return null
      const [e, n] = renderToWorld([hit.x, hit.y, hit.z], verticalExaggeration, origin)
      return [e, n, elevation]
    }

    /** The last cast ray meets the section plane here (world, exactly on the plane), or null. */
    const onSection = (anywhere: boolean): [number, number, number] | null => {
      const section = useSection.getState()
      if (!section.enabled) return null
      const { verticalExaggeration } = useViewer.getState()
      const o = raycaster.ray.origin
      const d = raycaster.ray.direction
      const p = rayPlane(
        section.plane,
        renderToWorld([o.x, o.y, o.z], verticalExaggeration, getOrigin()),
        [d.x, d.y, d.z / verticalExaggeration],
        anywhere,
      )
      if (!p || anywhere) return p
      // Free view: only where the section is drawn (the rectangle across the data).
      const bounds = allBounds(useViewer.getState())
      if (!bounds) return null
      const extent = sectionExtent(section.plane, bounds)
      const [s, t] = toPlane(section.plane, p)
      return s >= extent.s[0] && s <= extent.s[1] && t >= extent.t[0] && t <= extent.t[1] ? p : null
    }

    const probe = () => {
      scheduled = 0
      const { setCursor, verticalExaggeration } = useViewer.getState()
      if (!pointer) return
      const result = cast(pointer.x, pointer.y)
      if (!result) return
      const origin = getOrigin()
      const section = useSection.getState()
      if (section.draft) {
        // Drawing the section line: the pointer is a point of the plan.
        useDisplay.getState().setHover(null)
        const p = ground()
        setCursor(p ? { e: p[0], n: p[1], z: p[2], on: 'grid' } : null)
        if (p && section.draft.a) section.setDraft({ a: section.draft.a, b: p })
        return
      }
      const { hits, picked } = result
      if (picked) picked.hover(pointer.x, pointer.y)
      else useDisplay.getState().setHover(null)
      const square = section.locked ? onSection(true) : null
      if (square) {
        setCursor({ e: square[0], n: square[1], z: square[2], on: 'section' })
        return
      }
      if (picked) {
        const [e, n, z] = renderToWorld(picked.point, verticalExaggeration, origin)
        setCursor({ e, n, z, on: 'object' })
        return
      }
      if (hits.length > 0) {
        const p = hits[0].point
        const [e, n, z] = renderToWorld([p.x, p.y, p.z], verticalExaggeration, origin)
        setCursor({ e, n, z, on: 'object' })
        return
      }
      const cut = onSection(false)
      if (cut) {
        setCursor({ e: cut[0], n: cut[1], z: cut[2], on: 'section' })
        return
      }
      const p = ground()
      setCursor(p ? { e: p[0], n: p[1], z: p[2], on: 'grid' } : null)
    }

    const onMove = (event: PointerEvent) => {
      pointer = { x: event.clientX, y: event.clientY }
      if (!scheduled) scheduled = requestAnimationFrame(probe)
    }
    const onLeave = () => {
      pointer = null
      useViewer.getState().setCursor(null)
      useDisplay.getState().setHover(null)
    }
    // A click (press and release without dragging) selects what is under the pointer.
    const onDown = (event: PointerEvent) => {
      down = event.button === 0 ? { x: event.clientX, y: event.clientY } : null
    }
    const onUp = (event: PointerEvent) => {
      const start = down
      down = null
      if (!start || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return
      const section = useSection.getState()
      if (section.draft) {
        // First click: one end of the section line; second click: the other end.
        const p = cast(event.clientX, event.clientY) ? ground() : null
        if (!p) return
        if (!section.draft.a) section.setDraft({ a: p, b: p })
        else if (!section.fromTwoPoints(section.draft.a, p)) section.cancelDrawing()
        return
      }
      const picked = cast(event.clientX, event.clientY)?.picked
      // One thing is selected at a time: a hole or a block.
      useBlocks.getState().setPick(null)
      if (picked) picked.select()
      else useDisplay.getState().setSelection(null)
    }
    dom.addEventListener('pointermove', onMove)
    dom.addEventListener('pointerleave', onLeave)
    dom.addEventListener('pointerdown', onDown)
    dom.addEventListener('pointerup', onUp)
    return () => {
      dom.removeEventListener('pointermove', onMove)
      dom.removeEventListener('pointerleave', onLeave)
      dom.removeEventListener('pointerdown', onDown)
      dom.removeEventListener('pointerup', onUp)
      if (scheduled) cancelAnimationFrame(scheduled)
    }
  }, [gl])

  return null
}

function DebugHandle() {
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  useEffect(() => installDebugHandle(gl, scene), [gl, scene])
  return null
}

function OrientationGizmo() {
  return (
    <svg
      className="gizmo"
      viewBox="-44 -44 88 88"
      role="img"
      aria-label="Orientation: E east, N north, Z up"
      ref={(el) => {
        overlay.gizmo = el
      }}
    >
      <defs>
        <marker id="north-head" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0 0 L8 4 L0 8 Z" className="gizmo-north-head" />
        </marker>
      </defs>
      <circle r="41" className="gizmo-ring" />
      <line data-axis="e" x1="0" y1="0" x2="26" y2="0" className="gizmo-axis" />
      <line data-axis="z" x1="0" y1="0" x2="0" y2="-26" className="gizmo-axis" />
      <line data-axis="n" x1="0" y1="0" x2="0" y2="-26" className="gizmo-axis gizmo-north" markerEnd="url(#north-head)" />
      <circle r="1.8" className="gizmo-hub" />
      <text data-axis-label="e" className="gizmo-label">E</text>
      <text data-axis-label="z" className="gizmo-label">Z</text>
      <text data-axis-label="n" className="gizmo-label gizmo-north-label">N</text>
    </svg>
  )
}

/**
 * `api`: the project served by the API; `mock`: client-side stand-ins; `perf`: large
 * client-generated objects for the performance probe; `none`: empty.
 */
export type SceneMode = 'api' | 'mock' | 'perf' | 'none'

export function Viewport({ mode }: { mode: SceneMode }) {
  const theme = useViewer((s) => s.theme)
  const colors = SCENE_COLORS[theme]
  const debug = mode === 'mock' || mode === 'perf' || import.meta.env.DEV
  const ghost = useSection((s) => s.enabled && s.mode === 'ghost')
  const drawing = useSection((s) => s.draft !== null)

  return (
    <div className={'viewport' + (drawing ? ' is-drawing' : '')} data-testid="viewport">
      {/* `flat`: no tone mapping, so data colours on screen match their legend exactly. */}
      <Canvas flat frameloop="demand" dpr={[1, 2]} gl={{ antialias: true }} data-testid="scene-canvas">
        <color attach="background" args={[colors.background]} />
        {/* Key light from the west and above, a weak fill from the opposite side, and
            enough ambient that faces turned away keep their colour: surfaces show their
            form from every standard view, including the iso view the scene opens in. */}
        <ambientLight intensity={1.0} />
        <directionalLight position={[-1, -0.15, 0.75]} intensity={2.3} />
        <directionalLight position={[0.7, 0.9, 0.25]} intensity={0.55} />
        <CameraRig />
        <ReferenceGrid />
        <SceneContent>
          {mode === 'mock' && <MockLayer />}
          {mode === 'perf' && <PerfScene />}
          {mode === 'api' && <ProjectScene />}
        </SceneContent>
        <SectionIndicator />
        <SectionCamera />
        {ghost && <GhostRenderer />}
        <CursorProbe />
        {debug && <DebugHandle />}
      </Canvas>
      <div
        className="grid-labels"
        aria-hidden="true"
        ref={(el) => {
          overlay.labels = el
        }}
      />
      <div
        className="grid-labels hole-labels"
        aria-hidden="true"
        data-testid="hole-labels"
        ref={(el) => {
          overlay.holeLabels = el
        }}
      />
      <OrientationGizmo />
      <LegendStack />
      <HolePanel />
      <BlockPanel />
      <HoverTooltip />
      {drawing && (
        <div className="viewport-hint" data-testid="section-hint" role="status">
          Click the two ends of the section line · Esc cancels
        </div>
      )}
      <div className="scale" title="Scale at the view centre">
        <div
          className="scale-bar"
          ref={(el) => {
            overlay.scaleBar = el
          }}
        />
        <span
          data-testid="scale-text"
          ref={(el) => {
            overlay.scaleText = el
          }}
        />
      </div>
    </div>
  )
}

import { Canvas, useThree } from '@react-three/fiber'
import { useEffect } from 'react'
import { Object3D, Plane, Raycaster, Vector2, Vector3 } from 'three'
import { useViewer } from '../state/store'
import { CameraRig } from './CameraRig'
import { installDebugHandle } from './debug'
import { SceneContent } from './layers'
import { MockLayer } from './MockLayer'
import { getOrigin, renderToWorld } from './origin'
import { ProjectScene } from './ProjectScene'
import { gridElevation, ReferenceGrid } from './ReferenceGrid'
import { SCENE_COLORS } from './theme'
import { overlay, view } from './view'

// World convention: x = east, y = north, z = up.
Object3D.DEFAULT_UP.set(0, 0, 1)

/** Visible leaf objects under `root` (three's raycaster does not skip hidden ones). */
function pickable(root: Object3D, out: Object3D[] = []): Object3D[] {
  if (!root.visible) return out
  if ((root as Object3D & { geometry?: unknown }).geometry) out.push(root)
  for (const child of root.children) pickable(child, out)
  return out
}

/** Reports the true world coordinate under the cursor: picked object, else grid plane. */
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

    const probe = () => {
      scheduled = 0
      const camera = view.camera
      const { setCursor, verticalExaggeration } = useViewer.getState()
      if (!camera || !pointer) return
      const rect = dom.getBoundingClientRect()
      ndc.set(
        ((pointer.x - rect.left) / rect.width) * 2 - 1,
        -((pointer.y - rect.top) / rect.height) * 2 + 1,
      )
      raycaster.setFromCamera(ndc, camera)
      raycaster.params.Line.threshold = view.metresPerPixel * 4
      const origin = getOrigin()

      const hits = view.content ? raycaster.intersectObjects(pickable(view.content), false) : []
      if (hits.length > 0) {
        const p = hits[0].point
        const [e, n, z] = renderToWorld([p.x, p.y, p.z], verticalExaggeration, origin)
        setCursor({ e, n, z, on: 'object' })
        return
      }
      const elevation = gridElevation()
      plane.constant = -(elevation - origin[2]) * verticalExaggeration
      if (raycaster.ray.intersectPlane(plane, hit)) {
        const [e, n] = renderToWorld([hit.x, hit.y, hit.z], verticalExaggeration, origin)
        setCursor({ e, n, z: elevation, on: 'grid' })
      } else {
        setCursor(null)
      }
    }

    const onMove = (event: PointerEvent) => {
      pointer = { x: event.clientX, y: event.clientY }
      if (!scheduled) scheduled = requestAnimationFrame(probe)
    }
    const onLeave = () => {
      pointer = null
      useViewer.getState().setCursor(null)
    }
    dom.addEventListener('pointermove', onMove)
    dom.addEventListener('pointerleave', onLeave)
    return () => {
      dom.removeEventListener('pointermove', onMove)
      dom.removeEventListener('pointerleave', onLeave)
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

/** `api`: the project served by the API; `mock`: client-side stand-ins; `none`: empty. */
export type SceneMode = 'api' | 'mock' | 'none'

export function Viewport({ mode }: { mode: SceneMode }) {
  const theme = useViewer((s) => s.theme)
  const colors = SCENE_COLORS[theme]
  const debug = mode === 'mock' || import.meta.env.DEV

  return (
    <div className="viewport" data-testid="viewport">
      {/* `flat`: no tone mapping, so data colours on screen match their legend exactly. */}
      <Canvas flat frameloop="demand" dpr={[1, 2]} gl={{ antialias: true }} data-testid="scene-canvas">
        <color attach="background" args={[colors.background]} />
        <ambientLight intensity={1.5} />
        <directionalLight position={[-0.5, -0.8, 1]} intensity={1.9} />
        <CameraRig />
        <ReferenceGrid />
        <SceneContent>
          {mode === 'mock' && <MockLayer />}
          {mode === 'api' && <ProjectScene />}
        </SceneContent>
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
      <OrientationGizmo />
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

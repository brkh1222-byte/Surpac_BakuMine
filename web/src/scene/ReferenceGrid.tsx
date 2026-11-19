/**
 * Reference geometry: ground grid, elevation pole, coordinate labels, orientation gizmo
 * and scale bar. Spacing follows the zoom (1-2-5 steps) and every label is a real world
 * coordinate. Grid buffers are local to the grid's own centre, so they stay small.
 */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import {
  BufferGeometry,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Quaternion,
  Vector3,
  type Camera,
} from 'three'
import { allBounds, useViewer } from '../state/store'
import { decimalsFor, formatCoord, formatLength, niceStep } from './format'
import { getOrigin, renderToWorld } from './origin'
import { SCENE_COLORS } from './theme'
import { overlay, view } from './view'

/** Target on-screen distance between labelled grid lines, CSS pixels. */
const LABEL_SPACING_PX = 110
const MINOR_PER_MAJOR = 5
const Z_TICK_PX = 56

/** World elevation of the grid plane: just under the loaded data, on a round number. */
export function gridElevation(): number {
  const bounds = allBounds(useViewer.getState())
  if (!bounds) return getOrigin()[2]
  return Math.floor(bounds.min[2] / 10) * 10
}

function setLines(target: LineSegments, positions: number[]): void {
  target.geometry.dispose()
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  target.geometry = geometry
}

interface Placed {
  x: number
  y: number
  w: number
}

/** Pool of absolutely positioned label elements, rewritten every frame. */
class LabelPool {
  private used = 0
  private placed: Placed[] = []
  private readonly v = new Vector3()

  begin(): void {
    this.used = 0
    this.placed.length = 0
  }

  /** Screen position of a render-space point, or null when it is not in view. */
  private project(camera: Camera, width: number, height: number, x: number, y: number, z: number, text: string) {
    const v = this.v.set(x, y, z).project(camera)
    if (!(v.z > -1 && v.z < 1)) return null
    const sx = (v.x * 0.5 + 0.5) * width
    const sy = (-v.y * 0.5 + 0.5) * height
    const w = text.length * 6.4 + 10
    if (sx < w / 2 + 2 || sx > width - w / 2 - 2 || sy < 10 || sy > height - 10) return null
    return { sx, sy, w }
  }

  /** Places a label centred on a screen point unless it would overlap an earlier one. */
  private place(sx: number, sy: number, w: number, text: string, kind: string): void {
    const host = overlay.labels
    if (!host) return
    for (const p of this.placed) {
      if (Math.abs(p.x - sx) < (p.w + w) / 2 && Math.abs(p.y - sy) < 16) return
    }
    this.placed.push({ x: sx, y: sy, w })

    let el = host.children[this.used] as HTMLSpanElement | undefined
    if (!el) {
      el = document.createElement('span')
      host.appendChild(el)
    }
    this.used++
    if (el.textContent !== text) el.textContent = text
    if (el.dataset.kind !== kind) el.dataset.kind = kind
    el.style.display = ''
    el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) translate(-50%, -50%)`
  }

  /** Label at a render-space point, shifted `shiftPx` to the right of it (0 = centred). */
  add(camera: Camera, width: number, height: number, x: number, y: number, z: number, text: string, kind: string, shiftPx = 0): void {
    const p = this.project(camera, width, height, x, y, z, text)
    if (p) this.place(p.sx + (shiftPx ? shiftPx + p.w / 2 : 0), p.sy, p.w, text, kind)
  }

  /**
   * Label for a grid line: at `a` (just outside the grid edge) when that is on screen,
   * otherwise at the first point of the line a -> b that is, so that lines running off
   * the viewport are still labelled where they enter it.
   */
  addAlong(camera: Camera, width: number, height: number, a: Vector3, b: Vector3, text: string, kind: string): void {
    const STEPS = 40
    for (let i = 0; i <= STEPS; i++) {
      const t = i / STEPS
      const p = this.project(camera, width, height, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, text)
      if (p) {
        this.place(p.sx, p.sy, p.w, text, kind)
        return
      }
    }
  }

  end(): void {
    const host = overlay.labels
    if (!host) return
    for (let i = this.used; i < host.children.length; i++) {
      ;(host.children[i] as HTMLElement).style.display = 'none'
    }
  }
}

const AXES: { id: string; dir: [number, number, number] }[] = [
  { id: 'e', dir: [1, 0, 0] },
  { id: 'n', dir: [0, 1, 0] },
  { id: 'z', dir: [0, 0, 1] },
]
const GIZMO_RADIUS = 26
const from = new Vector3()
const to = new Vector3()
const inverse = new Quaternion()
const axis = new Vector3()

/** Draws the three world axes as seen by the camera into the SVG gizmo. */
function updateGizmo(camera: Camera): void {
  const svg = overlay.gizmo
  if (!svg) return
  inverse.copy(camera.quaternion).invert()
  for (const { id, dir } of AXES) {
    axis.set(...dir).applyQuaternion(inverse)
    const x = axis.x * GIZMO_RADIUS
    const y = -axis.y * GIZMO_RADIUS
    const line = svg.querySelector<SVGLineElement>(`[data-axis="${id}"]`)
    const label = svg.querySelector<SVGTextElement>(`[data-axis-label="${id}"]`)
    const length = Math.hypot(x, y)
    if (line) {
      line.setAttribute('x2', x.toFixed(2))
      line.setAttribute('y2', y.toFixed(2))
      // An axis pointing along the view direction has no direction on screen.
      line.style.opacity = length < 3 ? '0' : axis.z < -0.05 ? '0.55' : '1'
    }
    if (label) {
      const k = length < 3 ? 0 : (length + 9) / length
      label.setAttribute('x', (x * k).toFixed(2))
      label.setAttribute('y', (y * k).toFixed(2))
      label.style.opacity = length < 3 ? '0' : axis.z < -0.05 ? '0.6' : '1'
    }
  }
}

function updateScaleBar(metresPerPixel: number): void {
  if (!overlay.scaleBar || !overlay.scaleText) return
  const length = niceStep(metresPerPixel * 70)
  overlay.scaleBar.style.width = `${(length / metresPerPixel).toFixed(1)}px`
  const text = formatLength(length)
  if (overlay.scaleText.textContent !== text) overlay.scaleText.textContent = text
}

export function ReferenceGrid() {
  const theme = useViewer((s) => s.theme)
  const colors = SCENE_COLORS[theme]

  const parts = useMemo(() => {
    const material = (opacity: number) =>
      new LineBasicMaterial({ transparent: opacity < 1, opacity, depthWrite: false })
    const minor = new LineSegments(new BufferGeometry(), material(1))
    const major = new LineSegments(new BufferGeometry(), material(1))
    const border = new LineSegments(new BufferGeometry(), material(1))
    const pole = new LineSegments(new BufferGeometry(), material(1))
    // Draw order: fine lines first so the stronger ones stay on top.
    minor.renderOrder = -4
    major.renderOrder = -3
    border.renderOrder = -2
    pole.renderOrder = -2
    for (const lines of [minor, major, border, pole]) lines.frustumCulled = false
    const grid = new Group()
    grid.name = 'reference-grid'
    grid.add(minor, major, border)
    const root = new Group()
    root.name = 'reference'
    root.add(grid, pole)
    return { root, grid, minor, major, border, pole }
  }, [])

  useEffect(() => {
    ;(parts.minor.material as LineBasicMaterial).color.set(colors.gridMinor)
    ;(parts.major.material as LineBasicMaterial).color.set(colors.gridMajor)
    ;(parts.border.material as LineBasicMaterial).color.set(colors.gridBorder)
    ;(parts.pole.material as LineBasicMaterial).color.set(colors.gridBorder)
  }, [parts, colors])

  useEffect(
    () => () => {
      for (const lines of [parts.minor, parts.major, parts.border, parts.pole]) {
        lines.geometry.dispose()
        ;(lines.material as LineBasicMaterial).dispose()
      }
    },
    [parts],
  )

  const keys = useRef({ grid: '', pole: '' })
  const pool = useMemo(() => new LabelPool(), [])

  useFrame(({ size }) => {
    const camera = view.camera
    if (!camera) return
    const state = useViewer.getState()
    const exaggeration = state.verticalExaggeration
    const origin = getOrigin()
    const mpp = view.metresPerPixel

    // --- Grid patch: labelled lines every `step`, centred on the orbit centre. ---------
    const step = niceStep(mpp * LABEL_SPACING_PX)
    const cells = Math.min(8, Math.max(2, Math.round((Math.min(size.width, size.height) * 0.42 * mpp) / step)))
    const half = cells * step
    const targetWorld = renderToWorld([view.target.x, view.target.y, view.target.z], exaggeration, origin)
    const cx = Math.round(targetWorld[0] / step) * step
    const cy = Math.round(targetWorld[1] / step) * step
    const gz = gridElevation()

    const gridKey = `${step}|${cells}`
    if (gridKey !== keys.current.grid) {
      keys.current.grid = gridKey
      const minor: number[] = []
      const major: number[] = []
      const count = cells * MINOR_PER_MAJOR
      const fine = step / MINOR_PER_MAJOR
      for (let i = -count + 1; i < count; i++) {
        const t = i * fine
        const into = i % MINOR_PER_MAJOR === 0 ? major : minor
        into.push(t, -half, 0, t, half, 0, -half, t, 0, half, t, 0)
      }
      setLines(parts.minor, minor)
      setLines(parts.major, major)
      setLines(parts.border, [
        -half, -half, 0, half, -half, 0,
        half, -half, 0, half, half, 0,
        half, half, 0, -half, half, 0,
        -half, half, 0, -half, -half, 0,
      ])
      state.setGridSpacing(step)
    }
    const gridZ = (gz - origin[2]) * exaggeration
    parts.grid.position.set(cx - origin[0], cy - origin[1], gridZ)

    // --- Which edges face the camera (labels go there; the pole goes to the back). -----
    const dx = camera.position.x - view.target.x
    const dy = camera.position.y - view.target.y
    const reach = Math.hypot(dx, dy, camera.position.z - view.target.z) || 1
    const south = dy <= reach * 1e-3
    const west = dx <= reach * 1e-3
    const edgeY = south ? -half : half
    const edgeX = west ? -half : half
    const gap = step * 0.3
    // Seen edge-on (section views) the plane is a line: put the labels under it instead.
    const edgeOn = Math.abs(camera.position.z - view.target.z) / reach < 0.12
    const labelZ = gridZ - (edgeOn ? gap * 0.45 : 0)

    // --- Elevation pole at the far corner, ticks at true elevations. -------------------
    const zStep = niceStep((mpp * Z_TICK_PX) / exaggeration)
    const bounds = allBounds(state)
    const top = Math.max(bounds ? bounds.max[2] : gz, gz + (2.5 * step) / exaggeration)
    const firstTick = Math.ceil((gz + zStep * 0.25) / zStep) * zStep
    const tickCount = Math.min(400, Math.max(0, Math.floor((top - firstTick) / zStep) + 1))
    const lastTick = firstTick + (tickCount - 1) * zStep
    const poleTop = Math.max(top, lastTick)
    const poleKey = `${step}|${zStep}|${poleTop}|${exaggeration}|${gz}`
    if (poleKey !== keys.current.pole) {
      keys.current.pole = poleKey
      const tick = step * 0.06
      const lines: number[] = [0, 0, 0, 0, 0, (poleTop - gz) * exaggeration]
      for (let i = 0; i < tickCount; i++) {
        const h = (firstTick + i * zStep - gz) * exaggeration
        lines.push(-tick, 0, h, tick, 0, h, 0, -tick, h, 0, tick, h)
      }
      setLines(parts.pole, lines)
    }
    const poleX = cx - origin[0] - edgeX
    const poleY = cy - origin[1] - edgeY
    parts.pole.position.set(poleX, poleY, gridZ)
    // Seen from above the pole is a point: hide it, and skip its labels below.
    const poleOnScreen = ((poleTop - gz) * exaggeration * Math.hypot(dx, dy)) / reach / mpp
    parts.pole.visible = poleOnScreen > 20

    // --- Labels (DOM overlay). ---------------------------------------------------------
    const decimals = decimalsFor(step)
    pool.begin()
    const outward = edgeOn ? 0 : gap
    // In a section view the lines running away from the viewer collapse to one point,
    // so their labels would all land on the same spot: leave that axis unlabelled.
    const hideNorthings = edgeOn && Math.abs(dx) / reach < 0.12
    const hideEastings = edgeOn && Math.abs(dy) / reach < 0.12
    const eY = cy - origin[1] + edgeY + (south ? -outward : outward)
    const nX = cx - origin[0] + edgeX + (west ? -outward : outward)
    // From the middle outwards, so the labels nearest the view centre win overlaps.
    for (let k = 0; k <= 2 * cells; k++) {
      const i = k % 2 === 0 ? k / 2 : -(k + 1) / 2
      if (Math.abs(i) > cells) continue
      const e = cx + i * step
      from.set(e - origin[0], eY, labelZ)
      to.set(e - origin[0], cy - origin[1] - edgeY, labelZ)
      if (!hideEastings) pool.addAlong(camera, size.width, size.height, from, to, `${formatCoord(e, decimals)} E`, 'e')
      const n = cy + i * step
      from.set(nX, n - origin[1], labelZ)
      to.set(cx - origin[0] - edgeX, n - origin[1], labelZ)
      if (!hideNorthings) pool.addAlong(camera, size.width, size.height, from, to, `${formatCoord(n, decimals)} N`, 'n')
    }
    const zDecimals = decimalsFor(zStep)
    for (let i = 0; i < (poleOnScreen > 60 ? tickCount : 0); i++) {
      const elevation = firstTick + i * zStep
      const z = gridZ + (elevation - gz) * exaggeration
      pool.add(camera, size.width, size.height, poleX, poleY, z, `${formatCoord(elevation, zDecimals)} m`, 'z', 7)
    }
    pool.end()

    updateGizmo(camera)
    updateScaleBar(mpp)
  })

  return <primitive object={parts.root} />
}

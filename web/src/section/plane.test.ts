import { beforeEach, expect, test } from 'vitest'
import { renderToWorld, resetOrigin, worldToRender, type Vec3 } from '../scene/origin'
import { sectionTravel, sectionTravelRange, useSection } from '../state/section'
import { useViewer } from '../state/store'
import { describeSection } from './describe'
import {
  boxInPlane,
  compass,
  dot,
  frameOf,
  fromPlane,
  holesInSlab,
  inSlab,
  rayPlane,
  renderClipPlanes,
  sectionFrame,
  sectionFromTwoPoints,
  sectionKind,
  sectionThroughHole,
  signedDistance,
  stepPlane,
  toPlane,
  travelRange,
  type PlaneEq,
  type SectionPlane,
} from './plane'

const ORIGIN: Vec3 = [500600, 5300500, 1050]
const CENTRE: Vec3 = [500594, 5300510, 999]

function cross(a: Vec3, b: Vec3): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function evaluate(plane: PlaneEq, p: Vec3): number {
  return dot(plane.normal, p) + plane.constant
}

test('frames: W-E looks north with east on the right, S-N looks west, a plan looks down', () => {
  expect(sectionFrame(90, 90)).toEqual({ u: [1, 0, 0], v: [0, 0, 1], n: [0, -1, 0] })
  const sn = sectionFrame(0, 90)
  expect(sn.u).toEqual([0, 1, 0])
  expect(sn.n).toEqual([1, 0, 0])
  expect(sn.v).toEqual([0, 0, 1])
  const plan = sectionFrame(90, 0)
  expect(plan.u).toEqual([1, 0, 0])
  expect(plan.v).toEqual([0, 1, 0]) // north is up on screen
  expect(plan.n).toEqual([0, 0, 1])
})

test('every frame is orthonormal and right-handed, with a horizontal section line', () => {
  for (const azimuth of [0, 33, 90, 147.5, 200, 271, 359]) {
    for (const dip of [0, 20, 65, 90]) {
      const { u, v, n } = sectionFrame(azimuth, dip)
      for (const a of [u, v, n]) expect(Math.hypot(...a)).toBeCloseTo(1, 12)
      expect(dot(u, v)).toBeCloseTo(0, 12)
      expect(dot(u, n)).toBeCloseTo(0, 12)
      expect(dot(v, n)).toBeCloseTo(0, 12)
      const c = cross(u, v)
      for (let k = 0; k < 3; k++) expect(c[k]).toBeCloseTo(n[k], 12)
      expect(u[2]).toBe(0)
      expect(v[2]).toBeGreaterThanOrEqual(0) // v never points down
      // The dip is the angle between the plane and the horizontal.
      expect((Math.acos(Math.min(1, Math.abs(n[2]))) * 180) / Math.PI).toBeCloseTo(dip, 9)
    }
  }
})

test('render-space clipping planes keep exactly the slab, under origin offset and exaggeration', () => {
  const cases: { plane: SectionPlane; front: number; back: number }[] = [
    { plane: { centre: CENTRE, azimuth: 90, dip: 90 }, front: 12.5, back: 12.5 },
    { plane: { centre: CENTRE, azimuth: 0, dip: 90 }, front: 5, back: 20 },
    { plane: { centre: CENTRE, azimuth: 90, dip: 0 }, front: 2.5, back: 2.5 },
    { plane: { centre: [500700, 5300400, 1100], azimuth: 37, dip: 55 }, front: 10, back: 3 },
  ]
  for (const { plane, front, back } of cases) {
    // A deterministic cloud of world points around the plane centre.
    const points: Vec3[] = []
    for (let i = 0; i < 400; i++) {
      const f = (k: number) => (Math.abs(Math.sin(i * 12.9898 + k * 78.233) * 43758.5453) % 1) * 2 - 1
      points.push([plane.centre[0] + f(1) * 60, plane.centre[1] + f(2) * 60, plane.centre[2] + f(3) * 60])
    }
    for (const exaggeration of [1, 3, 0.5]) {
      for (const origin of [ORIGIN, [0, 0, 0] as Vec3]) {
        const planes = renderClipPlanes(plane, { front, back }, exaggeration, origin)
        for (const p of planes) expect(Math.hypot(...p.normal)).toBeCloseTo(1, 12)
        let kept = 0
        for (const world of points) {
          const d = signedDistance(plane, world)
          const render = worldToRender(world, exaggeration, origin)
          const shown = planes.every((p) => evaluate(p, render) >= 0)
          // Away from the two faces the answer must agree exactly.
          if (Math.abs(d - front) > 1e-6 && Math.abs(d + back) > 1e-6) expect(shown).toBe(inSlab(d, { front, back }))
          if (shown) kept++
        }
        expect(kept).toBeGreaterThan(0)
        expect(kept).toBeLessThan(points.length)
      }
    }
  }
})

test('clipping planes: distances are true metres along the normal for a vertical section', () => {
  // W-E at N 5 300 510 with +-12.5 m, seen from an origin 10 m south of the plane.
  const plane: SectionPlane = { centre: CENTRE, azimuth: 90, dip: 90 }
  const [near, far] = renderClipPlanes(plane, { front: 12.5, back: 12.5 }, 3, ORIGIN)
  expect(near.normal).toEqual([0, 1, 0])
  expect(far.normal).toEqual([0, -1, 0])
  // Render y of the south face is (510 - 12.5) - 500 = -2.5, of the north face 22.5.
  expect(evaluate(near, [0, -2.5, 999])).toBeCloseTo(0, 9)
  expect(evaluate(far, [123, 22.5, -40])).toBeCloseTo(0, 9)
  expect(evaluate(near, [0, 10, 0])).toBeCloseTo(12.5, 9)
})

test('clipping planes of a plan section scale with the exaggeration', () => {
  const plane: SectionPlane = { centre: CENTRE, azimuth: 90, dip: 0 }
  for (const exaggeration of [1, 3]) {
    const [upper, lower] = renderClipPlanes(plane, { front: 2, back: 4 }, exaggeration, ORIGIN)
    // World z 1001 (top face) and 995 (bottom face) in render space.
    expect(evaluate(upper, [7, 8, (1001 - ORIGIN[2]) * exaggeration])).toBeCloseTo(0, 9)
    expect(evaluate(lower, [7, 8, (995 - ORIGIN[2]) * exaggeration])).toBeCloseTo(0, 9)
    expect(evaluate(upper, [0, 0, (999 - ORIGIN[2]) * exaggeration])).toBeGreaterThan(0)
  }
})

test('stepping moves the plane along the looking direction only', () => {
  const we: SectionPlane = { centre: CENTRE, azimuth: 90, dip: 90 }
  expect(stepPlane(we, 25).centre).toEqual([500594, 5300535, 999]) // forward = north
  expect(stepPlane(we, -25).centre).toEqual([500594, 5300485, 999])
  const sn: SectionPlane = { centre: CENTRE, azimuth: 0, dip: 90 }
  expect(stepPlane(sn, 10).centre).toEqual([500584, 5300510, 999]) // looking west
  const plan: SectionPlane = { centre: CENTRE, azimuth: 90, dip: 0 }
  expect(stepPlane(plan, 5).centre).toEqual([500594, 5300510, 994]) // forward = down
  const oblique: SectionPlane = { centre: CENTRE, azimuth: 37, dip: 55 }
  const moved = stepPlane(oblique, 12)
  expect(signedDistance(oblique, moved.centre)).toBeCloseTo(-12, 9)
  const [s, t] = toPlane(oblique, moved.centre)
  expect(s).toBeCloseTo(0, 9)
  expect(t).toBeCloseTo(0, 9)
})

test('section from two points: the line runs a -> b, vertical, through the midpoint', () => {
  const a: Vec3 = [500500, 5300400, 1200]
  const b: Vec3 = [500700, 5300600, 1180]
  const plane = sectionFromTwoPoints(a, b, 1000)!
  expect(plane.azimuth).toBeCloseTo(45, 12)
  expect(plane.dip).toBe(90)
  expect(plane.centre).toEqual([500600, 5300500, 1000])
  expect(signedDistance(plane, a)).toBeCloseTo(0, 9)
  expect(signedDistance(plane, b)).toBeCloseTo(0, 9)
  // a is on the viewer's left.
  expect(toPlane(plane, a)[0]).toBeLessThan(0)
  expect(toPlane(plane, b)[0]).toBeGreaterThan(0)
  expect(sectionFromTwoPoints([500500, 5300400, 0], [500700, 5300400, 0])!.azimuth).toBe(90)
  expect(sectionFromTwoPoints([500500, 5300400, 0], [500500, 5300300, 0])!.azimuth).toBe(180)
  expect(sectionFromTwoPoints(a, a)).toBeNull()
})

test('section through a hole: the best-fit vertical plane contains a planar trace', () => {
  // A hole drilled towards azimuth 132.69 at -60 that steepens with depth, in one plane.
  const azimuth = (132.69 * Math.PI) / 180
  const xyz: number[] = []
  let x = 500548
  let y = 5300308
  let z = 1192
  for (let i = 0; i <= 20; i++) {
    xyz.push(x, y, z)
    const dip = ((60 + i) * Math.PI) / 180
    x += 5 * Math.cos(dip) * Math.sin(azimuth)
    y += 5 * Math.cos(dip) * Math.cos(azimuth)
    z -= 5 * Math.sin(dip)
  }
  const plane = sectionThroughHole(xyz)!
  expect(plane.dip).toBe(90)
  expect(plane.azimuth).toBeCloseTo(132.69, 6)
  for (let i = 0; i < xyz.length; i += 3) {
    expect(Math.abs(signedDistance(plane, [xyz[i], xyz[i + 1], xyz[i + 2]]))).toBeLessThan(1e-6)
  }
  // The hole runs from top left to bottom right.
  const first = toPlane(plane, [xyz[0], xyz[1], xyz[2]])
  const last = toPlane(plane, [xyz[60], xyz[61], xyz[62]])
  expect(last[0]).toBeGreaterThan(first[0])
  expect(last[1]).toBeLessThan(first[1])
})

test('section through a hole: least-squares for a bent trace, W-E for a vertical one', () => {
  // A trace that wanders sideways: residuals are small and balanced.
  const xyz: number[] = []
  for (let i = 0; i <= 30; i++) xyz.push(500000 + i * 3 + Math.sin(i) * 0.4, 5300000 + i * 4 - Math.sin(i) * 0.3, 1000 - i * 8)
  const plane = sectionThroughHole(xyz)!
  expect(plane.azimuth).toBeGreaterThan(35)
  expect(plane.azimuth).toBeLessThan(39) // atan2(3, 4) = 36.87
  let sum = 0
  let worst = 0
  for (let i = 0; i < xyz.length; i += 3) {
    const d = signedDistance(plane, [xyz[i], xyz[i + 1], xyz[i + 2]])
    sum += d
    worst = Math.max(worst, Math.abs(d))
  }
  expect(Math.abs(sum)).toBeLessThan(1e-6)
  expect(worst).toBeLessThan(0.6)

  const vertical = sectionThroughHole([500100, 5300200, 1200, 500100, 5300200, 1100, 500100, 5300200, 1000])!
  expect(vertical.azimuth).toBe(90)
  expect(vertical.centre).toEqual([500100, 5300200, 1100])
  expect(sectionThroughHole([])).toBeNull()
})

test('ray-plane picking: the hit is on the plane and on the ray, in world and through render space', () => {
  const plane: SectionPlane = { centre: CENTRE, azimuth: 90, dip: 90 }
  // Looking north from 400 m south of the plane.
  const hit = rayPlane(plane, [500650.25, 5300110, 1034.5], [0, 1, 0])!
  expect(hit).toEqual([500650.25, 5300510, 1034.5])
  expect(rayPlane(plane, [500650, 5300110, 1034], [1, 0, 0])).toBeNull() // parallel
  expect(rayPlane(plane, [500650, 5300110, 1034], [0, -1, 0])).toBeNull() // behind
  expect(rayPlane(plane, [500650, 5300110, 1034], [0, -1, 0], true)).not.toBeNull()

  // A perspective ray given in render space with exaggeration 3, as the cursor probe has it.
  const oblique: SectionPlane = { centre: [500700, 5300400, 1100], azimuth: 37, dip: 55 }
  for (const exaggeration of [1, 3]) {
    const originRender: Vec3 = [-320, -410, 900]
    const directionRender: Vec3 = [0.5, 0.62, -0.6]
    const p = rayPlane(
      oblique,
      renderToWorld(originRender, exaggeration, ORIGIN),
      [directionRender[0], directionRender[1], directionRender[2] / exaggeration],
    )!
    expect(Math.abs(signedDistance(oblique, p))).toBeLessThan(1e-9)
    // Back in render space the point lies on the original ray.
    const r = worldToRender(p, exaggeration, ORIGIN)
    const rel = [r[0] - originRender[0], r[1] - originRender[1], r[2] - originRender[2]] as Vec3
    const t = dot(rel, directionRender) / dot(directionRender, directionRender)
    expect(t).toBeGreaterThan(0)
    for (let k = 0; k < 3; k++) expect(rel[k]).toBeCloseTo(directionRender[k] * t, 6)
  }
})

test('plane coordinates round-trip and box extents', () => {
  const plane: SectionPlane = { centre: CENTRE, azimuth: 37, dip: 55 }
  const p = fromPlane(plane, 12, -7, 3)
  const [s, t, w] = toPlane(plane, p)
  expect(s).toBeCloseTo(12, 9)
  expect(t).toBeCloseTo(-7, 9)
  expect(w).toBeCloseTo(3, 9)
  expect(signedDistance(plane, p)).toBeCloseTo(3, 9)

  const we: SectionPlane = { centre: CENTRE, azimuth: 90, dip: 90 }
  const box = { min: [500400, 5300300, 900] as Vec3, max: [500800, 5300700, 1200] as Vec3 }
  const extent = boxInPlane(we, box)
  expect(extent.min).toEqual([-194, -99, -190])
  expect(extent.max).toEqual([206, 201, 210])
  // The plane is 210 m north of the south face and 190 m south of the north face.
  expect(travelRange(we, box)).toEqual([-210, 190])
  expect(sectionTravel(we)).toBe(5300510)
  expect(sectionTravelRange(we, box)).toEqual([5300300, 5300700])
})

test('holes in the slab: flags and the point where each hole enters it', () => {
  const plane: SectionPlane = { centre: [0, 100, 0], azimuth: 90, dip: 90 } // n = south
  const slab = { front: 10, back: 10 } // northing 90 .. 110
  const positions = Float64Array.from([
    // hole 0: vertical at N 100 (inside)
    0, 100, 50, 0, 100, 0,
    // hole 1: vertical at N 150 (outside)
    5, 150, 50, 5, 150, 0,
    // hole 2: drilled north from N 60 to N 140 in two pieces: enters at N 90
    10, 60, 50, 10, 80, 40, 10, 80, 40, 10, 140, 10,
    // hole 3: drilled south from N 130 to N 105: enters at N 110
    20, 130, 50, 20, 105, 25,
    // hole 4: one long piece right across the slab
    30, 0, 50, 30, 200, 50,
  ])
  const offsets = Uint32Array.from([0, 1, 2, 4, 5, 6])
  const { inside, anchors, count } = holesInSlab(plane, slab, positions, offsets)
  expect(Array.from(inside)).toEqual([1, 0, 1, 1, 1])
  expect(count).toBe(4)
  expect(Array.from(anchors.subarray(0, 3))).toEqual([0, 100, 50])
  expect(Number.isNaN(anchors[3])).toBe(true)
  expect(anchors[6]).toBeCloseTo(10, 9)
  expect(anchors[7]).toBeCloseTo(90, 9)
  expect(anchors[8]).toBeCloseTo(35, 9)
  expect(anchors[10]).toBeCloseTo(110, 9)
  expect(anchors[11]).toBeCloseTo(30, 9)
  expect(anchors[13]).toBeCloseTo(90, 9)
})

test('kinds, compass and the one-line description', () => {
  expect(sectionKind({ centre: CENTRE, azimuth: 90, dip: 90 })).toBe('we')
  expect(sectionKind({ centre: CENTRE, azimuth: 270, dip: 90 })).toBe('we')
  expect(sectionKind({ centre: CENTRE, azimuth: 0, dip: 90 })).toBe('sn')
  expect(sectionKind({ centre: CENTRE, azimuth: 45, dip: 90 })).toBe('vertical')
  expect(sectionKind({ centre: CENTRE, azimuth: 45, dip: 60 })).toBe('inclined')
  expect(sectionKind({ centre: CENTRE, azimuth: 90, dip: 0 })).toBe('plan')
  expect(compass(0)).toBe('north')
  expect(compass(350)).toBe('north')
  expect(compass(225)).toBe('south-west')
  expect(describeSection({ centre: [500594, 5300650, 999], azimuth: 90, dip: 90 }, { front: 12.5, back: 12.5 })).toBe(
    'N 5\u202f300\u202f650 · looking north · ±12.5 m',
  )
  expect(describeSection({ centre: CENTRE, azimuth: 0, dip: 90 }, { front: 5, back: 20 })).toBe('E 500\u202f594 · looking west · +5 / −20 m')
  expect(describeSection({ centre: CENTRE, azimuth: 90, dip: 0 }, { front: 2.5, back: 2.5 })).toBe('Z 999 · plan · ±2.5 m')
})

// ---- the store

beforeEach(() => {
  resetOrigin()
  useViewer.setState({ objects: {}, projection: 'perspective' })
  useSection.setState({
    enabled: false,
    plane: { centre: [0, 0, 0], azimuth: 90, dip: 90 },
    front: 12.5,
    back: 12.5,
    step: 25,
    mode: 'clip',
    locked: false,
    draft: null,
  })
  useViewer.getState().registerObject({
    id: 'box',
    name: 'Box',
    category: 'solids',
    bounds: { min: [500400, 5300300, 900], max: [500800, 5300700, 1200] },
  })
})

test('store: presets go through the middle of the data, or at a given coordinate', () => {
  const section = useSection.getState()
  section.preset('we')
  expect(useSection.getState().enabled).toBe(true)
  expect(useSection.getState().plane).toEqual({ centre: [500600, 5300500, 1050], azimuth: 90, dip: 90 })
  section.preset('sn', 500650)
  expect(useSection.getState().plane).toEqual({ centre: [500650, 5300500, 1050], azimuth: 0, dip: 90 })
  section.preset('plan', 1000)
  expect(useSection.getState().plane).toEqual({ centre: [500600, 5300500, 1000], azimuth: 90, dip: 0 })
})

test('store: stepping, travel, thickness and step size', () => {
  const section = useSection.getState()
  section.stepBy(1) // off: nothing moves
  expect(useSection.getState().plane.centre).toEqual([0, 0, 0])
  section.preset('we')
  section.stepBy(2)
  expect(useSection.getState().plane.centre[1]).toBe(5300550)
  section.setStep(10)
  section.stepBy(-1)
  expect(useSection.getState().plane.centre[1]).toBe(5300540)
  section.setStep(-3)
  section.setStep(Number.NaN)
  expect(useSection.getState().step).toBe(10)
  section.setTravel(5300333)
  expect(useSection.getState().plane.centre).toEqual([500600, 5300333, 1050])

  section.setThickness({ front: 5 })
  expect(useSection.getState().front).toBe(5)
  expect(useSection.getState().back).toBe(12.5)
  section.setThickness({ front: 0, back: 0 }) // an empty slab is refused
  expect(useSection.getState().front).toBe(5)
  section.setThickness({ front: 0 })
  expect(useSection.getState().front).toBe(0)
  section.setPlane({ azimuth: -90, dip: 120 })
  expect(useSection.getState().plane.azimuth).toBe(270)
  expect(useSection.getState().plane.dip).toBe(90)
})

test('store: the section view locks, switches to orthographic, and unlocks when switched off', () => {
  const section = useSection.getState()
  section.lookAt()
  expect(useSection.getState().enabled).toBe(true)
  expect(useSection.getState().locked).toBe(true)
  expect(useViewer.getState().projection).toBe('orthographic')
  // First use puts the plane through the middle of the data.
  expect(useSection.getState().plane.centre).toEqual([500600, 5300500, 1050])
  section.setEnabled(false)
  expect(useSection.getState().locked).toBe(false)
})

test('store: drawing a line and a section through a hole', () => {
  const section = useSection.getState()
  section.startDrawing()
  expect(useSection.getState().draft).toEqual({ a: null, b: null })
  section.setDraft({ a: [500500, 5300400, 900], b: [500500, 5300400, 900] })
  expect(section.fromTwoPoints([500500, 5300400, 900], [500700, 5300600, 900])).toBe(true)
  const state = useSection.getState()
  expect(state.draft).toBeNull()
  expect(state.enabled).toBe(true)
  expect(state.plane.azimuth).toBeCloseTo(45, 12)
  expect(state.plane.centre).toEqual([500600, 5300500, 1050]) // mid elevation of the data
  expect(frameOf(state.plane).n[2]).toBe(0)

  expect(section.throughHole([500500, 5300400, 1200, 500500, 5300300, 1000])).toBe(true)
  expect(useSection.getState().plane.azimuth).toBe(180)
  expect(section.throughHole([])).toBe(false)
})

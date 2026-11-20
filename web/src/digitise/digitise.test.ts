import { expect, test } from 'vitest'
import { decodeBkma } from '../api/bkma'
import { decodeDrillholes, type DrillholeSet } from '../api/client'
import drillholesUri from '../api/fixtures/drillholes.bkma?inline'
import { buildPieces } from '../drillholes/pieces'
import type { Vec3 } from '../scene/origin'
import { frameOf, fromPlane, signedDistance, type SectionPlane } from '../section/plane'
import {
  addVertex,
  beginEdit,
  canClose,
  canUndo,
  closeString,
  deleteString,
  deleteVertex,
  EMPTY_EDIT,
  insertVertex,
  loadStrings,
  MAX_UNDO,
  moveVertex,
  newString,
  undo,
  type EditState,
} from './edit'
import { hitEdge, hitVertex, inSlabString, nearestSnap, onPlane, renderDistance, sectionId, snapTargets, type PieceSource } from './snap'

const WE: SectionPlane = { centre: [1000, 2000, 300], azimuth: 90, dip: 90 }

function draw(points: Vec3[], section = 'A', state: EditState = EMPTY_EDIT): EditState {
  return points.reduce((s, p) => addVertex(s, section, p), state)
}

// ---------------------------------------------------------------------- editing

test('click-by-click: vertices go to one string, close needs three, then a new string starts', () => {
  let s = draw([[0, 0, 0], [10, 0, 0]])
  expect(s.strings).toHaveLength(1)
  expect(s.strings[0]).toMatchObject({ id: 'd1', section: 'A', closed: false })
  expect(canClose(s)).toBe(false)
  expect(closeString(s)).toBe(s)
  s = addVertex(s, 'A', [10, 0, 10])
  expect(canClose(s)).toBe(true)
  s = closeString(s)
  expect(s.strings[0].closed).toBe(true)
  expect(s.strings[0].points).toHaveLength(3) // the closing point is not repeated
  expect(s.active).toBeNull()
  s = addVertex(s, 'A', [50, 0, 0])
  expect(s.strings.map((x) => x.id)).toEqual(['d1', 'd2'])
  expect(s.active).toBe('d2')
})

test('a click on another section starts a new string there; newString finishes the active one', () => {
  let s = draw([[0, 0, 0], [1, 0, 0]], 'A')
  s = addVertex(s, 'B', [0, 25, 0])
  expect(s.strings.map((x) => [x.id, x.section, x.points.length])).toEqual([
    ['d1', 'A', 2],
    ['d2', 'B', 1],
  ])
  s = newString(s)
  expect(s.active).toBeNull()
  expect(newString(s)).toBe(s)
  s = addVertex(s, 'B', [5, 25, 0])
  expect(s.strings).toHaveLength(3)
})

test('move, insert and delete vertices; a closed string with two points left opens', () => {
  let s = closeString(draw([[0, 0, 0], [10, 0, 0], [10, 0, 10], [0, 0, 10]]))
  s = moveVertex(s, { string: 'd1', index: 2 }, [12, 0, 12])
  expect(s.strings[0].points[2]).toEqual([12, 0, 12])
  s = insertVertex(s, { string: 'd1', index: 3 }, [-2, 0, 5]) // on the closing edge
  expect(s.strings[0].points).toEqual([[0, 0, 0], [10, 0, 0], [12, 0, 12], [0, 0, 10], [-2, 0, 5]])
  expect(s.selected).toEqual({ string: 'd1', index: 4 })
  s = deleteVertex(s, { string: 'd1', index: 0 })
  s = deleteVertex(s, { string: 'd1', index: 0 })
  expect(s.strings[0].closed).toBe(true)
  s = deleteVertex(s, { string: 'd1', index: 0 })
  expect(s.strings[0].points).toHaveLength(2)
  expect(s.strings[0].closed).toBe(false)
  s = deleteVertex(deleteVertex(s, { string: 'd1', index: 0 }), { string: 'd1', index: 0 })
  expect(s.strings).toHaveLength(0)
  // Out-of-range references change nothing.
  expect(deleteVertex(s, { string: 'zz', index: 0 })).toBe(s)
  expect(moveVertex(s, { string: 'd1', index: 9 }, [0, 0, 0])).toBe(s)
})

test('undo steps back one edit at a time; a drag is one edit', () => {
  const states: EditState[] = [EMPTY_EDIT]
  let s = EMPTY_EDIT
  for (const p of [[0, 0, 0], [10, 0, 0], [10, 0, 10]] as Vec3[]) {
    s = addVertex(s, 'A', p)
    states.push(s)
  }
  s = closeString(s)
  // Drag: recorded once, then moved many times without recording.
  s = beginEdit(s)
  for (let k = 1; k <= 5; k++) s = moveVertex(s, { string: 'd1', index: 1 }, [10 + k, 0, 0], false)
  expect(s.strings[0].points[1]).toEqual([15, 0, 0])
  s = undo(s)
  expect(s.strings[0].points[1]).toEqual([10, 0, 0])
  expect(s.strings[0].closed).toBe(true)
  s = undo(s) // the close
  expect(s.strings[0].closed).toBe(false)
  expect(s.active).toBe('d1')
  for (let k = 3; k >= 0; k--) {
    expect(s.strings).toEqual(states[k].strings)
    s = undo(s)
  }
  expect(canUndo(s)).toBe(false)
  expect(undo(s)).toBe(s)

  let long = EMPTY_EDIT
  for (let k = 0; k < MAX_UNDO + 20; k++) long = addVertex(long, 'A', [k, 0, 0])
  expect(long.past).toHaveLength(MAX_UNDO)
})

test('deleting a string and loading saved strings', () => {
  let s = draw([[0, 0, 0], [1, 0, 0]])
  s = deleteString(s, 'd1')
  expect(s.strings).toEqual([])
  expect(s.active).toBeNull()
  expect(undo(s).strings).toHaveLength(1)
  const loaded = loadStrings([{ id: 'd7', section: 'A', closed: true, points: [[0, 0, 0], [1, 0, 0], [1, 0, 1]] }])
  expect(loaded.past).toEqual([])
  expect(addVertex(loaded, 'A', [5, 5, 5]).active).toBe('d8')
})

// ---------------------------------------------------------------------- snapping

/** One straight hole from `a` to `b`, cut into pieces at `depths`, intervals as given. */
function straightHole(a: Vec3, b: Vec3, depths: number[], intervals: number[]): PieceSource {
  const length = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])
  const at = (d: number) => [0, 1, 2].map((k) => a[k] + ((b[k] - a[k]) * d) / length)
  const positions: number[] = []
  const depth: number[] = []
  for (let k = 0; k + 1 < depths.length; k++) {
    positions.push(...at(depths[k]), ...at(depths[k + 1]))
    depth.push(depths[k], depths[k + 1])
  }
  const count = depths.length - 1
  return { count, positions, depth, interval: intervals, hole: new Array(count).fill(0), holeOffsets: [0, count] }
}

test('a hole crossing the plane snaps at the analytic crossing point', () => {
  // A W-E section at northing 2000 (normal along -y); a hole dipping north through it.
  const hole = straightHole([1010, 1990, 400], [1010, 2030, 300], [0, 20, 40, 60, Math.hypot(40, 100)], [-1, 0, 1, -1])
  const targets = snapTargets(WE, { front: 5, back: 5 }, hole, 'dh')
  const crossings = targets.filter((t) => t.kind === 'crossing')
  expect(crossings).toHaveLength(1)
  // y = 2000 is 10/40 of the way: z = 400 - 25 = 375, depth = 0.25 * length.
  expect(crossings[0].point[0]).toBeCloseTo(1010, 9)
  expect(crossings[0].point[1]).toBeCloseTo(2000, 9)
  expect(crossings[0].point[2]).toBeCloseTo(375, 9)
  expect(crossings[0].depth).toBeCloseTo(0.25 * Math.hypot(40, 100), 9)

  // Boundaries: interval 0 starts at depth 20 (y = 1997.4, inside the 5 m slab), its end
  // / interval 1's start at 40 (y = 2004.9, inside), interval 1's end at 60 (y = 2012.3, outside).
  const boundaries = targets.filter((t) => t.kind === 'boundary')
  expect(boundaries.map((t) => t.depth)).toEqual([20, 40])
  for (const t of targets) expect(Math.abs(signedDistance(WE, t.point))).toBeLessThan(1e-9)
  // The boundary at depth 20 is projected square onto the plane: x and z unchanged.
  const f = 20 / Math.hypot(40, 100)
  expect(boundaries[0].point[2]).toBeCloseTo(400 - 100 * f, 9)
  expect(boundaries[0].point[1]).toBeCloseTo(2000, 9)
})

test('nearestSnap picks the closest target within the tolerance, in exaggerated space', () => {
  const hole = straightHole([1010, 1990, 400], [1010, 2030, 300], [0, 20, 40, 60, Math.hypot(40, 100)], [-1, 0, 1, -1])
  const targets = snapTargets(WE, { front: 5, back: 5 }, hole, 'dh')
  const crossing = targets.find((t) => t.kind === 'crossing')!
  const near: Vec3 = [crossing.point[0] + 1.5, 2000, crossing.point[2]]
  expect(nearestSnap(targets, near, 2, 1)).toBe(crossing)
  expect(nearestSnap(targets, near, 1, 1)).toBeNull()
  // 1.5 m above: 3 m on screen at exaggeration 2.
  const above: Vec3 = [crossing.point[0], 2000, crossing.point[2] + 1.5]
  expect(renderDistance(above, crossing.point, 2)).toBeCloseTo(3, 12)
  expect(nearestSnap(targets, above, 2, 2)).toBeNull()
  expect(nearestSnap(targets, above, 2, 1)).toBe(crossing)
})

function fixture(): DrillholeSet {
  const uri = drillholesUri
  const bytes = Uint8Array.from(atob(uri.slice(uri.indexOf(',') + 1)), (c) => c.charCodeAt(0))
  return decodeDrillholes(decodeBkma(bytes.buffer))
}

test('on the fixture holes, boundaries are the interval ends located by the server', () => {
  const set = fixture()
  const seg = set.segments!
  const pieces = buildPieces(set)
  // A thick slab around a plane through the middle of the first hole: every interval end
  // inside it is a target, and nothing else is.
  const h0 = [set.pointXyz[0], set.pointXyz[1], set.pointXyz[2]]
  const plane: SectionPlane = { centre: [h0[0], h0[1], h0[2] - 30], azimuth: 90, dip: 90 }
  const slab = { front: 40, back: 40 }
  const targets = snapTargets(plane, slab, pieces, 'dh')
  const boundaries = targets.filter((t) => t.kind === 'boundary')
  const { n } = frameOf(plane)
  const expected = new Set<string>()
  const key = (h: number, d: number) => `${h}:${d.toFixed(6)}`
  for (let s = 0; s < seg.count; s++) {
    for (const [xyz, depth] of [
      [seg.start, seg.from[s]],
      [seg.end, seg.to[s]],
    ] as const) {
      const p: Vec3 = [xyz[s * 3], xyz[s * 3 + 1], xyz[s * 3 + 2]]
      const d = signedDistance(plane, p, n)
      if (d <= slab.front && d >= -slab.back) expected.add(key(seg.hole[s], depth))
    }
  }
  expect(expected.size).toBeGreaterThan(0)
  expect(new Set(boundaries.map((t) => key(t.hole, t.depth)))).toEqual(expected)
  expect(boundaries.length).toBe(expected.size) // each boundary once
})

test('section ids, on-plane and in-slab tests, vertex and edge hits', () => {
  expect(sectionId(WE)).toBe('az090.0-dip90.0-at+2000.00')
  expect(sectionId({ centre: [0, 0, 250], azimuth: 90, dip: 0 })).toBe('az090.0-dip0.0-at-250.00')
  const square = [fromPlane(WE, 0, 0), fromPlane(WE, 10, 0), fromPlane(WE, 10, 10), fromPlane(WE, 0, 10)]
  const string = { id: 'd1', section: sectionId(WE), closed: true, points: square }
  expect(onPlane(string, WE)).toBe(true)
  const moved = { ...WE, centre: [1000, 2003, 300] as Vec3 }
  expect(onPlane(string, moved)).toBe(false)
  expect(inSlabString(string, moved, { front: 5, back: 5 })).toBe(true)
  expect(inSlabString(string, moved, { front: 1, back: 1 })).toBe(false)

  expect(hitVertex([string], fromPlane(WE, 10.5, 9.6), 1, 1)).toEqual({ string: 'd1', index: 2 })
  expect(hitVertex([string], fromPlane(WE, 5, 5), 1, 1)).toBeNull()
  // The closing edge (3 -> 0) of a closed string can be hit; of an open one it cannot.
  expect(hitEdge([string], fromPlane(WE, 0.4, 5), 1, 1)).toEqual({ string: 'd1', index: 3 })
  expect(hitEdge([{ ...string, closed: false }], fromPlane(WE, 0.4, 5), 1, 1)).toBeNull()
  expect(hitEdge([string], fromPlane(WE, 5, 0.5), 1, 1)).toEqual({ string: 'd1', index: 0 })
})

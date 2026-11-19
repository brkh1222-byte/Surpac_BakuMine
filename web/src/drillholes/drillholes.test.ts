import { expect, test } from 'vitest'
import { decodeBkma } from '../api/bkma'
import { decodeDrillholes, type DrillholeSet } from '../api/client'
import drillholesUri from '../api/fixtures/drillholes.bkma?inline'
import { rampColor, MISSING_COLOR, CATEGORICAL, hexToRgb, srgbToLinear } from '../colour/ramps'
import { buildPerfDrillholes } from '../scene/perfData'
import { drillholeLegend, findAttribute, holeDetails, INK, pieceColors, readouts } from './colouring'
import { declutter } from './declutter'
import { buildPieces } from './pieces'
import { holeSpheres, pickPiece, type PickRay } from './pick'

function fixture(): DrillholeSet {
  const uri = drillholesUri
  const bytes = Uint8Array.from(atob(uri.slice(uri.indexOf(',') + 1)), (c) => c.charCodeAt(0))
  return decodeDrillholes(decodeBkma(bytes.buffer))
}

/** Sum of the depth ranges of the pieces of a hole that belong to `interval`. */
function coveredLength(pieces: ReturnType<typeof buildPieces>, hole: number, interval: number): number {
  let sum = 0
  for (let p = pieces.holeOffsets[hole]; p < pieces.holeOffsets[hole + 1]; p++) {
    if (pieces.interval[p] === interval) sum += pieces.depth[p * 2 + 1] - pieces.depth[p * 2]
  }
  return sum
}

test('pieces cover every hole from collar to end exactly once, intervals and gaps', () => {
  const set = fixture()
  const seg = set.segments
  if (!seg) throw new Error('no segments')
  const pieces = buildPieces(set)
  expect(pieces.holeOffsets[0]).toBe(0)
  expect(pieces.holeOffsets[3]).toBe(pieces.count)

  for (let h = 0; h < 3; h++) {
    const first = pieces.holeOffsets[h]
    const last = pieces.holeOffsets[h + 1] - 1
    // Contiguous in depth and in space, from 0 to the end of hole.
    expect(pieces.depth[first * 2]).toBe(0)
    expect(pieces.depth[last * 2 + 1]).toBe(set.maxDepth[h])
    for (let p = first; p <= last; p++) {
      expect(pieces.hole[p]).toBe(h)
      expect(pieces.depth[p * 2 + 1]).toBeGreaterThan(pieces.depth[p * 2])
      if (p > first) {
        expect(pieces.depth[p * 2]).toBe(pieces.depth[p * 2 - 1])
        for (let k = 0; k < 3; k++) expect(pieces.positions[p * 6 + k]).toBeCloseTo(pieces.positions[p * 6 - 3 + k], 9)
      }
    }
    // Starts at the collar.
    for (let k = 0; k < 3; k++) expect(pieces.positions[first * 6 + k]).toBe(set.collarXyz[h * 3 + k])
  }

  // Every interval is fully covered by its own pieces; its ends are the exact payload points.
  for (let s = 0; s < seg.count; s++) {
    const h = seg.hole[s]
    expect(coveredLength(pieces, h, s)).toBeCloseTo(seg.to[s] - seg.from[s], 9)
    const mine = [...pieces.interval.keys()].filter((p) => pieces.interval[p] === s)
    const a = mine[0]
    const b = mine[mine.length - 1]
    for (let k = 0; k < 3; k++) {
      expect(pieces.positions[a * 6 + k]).toBe(seg.start[s * 3 + k])
      expect(pieces.positions[b * 6 + 3 + k]).toBe(seg.end[s * 3 + k])
    }
  }
  // Gaps: T2 has none sampled between 20 and 45 m, T3 none between 0 and 5 m.
  expect(coveredLength(pieces, 1, -1)).toBeCloseTo(25, 9)
  expect(coveredLength(pieces, 2, -1)).toBeCloseTo(5, 9)
  expect(coveredLength(pieces, 0, -1)).toBe(0)

  // The curved hole's long intervals are cut at the polyline points inside them.
  const curved = [...pieces.interval.keys()].filter((p) => pieces.interval[p] === 3) // T2 45-70 m
  expect(curved.length).toBeGreaterThan(1)
  // The straight hole's intervals are single pieces.
  expect([...pieces.interval].filter((s) => s === 1).length).toBe(1)
})

test('without an interval table the pieces are the polyline', () => {
  const set = { ...fixture(), segments: null }
  const pieces = buildPieces(set)
  expect(pieces.count).toBe(set.pointDepth.length - set.holeIds.length)
  expect([...pieces.interval].every((s) => s === -1)).toBe(true)
  expect(Array.from(pieces.positions.slice(0, 3))).toEqual(Array.from(set.pointXyz.slice(0, 3)))
})

test('overlapping and unordered intervals are all drawn', () => {
  const set = fixture()
  const seg = set.segments
  if (!seg) throw new Error('no segments')
  // Make T1's second interval (10-25) overlap the first: 5-25.
  seg.from[2] = 5
  const pieces = buildPieces(set)
  expect(coveredLength(pieces, 0, 2)).toBeCloseTo(20, 9)
  expect(coveredLength(pieces, 0, 1)).toBeCloseTo(10, 9)
  expect(coveredLength(pieces, 0, -1)).toBe(0)
})

test('piece colours: value colours, neutral for missing values and for gaps', () => {
  const set = fixture()
  const pieces = buildPieces(set)
  const au = findAttribute(set, 'assay/au')
  const rock = findAttribute(set, 'assay/rock')
  expect(findAttribute(set, 'geology/rock')).toBeNull()
  expect(findAttribute(set, null)).toBeNull()
  const settings = { mode: 'continuous' as const, min: 0, max: 4, steps: 6 }
  const spec = drillholeLegend(au, undefined, settings)
  if (!spec || spec.kind !== 'numeric') throw new Error('no numeric legend')
  expect(spec.hasMissing).toBe(true)
  expect(spec.unit).toBe('g/t')

  const colors = pieceColors(pieces, au, spec, 'dark')
  expect(colors.length).toBe(pieces.count * 6)
  const linear = (hex: readonly number[]) => hex.map(srgbToLinear)
  const colorOfInterval = (s: number) => {
    const p = pieces.interval.indexOf(s)
    return Array.from(colors.slice(p * 6, p * 6 + 3))
  }
  const close = (got: number[], want: number[]) => got.forEach((v, k) => expect(v).toBeCloseTo(want[k], 6))
  close(colorOfInterval(3), linear(rampColor('grade', 'dark', 3.5 / 4))) // au 3.5
  close(colorOfInterval(6), linear(rampColor('grade', 'dark', 1))) // au 12.125, clamped
  close(colorOfInterval(2), linear(hexToRgb(MISSING_COLOR.dark))) // au missing: not the zero colour
  close(colorOfInterval(-1), linear(hexToRgb(MISSING_COLOR.dark))) // gap
  expect(colorOfInterval(2)).not.toEqual(Array.from(linear(rampColor('grade', 'dark', 0))))
  // Start and end colour of a piece are equal: flat colour per interval.
  for (let p = 0; p < pieces.count; p++) {
    expect(Array.from(colors.slice(p * 6, p * 6 + 3))).toEqual(Array.from(colors.slice(p * 6 + 3, p * 6 + 6)))
  }

  const rockSpec = drillholeLegend(rock, undefined, undefined)
  if (!rockSpec || rockSpec.kind !== 'categorical') throw new Error('no categorical legend')
  expect(rockSpec.categories).toEqual(['DYKE', 'ORE', 'WASTE'])
  expect(rockSpec.hasMissing).toBe(true)
  const rockColors = pieceColors(pieces, rock, rockSpec, 'light')
  const p5 = pieces.interval.indexOf(5) // DYKE
  close(Array.from(rockColors.slice(p5 * 6, p5 * 6 + 3)), linear(hexToRgb(CATEGORICAL.light[0])))
  const p4 = pieces.interval.indexOf(4) // no rock code
  close(Array.from(rockColors.slice(p4 * 6, p4 * 6 + 3)), linear(hexToRgb(MISSING_COLOR.light)))

  // No attribute: one trace colour everywhere.
  const plain = pieceColors(pieces, null, null, 'dark')
  close(Array.from(plain.slice(0, 3)), linear(hexToRgb(INK.dark.trace)))
})

test('read-outs and hole details', () => {
  const set = fixture()
  expect(readouts(set, 3)).toEqual([
    { name: 'au', text: '3.500 g/t' },
    { name: 'rock', text: 'ORE' },
  ])
  expect(readouts(set, 2)[0]).toEqual({ name: 'au', text: 'no value' })
  expect(readouts(set, 4)[1]).toEqual({ name: 'rock', text: 'no value' })
  expect(readouts(set, -1)).toEqual([])

  const details = holeDetails('holes', set, 0, null, null, 'dark')
  expect(details.holeId).toBe('T1')
  expect(details.maxDepth).toBe(40)
  expect(details.collar).toEqual([500100, 5300200, 1210.5])
  expect(details.columns).toEqual(['au (g/t)', 'rock'])
  expect(details.rows.map((r) => [r.from, r.to])).toEqual([[0, 10], [10, 25], [25, 40]])
  expect(details.rows[1].values).toEqual(['no value', 'ORE'])
  expect(details.rows[0].color).toBeNull()
})

// ---------------------------------------------------------------------- picking

/** Two vertical holes 100 m apart, each with ten 10 m pieces, plus a hole with none. */
function twoHoles() {
  const positions = new Float32Array(20 * 6)
  for (let h = 0; h < 2; h++) {
    for (let i = 0; i < 10; i++) {
      positions.set([h * 100, 0, -i * 10, h * 100, 0, -(i + 1) * 10], (h * 10 + i) * 6)
    }
  }
  const holeOffsets = Uint32Array.from([0, 10, 10, 20])
  return { positions, holeOffsets, spheres: holeSpheres(positions, holeOffsets) }
}

const ray = (origin: [number, number, number], direction: [number, number, number], tol0 = 1, tolSlope = 0): PickRay => {
  const n = Math.hypot(...direction)
  return { origin, direction: [direction[0] / n, direction[1] / n, direction[2] / n], tol0, tolSlope }
}

test('pick: the piece index under the ray, and where on it', () => {
  const { positions, holeOffsets, spheres } = twoHoles()
  expect(Array.from(spheres.slice(0, 4))).toEqual([0, 0, -50, 50])
  expect(spheres[7]).toBe(-1) // the empty hole can never be hit

  // Looking north (+y) at hole 0, 34 m down: piece 3, 40 % along it.
  const hit = pickPiece(ray([0.2, -500, -34], [0, 1, 0]), positions, holeOffsets, spheres)
  expect(hit?.piece).toBe(3)
  expect(hit?.s).toBeCloseTo(0.4, 6)
  expect(hit?.t).toBeCloseTo(500, 3)
  expect(hit?.closeness).toBeCloseTo(0.2, 5)

  // The second hole: pieces 10..19.
  expect(pickPiece(ray([100, -500, -95], [0, 1, 0]), positions, holeOffsets, spheres)?.piece).toBe(19)
  // Between the holes: nothing within 1 m.
  expect(pickPiece(ray([50, -500, -30], [0, 1, 0]), positions, holeOffsets, spheres)).toBeNull()
  // Just outside the tolerance, then inside a wider one.
  expect(pickPiece(ray([1.5, -500, -30], [0, 1, 0]), positions, holeOffsets, spheres)).toBeNull()
  expect(pickPiece(ray([1.5, -500, -30], [0, 1, 0], 2), positions, holeOffsets, spheres)?.piece).toBe(2)
  // Behind the camera: not picked.
  expect(pickPiece(ray([0, 500, -30], [0, 1, 0]), positions, holeOffsets, spheres)).toBeNull()
  // Below the end of the hole: the last piece's end point is still 5 m away.
  expect(pickPiece(ray([0, -500, -105], [0, 1, 0]), positions, holeOffsets, spheres)).toBeNull()
})

test('pick: of two candidates the one nearer to the cursor wins; perspective tolerance grows with distance', () => {
  const { positions, holeOffsets, spheres } = twoHoles()
  // A ray looking east passes hole 0 at 0.3 m and hole 1 at 0.3 m as well (same line):
  // equal closeness, so the nearer one along the ray is kept.
  const along = pickPiece(ray([-500, 0.3, -15], [1, 0, 0]), positions, holeOffsets, spheres)
  expect(along?.piece).toBe(1)
  // Slightly oblique: closer to hole 1 (0.1 m) than to hole 0 (0.7 m).
  const oblique = pickPiece(ray([-500, 4.3, -15], [1, -0.006, 0]), positions, holeOffsets, spheres)
  expect(oblique?.piece).toBe(11)

  // Perspective: 6 px at 0.002 m per pixel per metre -> 6 m radius at 500 m.
  const far = ray([4, -500, -30], [0, 1, 0], 0, 0.012)
  expect(pickPiece(far, positions, holeOffsets, spheres)?.piece).toBe(2)
  const near = ray([4, -50, -30], [0, 1, 0], 0, 0.012)
  expect(pickPiece(near, positions, holeOffsets, spheres)).toBeNull()
})

test('pick works in render space under vertical exaggeration', () => {
  const { positions, holeOffsets, spheres } = twoHoles()
  // With z stretched 3x, scene z -34 is drawn at -102.
  const hit = pickPiece(ray([0, -500, -102], [0, 1, 0]), positions, holeOffsets, spheres, 3)
  expect(hit?.piece).toBe(3)
  expect(hit?.s).toBeCloseTo(0.4, 6)
  expect(pickPiece(ray([0, -500, -290], [0, 1, 0]), positions, holeOffsets, spheres, 3)?.piece).toBe(9)
  expect(pickPiece(ray([0, -500, -102], [0, 1, 0]), positions, holeOffsets, spheres, 1)).toBeNull()
})

test('pick index -> interval lookup on the fixture, and speed on 100 000 pieces', () => {
  const set = fixture()
  const seg = set.segments
  if (!seg) throw new Error('no segments')
  const pieces = buildPieces(set)
  const origin = [500150, 5300200, 1200] as const
  const scene = new Float32Array(pieces.positions.length)
  for (let i = 0; i < scene.length; i++) scene[i] = pieces.positions[i] - origin[i % 3]
  const spheres = holeSpheres(scene, pieces.holeOffsets)
  // Aim at the middle of T3's only interval (5-30 m), looking north.
  const mid = [0, 1, 2].map((k) => (seg.start[6 * 3 + k] + seg.end[6 * 3 + k]) / 2 - origin[k])
  const hit = pickPiece(ray([mid[0], mid[1] - 300, mid[2]], [0, 1, 0], 0.5), scene, pieces.holeOffsets, spheres)
  if (!hit) throw new Error('nothing picked')
  expect(pieces.interval[hit.piece]).toBe(6)
  expect(set.holeIds[pieces.hole[hit.piece]]).toBe('T3')
  expect(seg.from[pieces.interval[hit.piece]]).toBe(5)

  // 500 holes x 200 intervals: a pick must cost far less than a frame.
  const perf = buildPerfDrillholes(500, 200)
  const big = buildPieces(perf.set)
  expect(big.count).toBe(100_000)
  const o = [perf.set.collarXyz[0], perf.set.collarXyz[1], perf.set.collarXyz[2]]
  const bigScene = new Float32Array(big.positions.length)
  for (let i = 0; i < bigScene.length; i++) bigScene[i] = big.positions[i] - o[i % 3]
  const bigSpheres = holeSpheres(bigScene, big.holeOffsets)
  const p = 37_123
  const target = [0, 1, 2].map((k) => (bigScene[p * 6 + k] + bigScene[p * 6 + 3 + k]) / 2)
  const aim = ray([target[0] + 300, target[1] - 900, target[2] + 400], [-300, 900, -400], 0.4)
  expect(pickPiece(aim, bigScene, big.holeOffsets, bigSpheres)?.piece).toBe(p)
  const start = performance.now()
  for (let i = 0; i < 200; i++) pickPiece(aim, bigScene, big.holeOffsets, bigSpheres)
  expect((performance.now() - start) / 200).toBeLessThan(4)
})

// ---------------------------------------------------------------------- labels

test('declutter keeps non-overlapping labels in priority order', () => {
  const options = { width: 400, height: 300, labelWidth: 40, labelHeight: 14, max: 100 }
  // 0 and 1 overlap; 2 is clear; 3 is off screen; 4 is hidden (NaN); 5 overlaps 2.
  const xy = [100, 100, 120, 105, 300, 100, 500, 100, Number.NaN, Number.NaN, 310, 110]
  expect(declutter(xy, options)).toEqual([0, 2])
  // With another priority the other one of each pair wins.
  expect(declutter(xy, options, [5, 1, 0, 2, 3, 4])).toEqual([5, 1])
  // Touching boxes (exactly one label apart) do not collide.
  expect(declutter([0, 0, 40, 0, 0, 14], options)).toEqual([0, 1, 2])
  expect(declutter(xy, { ...options, max: 1 })).toEqual([0])
  expect(declutter(xy, { ...options, max: 0 })).toEqual([])
})

test('declutter: zooming out drops labels, zooming in shows them all', () => {
  const options = { width: 800, height: 600, labelWidth: 44, labelHeight: 15, max: 150 }
  const grid = (spacing: number) => {
    const xy: number[] = []
    for (let i = 0; i < 20; i++) for (let j = 0; j < 20; j++) xy.push(100 + i * spacing, 100 + j * spacing)
    return xy
  }
  const zoomedOut = declutter(grid(6), options)
  const medium = declutter(grid(25), options)
  const zoomedIn = declutter(grid(50), { ...options, width: 2000, height: 2000, max: 1000 })
  expect(zoomedOut.length).toBeLessThan(40)
  expect(medium.length).toBeGreaterThan(zoomedOut.length)
  expect(zoomedIn.length).toBe(400)
  // No two kept boxes overlap.
  const xy = grid(6)
  for (const a of zoomedOut) {
    for (const b of zoomedOut) {
      if (a === b) continue
      const apart = Math.abs(xy[a * 2] - xy[b * 2]) >= 44 || Math.abs(xy[a * 2 + 1] - xy[b * 2 + 1]) >= 15
      expect(apart).toBe(true)
    }
  }
  expect(declutter(grid(50), { ...options, width: 2000, height: 2000 }).length).toBe(150)
})

/**
 * Client-generated data for the performance probe (`?mock=perf`): many drillholes with
 * many intervals, in exactly the structures the API client produces, so the real layers
 * draw them. Deterministic.
 */
import type { AttributeInfo, DrillholeSet, MeshData } from '../api/client'
import { MOCK_CENTRE } from './mockData'

function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface PerfDrillholes {
  set: DrillholeSet
  attributes: AttributeInfo[]
}

/**
 * `holes` inclined, gently curving holes on a 40 m grid, each with `perHole` one-metre
 * intervals carrying a grade (`au`, 3 % missing) and a rock code.
 */
export function buildPerfDrillholes(holes = 500, perHole = 200, seed = 7): PerfDrillholes {
  const random = rng(seed)
  const columns = Math.ceil(Math.sqrt(holes * 1.25))
  const stations = Math.ceil(perHole / 10) + 1
  const segments = holes * perHole
  const rocks = ['ORE', 'WASTE', 'DYKE']

  const holeIds: string[] = []
  const holeOffsets = new Uint32Array(holes + 1)
  const pointDepth = new Float64Array(holes * stations)
  const pointXyz = new Float64Array(holes * stations * 3)
  const collarXyz = new Float64Array(holes * 3)
  const maxDepth = new Float64Array(holes).fill(perHole)
  const hole = new Uint32Array(segments)
  const row = new Uint32Array(segments)
  const from = new Float64Array(segments)
  const to = new Float64Array(segments)
  const start = new Float64Array(segments * 3)
  const end = new Float64Array(segments * 3)
  const au = new Float64Array(segments)
  const rock = new Int32Array(segments)
  let auMin = Infinity
  let auMax = -Infinity
  let missing = 0

  const path = new Float64Array((perHole + 1) * 3)
  for (let h = 0; h < holes; h++) {
    holeIds.push(`P${String(h + 1).padStart(4, '0')}`)
    const x0 = MOCK_CENTRE.e + ((h % columns) - columns / 2) * 40 + (random() - 0.5) * 6
    const y0 = MOCK_CENTRE.n + (Math.floor(h / columns) - holes / columns / 2) * 40 + (random() - 0.5) * 6
    const z0 = MOCK_CENTRE.z + 60 + Math.sin(x0 / 180) * 12 + Math.cos(y0 / 140) * 9
    collarXyz.set([x0, y0, z0], h * 3)
    const azimuth = 1.2 + (random() - 0.5) * 0.3
    let dip = 1.05 + (random() - 0.5) * 0.2
    const lift = 0.0006 + random() * 0.0008
    // Position at every metre: the intervals and the polyline share these points.
    path.set([x0, y0, z0], 0)
    for (let d = 1; d <= perHole; d++) {
      dip = Math.max(0.5, dip - lift)
      path[d * 3] = path[d * 3 - 3] + Math.sin(azimuth) * Math.cos(dip)
      path[d * 3 + 1] = path[d * 3 - 2] + Math.cos(azimuth) * Math.cos(dip)
      path[d * 3 + 2] = path[d * 3 - 1] - Math.sin(dip)
    }
    holeOffsets[h] = h * stations
    for (let s = 0; s < stations; s++) {
      const d = Math.min(perHole, s * 10)
      pointDepth[h * stations + s] = d
      pointXyz.set(path.subarray(d * 3, d * 3 + 3), (h * stations + s) * 3)
    }
    const oreTop = 60 + random() * 60
    const oreBottom = oreTop + 15 + random() * 40
    for (let d = 0; d < perHole; d++) {
      const s = h * perHole + d
      hole[s] = h
      row[s] = s
      from[s] = d
      to[s] = d + 1
      start.set(path.subarray(d * 3, d * 3 + 3), s * 3)
      end.set(path.subarray(d * 3 + 3, d * 3 + 6), s * 3)
      const inOre = d >= oreTop && d < oreBottom
      rock[s] = inOre ? 0 : random() < 0.04 ? 2 : 1
      if (random() < 0.03) {
        au[s] = Number.NaN
        missing++
      } else {
        au[s] = Math.round((inOre ? 0.4 + random() * random() * 6 : random() * 0.15) * 1000) / 1000
        if (au[s] < auMin) auMin = au[s]
        if (au[s] > auMax) auMax = au[s]
      }
    }
  }
  holeOffsets[holes] = holes * stations

  return {
    set: {
      objectId: 'perf-holes',
      holeIds,
      holeOffsets,
      pointDepth,
      pointXyz,
      collarXyz,
      maxDepth,
      segments: {
        table: 'assay',
        count: segments,
        hole,
        row,
        from,
        to,
        start,
        end,
        attributes: [
          { name: 'au', type: 'numeric', unit: 'g/t', values: au },
          { name: 'rock', type: 'categorical', categories: rocks, codes: rock },
        ],
        dropped: 0,
      },
      byteLength: 0,
    },
    attributes: [
      {
        table: 'assay',
        name: 'au',
        type: 'numeric',
        unit: 'g/t',
        min: auMin,
        max: auMax,
        display_min: 0,
        display_max: 3,
        count: segments - missing,
        missing,
      },
      { table: 'assay', name: 'rock', type: 'categorical', categories: rocks, count: segments, missing: 0 },
    ],
  }
}

/**
 * An undulating surface of about `triangles` triangles (a square grid, two triangles
 * per cell) under the generated holes. No normals: the layer computes them.
 */
export function buildPerfMesh(triangles = 1_000_000): MeshData {
  const cells = Math.max(1, Math.round(Math.sqrt(triangles / 2)))
  const side = cells + 1
  const size = 1100
  const vertices = new Float64Array(side * side * 3)
  for (let j = 0; j < side; j++) {
    for (let i = 0; i < side; i++) {
      const x = (i / cells - 0.5) * size
      const y = (j / cells - 0.5) * size
      const v = (j * side + i) * 3
      vertices[v] = MOCK_CENTRE.e + x
      vertices[v + 1] = MOCK_CENTRE.n + y
      vertices[v + 2] = MOCK_CENTRE.z - 40 + Math.sin(x / 130) * 28 + Math.cos(y / 95) * 22 + Math.sin((x + y) / 37) * 4
    }
  }
  const faces = new Uint32Array(cells * cells * 6)
  let f = 0
  for (let j = 0; j < cells; j++) {
    for (let i = 0; i < cells; i++) {
      const a = j * side + i
      const b = a + 1
      const c = a + side
      const d = c + 1
      faces[f++] = a
      faces[f++] = b
      faces[f++] = d
      faces[f++] = a
      faces[f++] = d
      faces[f++] = c
    }
  }
  return { objectId: 'perf-mesh', vertices, faces, normals: null, closed: false, byteLength: 0 }
}

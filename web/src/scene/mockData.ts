/**
 * Deterministic stand-in data at mine coordinates, for `?mock=1`. World float64
 * throughout, exactly like the data real layers will receive from the API.
 */
import { boundsOf, type Bounds } from '../state/store'

export const MOCK_CENTRE = { e: 500_000, n: 5_300_000, z: 1_200 } as const

export interface MockBox {
  centre: [number, number, number]
  size: [number, number, number]
  color: [number, number, number]
}

export interface MockData {
  boxes: MockBox[]
  /** Triangle soup, world xyz (float64), 36 vertices per box. */
  boxPositions: Float64Array
  boxNormals: Float32Array
  boxColors: Float32Array
  boxBounds: Bounds
  /** Line segments, world xyz (float64), 2 vertices per segment. */
  linePositions: Float64Array
  lineColors: Float32Array
  lineBounds: Bounds
}

/** Small seeded generator (mulberry32) so screenshots are reproducible. */
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

// Muted categorical colours that read on both the dark and the light background.
const PALETTE: [number, number, number][] = [
  [0.82, 0.45, 0.2],
  [0.25, 0.58, 0.72],
  [0.55, 0.66, 0.27],
  [0.72, 0.36, 0.52],
  [0.86, 0.68, 0.24],
  [0.42, 0.45, 0.78],
]

// Unit cube faces: outward normal and four corners (counter-clockwise seen from outside).
const FACES: { n: [number, number, number]; c: [number, number, number][] }[] = [
  { n: [1, 0, 0], c: [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]] },
  { n: [-1, 0, 0], c: [[-1, 1, -1], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1]] },
  { n: [0, 1, 0], c: [[1, 1, -1], [-1, 1, -1], [-1, 1, 1], [1, 1, 1]] },
  { n: [0, -1, 0], c: [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]] },
  { n: [0, 0, 1], c: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]] },
  { n: [0, 0, -1], c: [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]] },
]

export function buildMockData(seed = 20261119, boxCount = 36, traceCount = 12): MockData {
  const random = rng(seed)
  const between = (lo: number, hi: number) => lo + (hi - lo) * random()
  const { e, n, z } = MOCK_CENTRE

  const boxes: MockBox[] = []
  for (let i = 0; i < boxCount; i++) {
    boxes.push({
      centre: [e + between(-320, 320), n + between(-260, 260), z + between(-90, 60)],
      size: [between(12, 48), between(12, 48), between(6, 30)],
      color: PALETTE[i % PALETTE.length],
    })
  }

  const boxPositions = new Float64Array(boxCount * 36 * 3)
  const boxNormals = new Float32Array(boxCount * 36 * 3)
  const boxColors = new Float32Array(boxCount * 36 * 3)
  let v = 0
  for (const box of boxes) {
    for (const face of FACES) {
      for (const corner of [0, 1, 2, 0, 2, 3]) {
        const c = face.c[corner]
        for (let k = 0; k < 3; k++) {
          boxPositions[v + k] = box.centre[k] + (c[k] * box.size[k]) / 2
          boxNormals[v + k] = face.n[k]
          boxColors[v + k] = box.color[k]
        }
        v += 3
      }
    }
  }

  // Drillhole-like traces: collar near surface, steep, drifting a little with depth.
  const segments: number[] = []
  const segmentColors: number[] = []
  for (let t = 0; t < traceCount; t++) {
    let p: [number, number, number] = [e + between(-300, 300), n + between(-240, 240), z + between(70, 95)]
    const azimuth = between(0, Math.PI * 2)
    let dip = between(0.9, 1.45)
    const steps = 10 + Math.floor(between(0, 8))
    const color = PALETTE[(t + 3) % PALETTE.length]
    for (let s = 0; s < steps; s++) {
      const length = 20
      const q: [number, number, number] = [
        p[0] + Math.sin(azimuth) * Math.cos(dip) * length,
        p[1] + Math.cos(azimuth) * Math.cos(dip) * length,
        p[2] - Math.sin(dip) * length,
      ]
      segments.push(...p, ...q)
      // Alternate shade per interval, as assay intervals would.
      const shade = s % 2 === 0 ? 1 : 0.62
      segmentColors.push(...color.map((c) => c * shade), ...color.map((c) => c * shade))
      p = q
      dip = Math.max(0.6, dip - 0.012)
    }
  }
  const linePositions = Float64Array.from(segments)

  return {
    boxes,
    boxPositions,
    boxNormals,
    boxColors,
    boxBounds: boundsOf(boxPositions),
    linePositions,
    lineColors: Float32Array.from(segmentColors),
    lineBounds: boundsOf(linePositions),
  }
}

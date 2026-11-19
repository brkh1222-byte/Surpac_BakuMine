/**
 * Turns a DrillholeSet into drawable "pieces": straight line segments that together
 * cover every hole from collar to end of hole exactly once per interval.
 *
 * An interval is drawn along the densified path, not as one chord: it is cut at every
 * polyline point strictly inside it, so a 100 m geology interval on a curved hole bends
 * with the hole. Its two ends are the exact `seg_start` / `seg_end` of the payload.
 * Stretches of a hole that no interval covers become gap pieces (interval -1), drawn in
 * the "no value" colour. Without an interval table the whole path is gap pieces.
 *
 * `interval[piece]` is the index -> interval lookup used for colouring and picking:
 * the scene draws one merged buffer, and a picked piece index leads back to its
 * interval (`DrillholeSet.segments` row) and hole.
 */
import type { DrillholeSet } from '../api/client'

export interface DrillholePieces {
  count: number
  /** World xyz of both ends, 6 numbers per piece. */
  positions: Float64Array
  /** Segment index in `DrillholeSet.segments`, or -1 for a stretch without interval. */
  interval: Int32Array
  /** Hole index per piece. */
  hole: Uint32Array
  /** Downhole depth of both ends, 2 numbers per piece. */
  depth: Float64Array
  /** Pieces of hole h are holeOffsets[h] .. holeOffsets[h + 1] - 1. */
  holeOffsets: Uint32Array
}

/** Depths closer than this are the same depth (metres). */
const EPS = 1e-6

export function buildPieces(set: DrillholeSet): DrillholePieces {
  const holes = set.holeIds.length
  const seg = set.segments
  const { pointDepth, pointXyz, holeOffsets } = set

  // Intervals of each hole, ordered by depth_from (counting sort by hole, then sort).
  const perHole: number[][] = Array.from({ length: holes }, () => [])
  if (seg) {
    for (let s = 0; s < seg.count; s++) {
      const h = seg.hole[s]
      if (h < holes) perHole[h].push(s)
    }
    for (const list of perHole) list.sort((a, b) => seg.from[a] - seg.from[b] || a - b)
  }

  const positions: number[] = []
  const interval: number[] = []
  const hole: number[] = []
  const depth: number[] = []
  const pieceOffsets = new Uint32Array(holes + 1)

  for (let h = 0; h < holes; h++) {
    pieceOffsets[h] = interval.length
    const first = holeOffsets[h]
    const end = holeOffsets[h + 1]
    if (end - first < 1) continue
    let scan = first // first polyline point that may lie inside the next range

    /** Emits the path from depth d0 (at xyz a) to d1 (at xyz b) as pieces. */
    const emit = (d0: number, ax: number, ay: number, az: number, d1: number, bx: number, by: number, bz: number, s: number) => {
      if (!(d1 > d0)) return
      while (scan < end && pointDepth[scan] <= d0 + EPS) scan++
      // Overlapping intervals start before the scan position: step back.
      while (scan > first && pointDepth[scan - 1] > d0 + EPS) scan--
      let px = ax
      let py = ay
      let pz = az
      let pd = d0
      for (let p = scan; p < end && pointDepth[p] < d1 - EPS; p++) {
        const q = p * 3
        positions.push(px, py, pz, pointXyz[q], pointXyz[q + 1], pointXyz[q + 2])
        interval.push(s)
        hole.push(h)
        depth.push(pd, pointDepth[p])
        px = pointXyz[q]
        py = pointXyz[q + 1]
        pz = pointXyz[q + 2]
        pd = pointDepth[p]
      }
      positions.push(px, py, pz, bx, by, bz)
      interval.push(s)
      hole.push(h)
      depth.push(pd, d1)
    }

    // The covered depth so far and the position there.
    let covered = pointDepth[first]
    let cx = pointXyz[first * 3]
    let cy = pointXyz[first * 3 + 1]
    let cz = pointXyz[first * 3 + 2]
    if (seg) {
      for (const s of perHole[h]) {
        const s3 = s * 3
        if (seg.from[s] > covered + EPS) {
          emit(covered, cx, cy, cz, seg.from[s], seg.start[s3], seg.start[s3 + 1], seg.start[s3 + 2], -1)
        }
        emit(seg.from[s], seg.start[s3], seg.start[s3 + 1], seg.start[s3 + 2], seg.to[s], seg.end[s3], seg.end[s3 + 1], seg.end[s3 + 2], s)
        if (seg.to[s] > covered) {
          covered = seg.to[s]
          cx = seg.end[s3]
          cy = seg.end[s3 + 1]
          cz = seg.end[s3 + 2]
        }
      }
    }
    const last = end - 1
    if (pointDepth[last] > covered + EPS) {
      emit(covered, cx, cy, cz, pointDepth[last], pointXyz[last * 3], pointXyz[last * 3 + 1], pointXyz[last * 3 + 2], -1)
    }
  }
  pieceOffsets[holes] = interval.length

  return {
    count: interval.length,
    positions: Float64Array.from(positions),
    interval: Int32Array.from(interval),
    hole: Uint32Array.from(hole),
    depth: Float64Array.from(depth),
    holeOffsets: pieceOffsets,
  }
}

/**
 * Snapping and picking on the section plane, for section digitising. Pure functions.
 *
 * Snap targets come from the drillholes within the section slab:
 *   - `crossing`: where a hole trace passes through the section plane;
 *   - `boundary`: an interval boundary (geology or assay contact, the start or end of an
 *     interval of the table the holes are coloured by) inside the slab, projected
 *     square onto the plane.
 * Every target is ON the plane, so a snapped vertex stays on its section.
 *
 * Distances are measured in render space (z multiplied by the vertical exaggeration),
 * which is what the user sees, and compared with a tolerance in metres derived from
 * pixels by the caller.
 */
import { frameOf, signedDistance, toPlane, type SectionPlane, type Slab } from '../section/plane'
import type { Vec3 } from '../scene/origin'
import type { DigitString, VertexRef } from './edit'

export type SnapKind = 'crossing' | 'boundary'

export interface SnapTarget {
  kind: SnapKind
  /** World point on the section plane. */
  point: Vec3
  /** Drillhole object id and hole index. */
  object: string
  hole: number
  /** Depth down the hole of the original (unprojected) point. */
  depth: number
}

/** What snapping needs of a drillhole layer (a subset of drillholes/pieces.ts). */
export interface PieceSource {
  count: number
  /** World xyz of both ends, 6 numbers per piece. */
  positions: ArrayLike<number>
  /** Interval index per piece, -1 where no interval covers it. */
  interval: ArrayLike<number>
  hole: ArrayLike<number>
  /** Depth of both ends, 2 numbers per piece. */
  depth: ArrayLike<number>
  holeOffsets: ArrayLike<number>
}

/** Section id: orientation and position, e.g. "az090.0-dip90.0-at-5300500.00". */
export function sectionId(plane: SectionPlane): string {
  const { n } = frameOf(plane)
  const travel = -(plane.centre[0] * n[0] + plane.centre[1] * n[1] + plane.centre[2] * n[2])
  const az = plane.azimuth.toFixed(1).padStart(5, '0')
  return `az${az}-dip${plane.dip.toFixed(1)}-at${travel >= 0 ? '+' : ''}${travel.toFixed(2)}`
}

function project(plane: SectionPlane, n: Vec3, p: Vec3): Vec3 {
  const d = signedDistance(plane, p, n)
  return [p[0] - d * n[0], p[1] - d * n[1], p[2] - d * n[2]]
}

/** Snap targets of one drillhole layer for a section and its slab. */
export function snapTargets(plane: SectionPlane, slab: Slab, pieces: PieceSource, object: string): SnapTarget[] {
  const { n } = frameOf(plane)
  const out: SnapTarget[] = []
  const pos = pieces.positions
  const holes = pieces.holeOffsets.length - 1
  for (let h = 0; h < holes; h++) {
    const first = pieces.holeOffsets[h]
    const end = pieces.holeOffsets[h + 1]
    for (let k = first; k < end; k++) {
      const i = k * 6
      const a: Vec3 = [pos[i], pos[i + 1], pos[i + 2]]
      const b: Vec3 = [pos[i + 3], pos[i + 4], pos[i + 5]]
      const da = signedDistance(plane, a, n)
      const db = signedDistance(plane, b, n)
      // Trace crossing the plane (a piece ending exactly on it counts once, at its end).
      if ((da < 0 && db >= 0) || (da > 0 && db <= 0)) {
        const t = da / (da - db)
        const p: Vec3 = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
        const depth = pieces.depth[k * 2] + (pieces.depth[k * 2 + 1] - pieces.depth[k * 2]) * t
        out.push({ kind: 'crossing', point: project(plane, n, p), object, hole: h, depth })
      }
      // Interval boundary at the start of this piece, and at the end of the hole.
      const before = k === first ? -1 : pieces.interval[k - 1]
      const here = pieces.interval[k]
      if (here !== before && (here !== -1 || before !== -1) && da <= slab.front && da >= -slab.back) {
        out.push({ kind: 'boundary', point: project(plane, n, a), object, hole: h, depth: pieces.depth[k * 2] })
      }
      if (k === end - 1 && here !== -1 && db <= slab.front && db >= -slab.back) {
        out.push({ kind: 'boundary', point: project(plane, n, b), object, hole: h, depth: pieces.depth[k * 2 + 1] })
      }
    }
  }
  return out
}

/** Distance between two world points as seen on screen (z exaggerated). */
export function renderDistance(a: Vec3, b: Vec3, exaggeration: number): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], (a[2] - b[2]) * exaggeration)
}

/** The target nearest to `p` within `tolerance` (render-space metres), or null. */
export function nearestSnap(targets: readonly SnapTarget[], p: Vec3, tolerance: number, exaggeration: number): SnapTarget | null {
  let best: SnapTarget | null = null
  let bestDistance = tolerance
  for (const target of targets) {
    const d = renderDistance(target.point, p, exaggeration)
    if (d <= bestDistance) {
      best = target
      bestDistance = d
    }
  }
  return best
}

/** Whether every point of a string lies on the plane (within `tolerance` metres). */
export function onPlane(string: DigitString, plane: SectionPlane, tolerance = 0.01): boolean {
  const { n } = frameOf(plane)
  return string.points.length > 0 && string.points.every((p) => Math.abs(signedDistance(plane, p, n)) <= tolerance)
}

/** Whether every point of a string lies inside the slab. */
export function inSlabString(string: DigitString, plane: SectionPlane, slab: Slab): boolean {
  const frame = frameOf(plane)
  return string.points.every((p) => {
    const w = toPlane(plane, p, frame)[2]
    return w <= slab.front + 1e-9 && w >= -slab.back - 1e-9
  })
}

/** The vertex nearest to `p` within `tolerance` among `strings`, or null. */
export function hitVertex(strings: readonly DigitString[], p: Vec3, tolerance: number, exaggeration: number): VertexRef | null {
  let best: VertexRef | null = null
  let bestDistance = tolerance
  for (const s of strings) {
    s.points.forEach((q, index) => {
      const d = renderDistance(q, p, exaggeration)
      if (d <= bestDistance) {
        best = { string: s.id, index }
        bestDistance = d
      }
    })
  }
  return best
}

/**
 * The edge of `strings` nearest to `p` within `tolerance`: the vertex it starts at (the
 * new vertex goes after it). Used to insert a vertex on an existing edge.
 */
export function hitEdge(strings: readonly DigitString[], p: Vec3, tolerance: number, exaggeration: number): VertexRef | null {
  let best: VertexRef | null = null
  let bestDistance = tolerance
  const e = exaggeration
  for (const s of strings) {
    const m = s.points.length
    const edges = s.closed ? m : m - 1
    for (let i = 0; i < edges; i++) {
      const a = s.points[i]
      const b = s.points[(i + 1) % m]
      const ab = [b[0] - a[0], b[1] - a[1], (b[2] - a[2]) * e]
      const ap = [p[0] - a[0], p[1] - a[1], (p[2] - a[2]) * e]
      const len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
      if (len2 === 0) continue
      const t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2
      if (t <= 0 || t >= 1) continue
      const d = Math.hypot(ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2])
      if (d <= bestDistance) {
        best = { string: s.id, index: i }
        bestDistance = d
      }
    }
  }
  return best
}

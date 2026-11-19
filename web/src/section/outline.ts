/**
 * Mesh-plane intersection on typed arrays: the polylines where a plane cuts a triangle
 * mesh. A port of `bakumine.core.geometry.section.section` (same conventions, checked
 * against polylines produced by the Python code in outline.test.ts):
 *
 *  - every vertex is *positive* (d > 0) or not; vertices within `tol` of the plane have
 *    d = 0 exactly;
 *  - an edge whose ends differ in class has ONE crossing point, computed once per edge,
 *    so neighbouring faces share it exactly and loops close without gaps;
 *  - a face with one or two positive corners contributes one segment, directed so that
 *    the solid is on the left of the loop in the plane frame (outer loops run
 *    counter-clockwise seen by the viewer, i.e. against the normal).
 *
 * Runs in a worker (outline.worker.ts) for the viewer; pure and synchronous here.
 * World float64 in, world float64 out. One pass over the faces: a 1M-triangle mesh
 * takes ~10 ms.
 */
import type { Vec3 } from '../scene/origin'

export interface Outline {
  /** World xyz of all polylines, one after the other. */
  points: Float64Array
  /** Polyline k is points `offsets[k] .. offsets[k + 1] - 1` (point indices). */
  offsets: Uint32Array
  /** 1 = closed loop (the first point is not repeated), 0 = open polyline. */
  closed: Uint8Array
}

export const EMPTY_OUTLINE: Outline = { points: new Float64Array(0), offsets: new Uint32Array(1), closed: new Uint8Array(0) }

/** Number of polylines. */
export function outlineCount(outline: Outline): number {
  return outline.closed.length
}

/** Points of polyline `k` as a view (flat xyz). */
export function outlinePoints(outline: Outline, k: number): Float64Array {
  return outline.points.subarray(outline.offsets[k] * 3, outline.offsets[k + 1] * 3)
}

/** Joins directed segments (node pairs) into node paths; see `_chain` in section.py. */
function chain(start: Int32Array, end: Int32Array, nodes: number): number[][] {
  const segments = start.length
  // Incident segments per node (CSR), in segment order.
  const degree = new Uint32Array(nodes + 1)
  for (let s = 0; s < segments; s++) {
    degree[start[s] + 1]++
    degree[end[s] + 1]++
  }
  for (let i = 0; i < nodes; i++) degree[i + 1] += degree[i]
  const fill = degree.slice(0, nodes)
  const incident = new Int32Array(segments * 2)
  for (let s = 0; s < segments; s++) {
    incident[fill[start[s]]++] = s
    incident[fill[end[s]]++] = s
  }
  const used = new Uint8Array(segments)

  const walk = (first: number, firstSegment: number): number[] => {
    const path = [first]
    let node = first
    let s = firstSegment
    let forward = 0
    let steps = 0
    for (;;) {
      used[s] = 1
      steps++
      if (start[s] === node) {
        forward++
        node = end[s]
      } else {
        node = start[s]
      }
      path.push(node)
      let next = -1
      for (let k = degree[node]; k < degree[node + 1]; k++) {
        const c = incident[k]
        if (used[c]) continue
        if (start[c] === node) {
          next = c
          break
        }
        if (next < 0) next = c
      }
      if (next < 0) break
      s = next
    }
    if (2 * forward < steps) path.reverse()
    return path
  }

  const paths: number[][] = []
  // Open polylines first: they start at nodes with an odd number of segments.
  for (let node = 0; node < nodes; node++) {
    if ((degree[node + 1] - degree[node]) & 1) {
      for (let k = degree[node]; k < degree[node + 1]; k++) {
        if (!used[incident[k]]) {
          paths.push(walk(node, incident[k]))
          break
        }
      }
    }
  }
  for (let s = 0; s < segments; s++) if (!used[s]) paths.push(walk(start[s], s))
  return paths
}

type Point = [number, number, number]

function same(a: Point, b: Point): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2]
}

/** Drops repeated points and zero-area spikes (a, b, a); see `_clean` in section.py. */
function clean(points: Point[], closed: boolean): Point[] {
  let pts = points
  for (;;) {
    let out: Point[] = []
    for (const p of pts) {
      if (out.length > 0 && same(out[out.length - 1], p)) continue
      if (out.length >= 2 && same(out[out.length - 2], p)) {
        out.pop()
        continue
      }
      out.push(p)
    }
    if (closed) {
      let changed = true
      while (changed && out.length >= 2) {
        changed = false
        if (same(out[0], out[out.length - 1])) {
          out.pop()
          changed = true
        } else if (out.length >= 3 && same(out[1], out[out.length - 1])) {
          out = out.slice(1, -1)
          changed = true
        } else if (out.length >= 3 && same(out[0], out[out.length - 2])) {
          out = out.slice(0, -2)
          changed = true
        }
      }
    }
    if (out.length === pts.length) return out
    pts = out
  }
}

/** Canonical text of a closed loop: rotated to start at its smallest point. */
function loopKeys(points: Point[]): { forward: string; reverse: string } {
  let first = 0
  for (let i = 1; i < points.length; i++) {
    const a = points[i]
    const b = points[first]
    if (a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])))) first = i
  }
  const rotated = [...points.slice(first), ...points.slice(0, first)]
  const reversed = [rotated[0], ...rotated.slice(1).reverse()]
  return { forward: rotated.map((p) => p.join(',')).join(';'), reverse: reversed.map((p) => p.join(',')).join(';') }
}

/**
 * Cuts the mesh (`vertices`: flat world xyz; `faces`: 3 vertex indices per triangle;
 * shared vertices must be shared by index) with the plane through `origin` with normal
 * `normal` (any length). `tol`: vertices closer than this to the plane count as on it;
 * default 1e-12 times the diagonal of the mesh bounding box, as in the Python code.
 */
export function cutMesh(
  vertices: ArrayLike<number>,
  faces: ArrayLike<number>,
  origin: Vec3,
  normal: Vec3,
  tol?: number,
): Outline {
  const vertexCount = Math.floor(vertices.length / 3)
  const faceCount = Math.floor(faces.length / 3)
  const length = Math.hypot(normal[0], normal[1], normal[2])
  if (faceCount === 0 || vertexCount === 0 || !(length > 0)) return EMPTY_OUTLINE
  const nx = normal[0] / length
  const ny = normal[1] / length
  const nz = normal[2] / length
  const [ox, oy, oz] = origin

  // Signed distances and the bounding box (for the default tolerance) in one pass.
  const d = new Float64Array(vertexCount)
  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
  for (let i = 0, k = 0; i < vertexCount; i++, k += 3) {
    const x = vertices[k]
    const y = vertices[k + 1]
    const z = vertices[k + 2]
    d[i] = (x - ox) * nx + (y - oy) * ny + (z - oz) * nz
    if (tol === undefined) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (z < minZ) minZ = z
      if (z > maxZ) maxZ = z
    }
  }
  const eps = tol ?? 1e-12 * Math.hypot(maxX - minX, maxY - minY, maxZ - minZ)
  const positive = new Uint8Array(vertexCount)
  for (let i = 0; i < vertexCount; i++) {
    if (Math.abs(d[i]) <= eps) d[i] = 0
    else if (d[i] > 0) positive[i] = 1
  }

  // Crossing faces -> directed segments between edge nodes.
  const nodeOfEdge = new Map<number, number>()
  const nodePoints: number[] = []
  const node = (a: number, b: number): number => {
    const key = a < b ? a * vertexCount + b : b * vertexCount + a
    let id = nodeOfEdge.get(key)
    if (id === undefined) {
      id = nodeOfEdge.size
      nodeOfEdge.set(key, id)
      const neg = positive[a] ? b : a
      const pos = positive[a] ? a : b
      // 0 <= t < 1; exactly 0 when the non-positive vertex is on the plane.
      const t = d[neg] / (d[neg] - d[pos])
      const n3 = neg * 3
      const p3 = pos * 3
      nodePoints.push(
        vertices[n3] + t * (vertices[p3] - vertices[n3]),
        vertices[n3 + 1] + t * (vertices[p3 + 1] - vertices[n3 + 1]),
        vertices[n3 + 2] + t * (vertices[p3 + 2] - vertices[n3 + 2]),
      )
    }
    return id
  }
  const starts: number[] = []
  const ends: number[] = []
  for (let f = 0, k = 0; f < faceCount; f++, k += 3) {
    const a = faces[k]
    const b = faces[k + 1]
    const c = faces[k + 2]
    const count = positive[a] + positive[b] + positive[c]
    if (count === 0 || count === 3) continue
    const lonePositive = count === 1
    // The corner that is alone on its side, the next one and the previous one.
    let lone: number, next: number, previous: number
    if (positive[a] === (lonePositive ? 1 : 0) && positive[b] !== positive[a] && positive[c] !== positive[a]) {
      lone = a
      next = b
      previous = c
    } else if (positive[b] === (lonePositive ? 1 : 0) && positive[a] !== positive[b] && positive[c] !== positive[b]) {
      lone = b
      next = c
      previous = a
    } else {
      lone = c
      next = a
      previous = b
    }
    const ab = node(lone, next)
    const ca = node(previous, lone)
    // Solid on the left seen against the normal.
    starts.push(lonePositive ? ab : ca)
    ends.push(lonePositive ? ca : ab)
  }
  if (starts.length === 0) return EMPTY_OUTLINE

  const paths = chain(Int32Array.from(starts), Int32Array.from(ends), nodeOfEdge.size)
  let polylines: Point[][] = []
  let closedFlags: boolean[] = []
  for (const path of paths) {
    const isClosed = path.length > 2 && path[0] === path[path.length - 1]
    const pts = clean(
      path.map((id): Point => [nodePoints[id * 3], nodePoints[id * 3 + 1], nodePoints[id * 3 + 2]]),
      isClosed,
    )
    if (pts.length < (isClosed ? 3 : 2)) continue
    polylines.push(pts)
    closedFlags.push(isClosed)
  }

  // Pairs of closed loops through the same points in opposite directions enclose
  // nothing (the mesh touches the plane along a closed curve of on-plane edges).
  if (closedFlags.filter(Boolean).length > 1) {
    const unmatched = new Map<string, number>()
    const drop = new Set<number>()
    polylines.forEach((pts, i) => {
      if (!closedFlags[i]) return
      const { forward, reverse } = loopKeys(pts)
      const partner = unmatched.get(reverse)
      if (partner === undefined) unmatched.set(forward, i)
      else {
        unmatched.delete(reverse)
        drop.add(i)
        drop.add(partner)
      }
    })
    if (drop.size > 0) {
      polylines = polylines.filter((_, i) => !drop.has(i))
      closedFlags = closedFlags.filter((_, i) => !drop.has(i))
    }
  }

  const total = polylines.reduce((sum, p) => sum + p.length, 0)
  const points = new Float64Array(total * 3)
  const offsets = new Uint32Array(polylines.length + 1)
  let at = 0
  polylines.forEach((pts, k) => {
    offsets[k] = at
    for (const p of pts) {
      points[at * 3] = p[0]
      points[at * 3 + 1] = p[1]
      points[at * 3 + 2] = p[2]
      at++
    }
  })
  offsets[polylines.length] = at
  return { points, offsets, closed: Uint8Array.from(closedFlags, (c) => (c ? 1 : 0)) }
}

/**
 * The outline as line segments in SCENE space (float32, origin subtracted): 6 numbers
 * per segment, closed loops closed. This is what the fat-line geometry takes.
 */
export function outlineSegments(outline: Outline, origin: Vec3): Float32Array {
  const count = outlineCount(outline)
  let segments = 0
  for (let k = 0; k < count; k++) {
    const n = outline.offsets[k + 1] - outline.offsets[k]
    segments += outline.closed[k] ? n : n - 1
  }
  const out = new Float32Array(segments * 6)
  const p = outline.points
  let at = 0
  for (let k = 0; k < count; k++) {
    const first = outline.offsets[k]
    const end = outline.offsets[k + 1]
    const n = end - first
    const pieces = outline.closed[k] ? n : n - 1
    for (let i = 0; i < pieces; i++) {
      const a = (first + i) * 3
      const b = (first + ((i + 1) % n)) * 3
      out[at++] = p[a] - origin[0]
      out[at++] = p[a + 1] - origin[1]
      out[at++] = p[a + 2] - origin[2]
      out[at++] = p[b] - origin[0]
      out[at++] = p[b + 1] - origin[1]
      out[at++] = p[b + 2] - origin[2]
    }
  }
  return out
}

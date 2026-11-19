/**
 * Section geometry: pure functions, world coordinates, float64, no three.js.
 *
 * A section is a plane with a slab around it:
 *
 *   centre   a world point on the plane
 *   azimuth  direction of the section line, degrees clockwise from north. The section
 *            line is the horizontal line in the plane; `u` points along it and is to the
 *            RIGHT of a viewer who looks at the section.
 *   dip      dip of the plane, degrees: 90 = vertical section, 0 = horizontal plan/bench.
 *
 * Plane frame (u, v, n), right-handed, u x v = n - the same frame as
 * `bakumine.core.geometry.section.plane_frame`:
 *
 *   u  horizontal, along the section line
 *   v  "up" in the plane (straight up for a vertical section, north for a plan with
 *      azimuth 90)
 *   n  normal, pointing TOWARDS the viewer. The viewer looks along -n.
 *
 * So a W-E section (azimuth 90) has u = east, v = up, n = south: the viewer looks north
 * with east on the right. "Forward" (step forward, slab `back` side) is away from the
 * viewer, along -n; the slab keeps the points with  -back <= (p - centre) . n <= front.
 */
import type { Vec3 } from '../scene/origin'

export interface SectionPlane {
  centre: Vec3
  /** Azimuth of the section line, degrees clockwise from north, [0, 360). */
  azimuth: number
  /** Dip of the plane, degrees, [0, 90]. */
  dip: number
}

export interface Slab {
  /** Thickness kept on the viewer's side of the plane, metres. */
  front: number
  /** Thickness kept behind the plane (away from the viewer), metres. */
  back: number
}

export interface Frame {
  u: Vec3
  v: Vec3
  n: Vec3
}

/** A plane as `normal . p + constant = 0`; a half-space keeps `normal . p + constant >= 0`. */
export interface PlaneEq {
  normal: [number, number, number]
  constant: number
}

const RAD = Math.PI / 180

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function normaliseAzimuth(azimuth: number): number {
  const a = azimuth % 360
  return a < 0 ? a + 360 : a
}

/** Sine and cosine with exact values at multiples of 90 degrees (no 6e-17 residue). */
function sincos(degrees: number): [number, number] {
  const d = normaliseAzimuth(degrees)
  if (d === 0) return [0, 1]
  if (d === 90) return [1, 0]
  if (d === 180) return [0, -1]
  if (d === 270) return [-1, 0]
  return [Math.sin(d * RAD), Math.cos(d * RAD)]
}

/** The plane frame of a section with the given line azimuth and dip (see file header). */
export function sectionFrame(azimuth: number, dip: number): Frame {
  const [sa, ca] = sincos(azimuth)
  const [st, ct] = sincos(90 - Math.min(90, Math.max(0, dip))) // tilt of the normal from horizontal
  const u: Vec3 = [sa, ca, 0]
  // `+ 0` turns a negative zero into zero.
  const n: Vec3 = [ct * ca + 0, -ct * sa + 0, st]
  // v = n x u
  const v: Vec3 = [n[1] * u[2] - n[2] * u[1] + 0, n[2] * u[0] - n[0] * u[2] + 0, n[0] * u[1] - n[1] * u[0] + 0]
  return { u, v, n }
}

export function frameOf(plane: SectionPlane): Frame {
  return sectionFrame(plane.azimuth, plane.dip)
}

/** Signed distance of a world point from the plane, positive towards the viewer. */
export function signedDistance(plane: SectionPlane, p: Vec3, n: Vec3 = frameOf(plane).n): number {
  return (p[0] - plane.centre[0]) * n[0] + (p[1] - plane.centre[1]) * n[1] + (p[2] - plane.centre[2]) * n[2]
}

export function inSlab(distance: number, slab: Slab, eps = 0): boolean {
  return distance <= slab.front + eps && distance >= -slab.back - eps
}

/** The plane moved `distance` metres forward (away from the viewer; negative = back). */
export function stepPlane(plane: SectionPlane, distance: number): SectionPlane {
  const { n } = frameOf(plane)
  return {
    ...plane,
    centre: [plane.centre[0] - n[0] * distance, plane.centre[1] - n[1] * distance, plane.centre[2] - n[2] * distance],
  }
}

/**
 * The two half-spaces of the slab in RENDER space (scene origin subtracted, z multiplied
 * by the vertical exaggeration), with unit normals, ready for three.js clipping planes:
 * a fragment at render position p is kept when `normal . p + constant >= 0` for both.
 *
 * World p_w = (p.x + o.x, p.y + o.y, p.z / ve + o.z), so the world half-space
 * `n . (p_w - c) + k >= 0` has the render-space normal (n.x, n.y, n.z / ve).
 */
export function renderClipPlanes(
  plane: SectionPlane,
  slab: Slab,
  exaggeration: number,
  origin: Vec3,
): [PlaneEq, PlaneEq] {
  const { n } = frameOf(plane)
  const nx = n[0]
  const ny = n[1]
  const nz = n[2] / exaggeration
  const length = Math.hypot(nx, ny, nz)
  // n . (o - c): the world distance of the scene origin from the plane.
  const offset = signedDistance(plane, origin, n)
  return [
    // d <= front
    { normal: [-nx / length + 0, -ny / length + 0, -nz / length + 0], constant: (slab.front - offset) / length },
    // d >= -back
    { normal: [nx / length, ny / length, nz / length], constant: (slab.back + offset) / length },
  ]
}

/** The section plane itself in render space (unit normal towards the viewer). */
export function renderPlane(plane: SectionPlane, exaggeration: number, origin: Vec3): PlaneEq {
  return renderClipPlanes(plane, { front: 0, back: 0 }, exaggeration, origin)[1]
}

/**
 * Vertical section through two world points (the ends of a line drawn in plan): the line
 * runs from `a` to `b`, so the viewer has `a` on the left. The centre is the midpoint, at
 * elevation `z`. Returns null when the points coincide.
 */
export function sectionFromTwoPoints(a: Vec3, b: Vec3, z: number = (a[2] + b[2]) / 2): SectionPlane | null {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  if (!(Math.hypot(dx, dy) > 1e-9)) return null
  return {
    centre: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z],
    azimuth: normaliseAzimuth(Math.atan2(dx, dy) / RAD),
    dip: 90,
  }
}

/**
 * Best-fit vertical plane through a drillhole trace (flat world xyz triples): the plane
 * contains the centroid and the principal horizontal direction of the points (total
 * least squares in plan). The section line points down the hole's horizontal drift, so
 * an inclined hole runs from top left to bottom right. A vertical hole has no drift:
 * it gets a W-E section through the collar.
 */
export function sectionThroughHole(xyz: ArrayLike<number>): SectionPlane | null {
  const count = Math.floor(xyz.length / 3)
  if (count < 1) return null
  let cx = 0
  let cy = 0
  let cz = 0
  for (let i = 0; i < count; i++) {
    cx += xyz[i * 3]
    cy += xyz[i * 3 + 1]
    cz += xyz[i * 3 + 2]
  }
  cx /= count
  cy /= count
  cz /= count
  let sxx = 0
  let sxy = 0
  let syy = 0
  for (let i = 0; i < count; i++) {
    const x = xyz[i * 3] - cx
    const y = xyz[i * 3 + 1] - cy
    sxx += x * x
    sxy += x * y
    syy += y * y
  }
  if (sxx + syy < 1e-6 * count) return { centre: [cx, cy, cz], azimuth: 90, dip: 90 }
  // Principal axis of the 2x2 covariance.
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy)
  let ux = Math.cos(angle)
  let uy = Math.sin(angle)
  // Point it from the first point to the last (down the hole).
  const last = (count - 1) * 3
  if ((xyz[last] - xyz[0]) * ux + (xyz[last + 1] - xyz[1]) * uy < 0) {
    ux = -ux
    uy = -uy
  }
  return { centre: [cx, cy, cz], azimuth: normaliseAzimuth(Math.atan2(ux, uy) / RAD), dip: 90 }
}

/**
 * Where a ray meets the plane, world coordinates; null when the ray is parallel to it
 * (or the hit is behind the origin and `bothWays` is false). The result is projected
 * back onto the plane, so its distance from the plane is zero to rounding.
 */
export function rayPlane(plane: SectionPlane, origin: Vec3, direction: Vec3, bothWays = false): [number, number, number] | null {
  const { n } = frameOf(plane)
  const denominator = dot(direction, n)
  const scale = Math.hypot(direction[0], direction[1], direction[2])
  if (!(Math.abs(denominator) > 1e-9 * scale)) return null
  const t = -signedDistance(plane, origin, n) / denominator
  if (t < 0 && !bothWays) return null
  const p: [number, number, number] = [origin[0] + direction[0] * t, origin[1] + direction[1] * t, origin[2] + direction[2] * t]
  const residual = signedDistance(plane, p, n)
  p[0] -= residual * n[0]
  p[1] -= residual * n[1]
  p[2] -= residual * n[2]
  return p
}

/** Plane coordinates (along the section line, up the plane, towards the viewer) of a world point. */
export function toPlane(plane: SectionPlane, p: Vec3, frame: Frame = frameOf(plane)): [number, number, number] {
  const d: Vec3 = [p[0] - plane.centre[0], p[1] - plane.centre[1], p[2] - plane.centre[2]]
  return [dot(d, frame.u), dot(d, frame.v), dot(d, frame.n)]
}

/** World point of plane coordinates (s along the line, t up the plane, w towards the viewer). */
export function fromPlane(plane: SectionPlane, s: number, t: number, w = 0, frame: Frame = frameOf(plane)): [number, number, number] {
  const { u, v, n } = frame
  return [
    plane.centre[0] + s * u[0] + t * v[0] + w * n[0],
    plane.centre[1] + s * u[1] + t * v[1] + w * n[1],
    plane.centre[2] + s * u[2] + t * v[2] + w * n[2],
  ]
}

export interface BoxLike {
  min: Vec3
  max: Vec3
}

/** Extent of a world box in plane coordinates: [min, max] along u, v and n. */
export function boxInPlane(plane: SectionPlane, box: BoxLike, frame: Frame = frameOf(plane)): { min: [number, number, number]; max: [number, number, number] } {
  const min: [number, number, number] = [Infinity, Infinity, Infinity]
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let corner = 0; corner < 8; corner++) {
    const p = toPlane(
      plane,
      [corner & 1 ? box.max[0] : box.min[0], corner & 2 ? box.max[1] : box.min[1], corner & 4 ? box.max[2] : box.min[2]],
      frame,
    )
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k]
      if (p[k] > max[k]) max[k] = p[k]
    }
  }
  return { min, max }
}

/**
 * How far the plane can travel across a box: the forward distances (see `stepPlane`) at
 * which it reaches the near and the far side of the box. The slider runs over this.
 */
export function travelRange(plane: SectionPlane, box: BoxLike): [number, number] {
  const extent = boxInPlane(plane, box)
  // Forward is -n: a point at n-distance w is reached after stepping -w.
  return [-extent.max[2], -extent.min[2]]
}

/**
 * Which holes have any part inside the slab, and where each one enters it.
 * `positions`: world xyz of both ends of every piece (6 numbers per piece, pieces of
 * hole h are `holeOffsets[h] .. holeOffsets[h + 1] - 1`, ordered down the hole).
 * `anchors` gets the first in-slab point of each hole (3 per hole; NaN when none).
 */
export function holesInSlab(
  plane: SectionPlane,
  slab: Slab,
  positions: ArrayLike<number>,
  holeOffsets: ArrayLike<number>,
): { inside: Uint8Array; anchors: Float64Array; count: number } {
  const { n } = frameOf(plane)
  const holes = holeOffsets.length - 1
  const inside = new Uint8Array(holes)
  const anchors = new Float64Array(holes * 3).fill(Number.NaN)
  const c = plane.centre
  let count = 0
  for (let h = 0; h < holes; h++) {
    for (let piece = holeOffsets[h]; piece < holeOffsets[h + 1]; piece++) {
      const i = piece * 6
      const d0 = (positions[i] - c[0]) * n[0] + (positions[i + 1] - c[1]) * n[1] + (positions[i + 2] - c[2]) * n[2]
      const d1 = (positions[i + 3] - c[0]) * n[0] + (positions[i + 4] - c[1]) * n[1] + (positions[i + 5] - c[2]) * n[2]
      if ((d0 > slab.front && d1 > slab.front) || (d0 < -slab.back && d1 < -slab.back)) continue
      // Parameter of the first in-slab point along the piece.
      let s = 0
      if (d0 > slab.front) s = (d0 - slab.front) / (d0 - d1)
      else if (d0 < -slab.back) s = (-slab.back - d0) / (d1 - d0)
      anchors[h * 3] = positions[i] + (positions[i + 3] - positions[i]) * s
      anchors[h * 3 + 1] = positions[i + 1] + (positions[i + 4] - positions[i + 1]) * s
      anchors[h * 3 + 2] = positions[i + 2] + (positions[i + 5] - positions[i + 2]) * s
      inside[h] = 1
      count++
      break
    }
  }
  return { inside, anchors, count }
}

const COMPASS = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west']

/** "north", "south-east", ...: the compass direction nearest to an azimuth. */
export function compass(azimuth: number): string {
  return COMPASS[Math.round(normaliseAzimuth(azimuth) / 45) % 8]
}

export type SectionKind = 'we' | 'sn' | 'plan' | 'vertical' | 'inclined'

/** W-E (line along east or west), S-N, horizontal plan, other vertical, or inclined. */
export function sectionKind(plane: SectionPlane): SectionKind {
  if (plane.dip < 1e-9) return 'plan'
  if (plane.dip < 90 - 1e-9) return 'inclined'
  const a = normaliseAzimuth(plane.azimuth)
  if (a === 90 || a === 270) return 'we'
  if (a === 0 || a === 180) return 'sn'
  return 'vertical'
}

/** Azimuth the viewer looks towards (horizontal part of -n). */
export function lookAzimuth(plane: SectionPlane): number {
  return normaliseAzimuth(plane.azimuth - 90)
}

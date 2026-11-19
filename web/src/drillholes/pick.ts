/**
 * Picking of drillhole pieces without three.js objects: one pass over the merged
 * position buffer, culled per hole by a bounding sphere. 100 000 pieces in 500 holes
 * cost well under a millisecond per pointer move, because only the few holes whose
 * sphere the ray passes are tested piece by piece.
 *
 * The test runs in RENDER space (scene coordinates with z multiplied by `zScale`, the
 * vertical exaggeration), where screen distances are isotropic. A piece is hit when the
 * ray passes within `tolerance(t) = tol0 + tolSlope * t` of it, t being the distance
 * along the ray: constant for an orthographic camera, growing with distance for a
 * perspective one (a fixed number of pixels either way). The hit closest to the cursor
 * in pixels wins.
 */

export interface PickRay {
  /** Origin and unit direction, render space. */
  origin: readonly [number, number, number]
  direction: readonly [number, number, number]
  /** Pick radius in metres at distance t: tol0 + tolSlope * t. */
  tol0: number
  tolSlope: number
}

export interface PickHit {
  piece: number
  /** Distance along the ray to the closest approach, render space. */
  t: number
  /** Position along the piece, 0 = start, 1 = end. */
  s: number
  /** Miss distance divided by the tolerance there: 0 = dead centre, 1 = at the edge. */
  closeness: number
}

/** Bounding spheres of each hole's pieces, scene space: cx, cy, cz, r per hole. */
export function holeSpheres(positions: Float32Array, holeOffsets: Uint32Array): Float32Array {
  const holes = holeOffsets.length - 1
  const out = new Float32Array(holes * 4)
  for (let h = 0; h < holes; h++) {
    const a = holeOffsets[h] * 6
    const b = holeOffsets[h + 1] * 6
    if (b <= a) {
      out[h * 4 + 3] = -1 // no pieces: never hit
      continue
    }
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
    for (let i = a; i < b; i += 3) {
      const x = positions[i], y = positions[i + 1], z = positions[i + 2]
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (z < minZ) minZ = z
      if (z > maxZ) maxZ = z
    }
    out[h * 4] = (minX + maxX) / 2
    out[h * 4 + 1] = (minY + maxY) / 2
    out[h * 4 + 2] = (minZ + maxZ) / 2
    out[h * 4 + 3] = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2
  }
  return out
}

/**
 * The piece nearest to the ray, or null when none is within tolerance.
 * `positions`: scene xyz of both ends, 6 per piece. `spheres` / `holeOffsets`: from
 * `holeSpheres` and `DrillholePieces.holeOffsets`. `shown`, when given, is asked whether
 * the render-space point of closest approach is drawn at all (section clipping): pieces
 * that are not shown cannot be picked and do not hide the ones behind them.
 */
export function pickPiece(
  ray: PickRay,
  positions: Float32Array,
  holeOffsets: Uint32Array,
  spheres: Float32Array,
  zScale = 1,
  shown?: (x: number, y: number, z: number) => boolean,
): PickHit | null {
  const [ox, oy, oz] = ray.origin
  const [dx, dy, dz] = ray.direction
  const stretch = Math.max(1, zScale)
  let best: PickHit | null = null
  let bestCloseness = 1

  const holes = holeOffsets.length - 1
  for (let h = 0; h < holes; h++) {
    const r = spheres[h * 4 + 3]
    if (r < 0) continue
    // Sphere cull (conservative under z stretch: the radius grows by the stretch).
    const cx = spheres[h * 4] - ox
    const cy = spheres[h * 4 + 1] - oy
    const cz = spheres[h * 4 + 2] * zScale - oz
    const along = cx * dx + cy * dy + cz * dz
    const reach = r * stretch
    if (along < -reach) continue
    const off2 = cx * cx + cy * cy + cz * cz - along * along
    const limit = reach + ray.tol0 + ray.tolSlope * Math.max(0, along + reach)
    if (off2 > limit * limit) continue

    for (let piece = holeOffsets[h]; piece < holeOffsets[h + 1]; piece++) {
      const i = piece * 6
      // Piece start relative to the ray origin (w), and the piece vector (u).
      const wx = positions[i] - ox
      const wy = positions[i + 1] - oy
      const wz = positions[i + 2] * zScale - oz
      const ux = positions[i + 3] - ox - wx
      const uy = positions[i + 4] - oy - wy
      const uz = positions[i + 5] * zScale - oz - wz
      const uu = ux * ux + uy * uy + uz * uz
      const ud = ux * dx + uy * dy + uz * dz
      const wd = wx * dx + wy * dy + wz * dz
      const wu = wx * ux + wy * uy + wz * uz
      // Closest approach of the line w + s u to the ray t d (|d| = 1).
      const denominator = uu - ud * ud
      let s = denominator > 1e-12 * uu ? (ud * wd - wu) / denominator : 0
      s = s < 0 ? 0 : s > 1 ? 1 : s
      const px = wx + s * ux
      const py = wy + s * uy
      const pz = wz + s * uz
      const t = px * dx + py * dy + pz * dz
      if (t <= 0) continue
      const mx = px - t * dx
      const my = py - t * dy
      const mz = pz - t * dz
      const tolerance = ray.tol0 + ray.tolSlope * t
      const closeness = Math.sqrt(mx * mx + my * my + mz * mz) / tolerance
      if (closeness < bestCloseness || (closeness === bestCloseness && best !== null && t < best.t)) {
        if (shown && !shown(px + ox, py + oy, pz + oz)) continue
        bestCloseness = closeness
        best = { piece, t, s, closeness }
      }
    }
  }
  return best
}

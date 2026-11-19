/** Pure camera arithmetic, kept apart from three.js objects so it can be unit-tested. */

const DEG = Math.PI / 180

/** Height in metres of what a perspective camera sees at `distance`. */
export function visibleHeight(fovDeg: number, distance: number): number {
  return 2 * distance * Math.tan((fovDeg * DEG) / 2)
}

/**
 * Orthographic zoom (pixels per metre, frustum = viewport in pixels) that shows the same
 * extent as a perspective camera `distance` away from the orbit centre.
 */
export function orthoZoomForDistance(fovDeg: number, distance: number, viewportHeight: number): number {
  return viewportHeight / visibleHeight(fovDeg, distance)
}

/** Inverse of `orthoZoomForDistance`. */
export function distanceForOrthoZoom(fovDeg: number, zoom: number, viewportHeight: number): number {
  return viewportHeight / zoom / (2 * Math.tan((fovDeg * DEG) / 2))
}

export interface DepthSphere {
  /** Signed distance from the camera to the sphere centre along the view direction. */
  depth: number
  radius: number
}

/**
 * Near/far planes that enclose the given spheres (scene extents, grid).
 *
 * Perspective: near must stay positive, and the far/near ratio bounded, or the depth
 * buffer loses precision at mine scale; near follows the orbit distance when the camera
 * is inside the scene so close-up work is not clipped.
 * Orthographic: depth is linear, so near may be negative (camera inside the model).
 */
export function clipPlanes(
  perspective: boolean,
  spheres: readonly DepthSphere[],
  targetDistance: number,
): { near: number; far: number } {
  let nearest = Infinity
  let farthest = -Infinity
  for (const s of spheres) {
    nearest = Math.min(nearest, s.depth - s.radius)
    farthest = Math.max(farthest, s.depth + s.radius)
  }
  if (!Number.isFinite(nearest) || !Number.isFinite(farthest)) {
    nearest = targetDistance * 0.1
    farthest = targetDistance * 10
  }
  const margin = (farthest - nearest) * 0.02 + 1e-3
  if (!perspective) {
    return { near: nearest - margin, far: farthest + margin }
  }
  const far = Math.max(farthest + margin, targetDistance * 2, 1)
  const near = Math.max(nearest - margin, targetDistance * 0.02, far * 1e-6, 1e-3)
  return { near, far }
}

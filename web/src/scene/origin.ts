/**
 * Scene origin: the one place where world (mine) coordinates become GPU coordinates.
 *
 * Mine coordinates (E 500 000, N 5 300 000) do not fit float32: at 5.3e6 the float32
 * spacing is 0.5 m, so raw coordinates in a vertex buffer jitter and snap. Every layer
 * therefore keeps its data in world float64 and subtracts a shared local origin before
 * creating float32 buffers. The origin is set once, from the bounds centre of the first
 * object registered with the store, and stays fixed while objects are loaded.
 *
 * Three spaces:
 *   world  - x = east, y = north, z = up, metres, float64 (true coordinates)
 *   scene  - world minus origin; what goes into float32 vertex buffers
 *   render - scene with z multiplied by the vertical exaggeration (applied by the
 *            content group's scale, never baked into buffers); camera, grid and
 *            picking work in render space
 *
 * Rule for layers: build buffers with `worldToScene`, report positions to the user
 * with `sceneToWorld` / `renderToWorld`. Never put a world coordinate in a Float32Array.
 */
import { useSyncExternalStore } from 'react'

export type Vec3 = readonly [number, number, number]

const ZERO: Vec3 = [0, 0, 0]

let origin: Vec3 | null = null
const listeners = new Set<() => void>()

function emit(): void {
  for (const listener of listeners) listener()
}

/** The current origin in world coordinates; (0, 0, 0) until the first object is loaded. */
export function getOrigin(): Vec3 {
  return origin ?? ZERO
}

export function hasOrigin(): boolean {
  return origin !== null
}

/**
 * Sets the origin if none is set yet (whole metres, so the subtraction is exact).
 * Returns the origin in force. Later calls do not move it: buffers already built
 * against the origin stay valid.
 */
export function ensureOrigin(centre: Vec3): Vec3 {
  if (origin === null) {
    origin = [Math.round(centre[0]), Math.round(centre[1]), Math.round(centre[2])]
    emit()
  }
  return origin
}

/** Forgets the origin (project closed / tests). Every layer must rebuild its buffers. */
export function resetOrigin(): void {
  if (origin !== null) {
    origin = null
    emit()
  }
}

export function subscribeOrigin(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** React hook: the origin, re-rendering the caller when it changes. */
export function useSceneOrigin(): Vec3 {
  return useSyncExternalStore(subscribeOrigin, getOrigin, getOrigin)
}

/**
 * World float64 xyz triples -> scene float32 xyz triples (ready for a BufferAttribute).
 * The subtraction happens in float64; only the small local offset is rounded to float32.
 */
export function worldToScene(world: ArrayLike<number>, o: Vec3 = getOrigin()): Float32Array {
  if (world.length % 3 !== 0) throw new Error('worldToScene: length must be a multiple of 3')
  const out = new Float32Array(world.length)
  for (let i = 0; i < world.length; i += 3) {
    out[i] = world[i] - o[0]
    out[i + 1] = world[i + 1] - o[1]
    out[i + 2] = world[i + 2] - o[2]
  }
  return out
}

/** Scene xyz triples -> world float64 xyz triples. */
export function sceneToWorld(scene: ArrayLike<number>, o: Vec3 = getOrigin()): Float64Array {
  if (scene.length % 3 !== 0) throw new Error('sceneToWorld: length must be a multiple of 3')
  const out = new Float64Array(scene.length)
  for (let i = 0; i < scene.length; i += 3) {
    out[i] = scene[i] + o[0]
    out[i + 1] = scene[i + 1] + o[1]
    out[i + 2] = scene[i + 2] + o[2]
  }
  return out
}

/** One world point -> render space (float64; for object positions, camera targets). */
export function worldToRender(
  p: Vec3,
  exaggeration: number,
  o: Vec3 = getOrigin(),
): [number, number, number] {
  return [p[0] - o[0], p[1] - o[1], (p[2] - o[2]) * exaggeration]
}

/** One render-space point (e.g. a ray hit) -> true world coordinates. */
export function renderToWorld(
  p: Vec3,
  exaggeration: number,
  o: Vec3 = getOrigin(),
): [number, number, number] {
  return [p[0] + o[0], p[1] + o[1], p[2] / exaggeration + o[2]]
}

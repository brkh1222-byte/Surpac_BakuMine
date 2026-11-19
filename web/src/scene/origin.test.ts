import { beforeEach, expect, test } from 'vitest'
import {
  ensureOrigin,
  getOrigin,
  hasOrigin,
  renderToWorld,
  resetOrigin,
  sceneToWorld,
  subscribeOrigin,
  worldToRender,
  worldToScene,
} from './origin'

const E = 500_000
const N = 5_300_000
const Z = 1_200

beforeEach(() => resetOrigin())

test('origin is (0,0,0) until set, then fixed by the first caller', () => {
  expect(hasOrigin()).toBe(false)
  expect(getOrigin()).toEqual([0, 0, 0])
  expect(ensureOrigin([E + 0.4, N - 0.4, Z])).toEqual([E, N, Z])
  expect(ensureOrigin([1, 2, 3])).toEqual([E, N, Z])
  expect(hasOrigin()).toBe(true)
})

test('subscribers hear the origin being set and reset', () => {
  let calls = 0
  const off = subscribeOrigin(() => calls++)
  ensureOrigin([E, N, Z])
  ensureOrigin([0, 0, 0])
  resetOrigin()
  off()
  ensureOrigin([E, N, Z])
  expect(calls).toBe(2)
})

test('raw float32 cannot hold a 0.01 m offset at mine coordinates (why the origin exists)', () => {
  expect(Math.abs(Math.fround(N + 0.01) - Math.fround(N) - 0.01)).toBeGreaterThan(0.005)
  expect(Math.abs(Math.fround(E + 0.01) - Math.fround(E) - 0.01)).toBeGreaterThan(0.005)
})

test('a 0.01 m offset at E 500 000 / N 5 300 000 survives the float32 round trip', () => {
  // Object 800 m away from the origin, as in a real deposit.
  ensureOrigin([E, N, Z])
  const a = [E + 800, N - 800, Z + 150]
  const b = [a[0] + 0.01, a[1] + 0.01, a[2] + 0.01]
  const scene = worldToScene([...a, ...b])
  expect(scene).toBeInstanceOf(Float32Array)

  // The offset is still there inside the float32 buffer...
  for (let k = 0; k < 3; k++) {
    expect(Math.abs(scene[3 + k] - scene[k] - 0.01)).toBeLessThan(1e-4)
  }
  // ...and comes back in world coordinates to well under a millimetre.
  const world = sceneToWorld(scene)
  expect(world).toBeInstanceOf(Float64Array)
  for (let k = 0; k < 3; k++) {
    expect(Math.abs(world[k] - a[k])).toBeLessThan(1e-4)
    expect(Math.abs(world[3 + k] - b[k])).toBeLessThan(1e-4)
    expect(Math.abs(world[3 + k] - world[k] - 0.01)).toBeLessThan(1e-4)
  }
})

test('scene buffers hold small numbers only', () => {
  ensureOrigin([E, N, Z])
  const scene = worldToScene([E - 2500, N + 2500, Z - 400, E + 2500, N - 2500, Z + 400])
  for (const v of scene) expect(Math.abs(v)).toBeLessThan(1e4)
})

test('explicit origin argument and length check', () => {
  expect(Array.from(worldToScene([10, 20, 30], [1, 2, 3]))).toEqual([9, 18, 27])
  expect(Array.from(sceneToWorld([9, 18, 27], [1, 2, 3]))).toEqual([10, 20, 30])
  expect(() => worldToScene([1, 2])).toThrow()
  expect(() => sceneToWorld([1, 2, 3, 4])).toThrow()
})

test('vertical exaggeration scales render Z only and read-outs report true Z', () => {
  ensureOrigin([E, N, Z])
  const p = [E + 12.5, N - 7.25, Z + 40] as const
  const r = worldToRender(p, 5)
  expect(r).toEqual([12.5, -7.25, 200])
  const back = renderToWorld(r, 5)
  expect(back[0]).toBeCloseTo(p[0], 9)
  expect(back[1]).toBeCloseTo(p[1], 9)
  expect(back[2]).toBeCloseTo(p[2], 9)
})

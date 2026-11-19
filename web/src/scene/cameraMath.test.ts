import { expect, test } from 'vitest'
import { clipPlanes, distanceForOrthoZoom, orthoZoomForDistance, visibleHeight } from './cameraMath'

test('ortho <-> perspective conversion preserves the visible extent', () => {
  const fov = 35
  const height = 900
  const distance = 1234.5
  const zoom = orthoZoomForDistance(fov, distance, height)
  // Orthographic: viewport pixels / zoom = metres shown.
  expect(height / zoom).toBeCloseTo(visibleHeight(fov, distance), 9)
  expect(distanceForOrthoZoom(fov, zoom, height)).toBeCloseTo(distance, 9)
})

test('perspective clip planes enclose the scene and keep a sane depth ratio', () => {
  // Camera 2 km from a 600 m radius deposit.
  const far1 = clipPlanes(true, [{ depth: 2000, radius: 600 }], 2000)
  expect(far1.near).toBeGreaterThan(1000)
  expect(far1.near).toBeLessThanOrEqual(1400)
  expect(far1.far).toBeGreaterThanOrEqual(2600)

  // Camera inside the deposit, 0.5 m from the orbit centre: near stays positive and small.
  const close = clipPlanes(true, [{ depth: 100, radius: 600 }], 0.5)
  expect(close.near).toBeGreaterThan(0)
  expect(close.near).toBeLessThan(0.5)
  expect(close.far).toBeGreaterThanOrEqual(700)
  expect(close.far / close.near).toBeLessThanOrEqual(1e6 + 1)
})

test('orthographic clip planes may start behind the camera', () => {
  const planes = clipPlanes(false, [{ depth: 100, radius: 600 }], 50)
  expect(planes.near).toBeLessThan(-500)
  expect(planes.far).toBeGreaterThan(700)
})

test('no extents falls back to planes around the orbit distance', () => {
  const planes = clipPlanes(true, [], 100)
  expect(planes.near).toBeGreaterThan(0)
  expect(planes.near).toBeLessThan(100)
  expect(planes.far).toBeGreaterThan(100)
})

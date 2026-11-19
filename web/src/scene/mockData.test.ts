import { expect, test } from 'vitest'
import { buildMockData, MOCK_CENTRE } from './mockData'
import { worldToScene } from './origin'

test('mock data sits at mine coordinates and is deterministic', () => {
  const a = buildMockData()
  const b = buildMockData()
  expect(a.boxes).toHaveLength(36)
  expect(Array.from(a.boxPositions)).toEqual(Array.from(b.boxPositions))
  expect(a.boxPositions).toBeInstanceOf(Float64Array)
  expect(a.linePositions).toBeInstanceOf(Float64Array)
  expect(a.boxPositions.length).toBe(36 * 36 * 3)
  expect(a.linePositions.length % 6).toBe(0)

  for (const bounds of [a.boxBounds, a.lineBounds]) {
    expect(Math.abs(bounds.min[0] - MOCK_CENTRE.e)).toBeLessThan(700)
    expect(Math.abs(bounds.max[1] - MOCK_CENTRE.n)).toBeLessThan(700)
    expect(bounds.min[2]).toBeGreaterThan(800)
    expect(bounds.max[2]).toBeLessThan(1400)
  }
})

test('mock buffers are small once the origin is subtracted', () => {
  const data = buildMockData()
  const scene = worldToScene(data.boxPositions, [MOCK_CENTRE.e, MOCK_CENTRE.n, MOCK_CENTRE.z])
  let max = 0
  for (const value of scene) max = Math.max(max, Math.abs(value))
  expect(max).toBeLessThan(1000)
})

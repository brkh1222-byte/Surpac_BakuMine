import { beforeEach, expect, test } from 'vitest'
import { getOrigin, resetOrigin } from '../scene/origin'
import { allBounds, boundsOf, isObjectVisible, useViewer, visibleBounds } from './store'

const box = (x: number, y: number, z: number, half: number) => ({
  min: [x - half, y - half, z - half] as const,
  max: [x + half, y + half, z + half] as const,
})

beforeEach(() => {
  resetOrigin()
  useViewer.setState({
    objects: {},
    categoryVisible: { drillholes: true, surfaces: true, solids: true, blockModels: true },
    verticalExaggeration: 1,
  })
})

test('the first registered object fixes the scene origin at its bounds centre', () => {
  const { registerObject } = useViewer.getState()
  registerObject({ id: 'a', name: 'A', category: 'solids', bounds: box(500_000, 5_300_000, 1_200, 50) })
  registerObject({ id: 'b', name: 'B', category: 'drillholes', bounds: box(500_400, 5_300_400, 1_100, 50) })
  expect(getOrigin()).toEqual([500_000, 5_300_000, 1_200])
})

test('visibility is the object toggle AND its category toggle', () => {
  const s = useViewer.getState()
  s.registerObject({ id: 'a', name: 'A', category: 'solids', bounds: box(0, 0, 0, 10) })
  s.registerObject({ id: 'b', name: 'B', category: 'drillholes', bounds: box(100, 0, 0, 10) })
  expect(isObjectVisible(useViewer.getState(), 'a')).toBe(true)

  s.setCategoryVisible('solids', false)
  expect(isObjectVisible(useViewer.getState(), 'a')).toBe(false)
  expect(visibleBounds(useViewer.getState())).toEqual(box(100, 0, 0, 10))
  expect(allBounds(useViewer.getState())).toEqual({ min: [-10, -10, -10], max: [110, 10, 10] })

  s.setObjectVisible('b', false)
  expect(visibleBounds(useViewer.getState())).toBeNull()

  // Re-registering (data reloaded) keeps the user's visibility choice.
  s.registerObject({ id: 'b', name: 'B2', category: 'drillholes', bounds: box(100, 0, 0, 10) })
  expect(useViewer.getState().objects.b.visible).toBe(false)

  s.removeObject('a')
  expect(useViewer.getState().objects.a).toBeUndefined()
})

test('vertical exaggeration is clamped and ignores NaN', () => {
  const s = useViewer.getState()
  s.setVerticalExaggeration(1000)
  expect(useViewer.getState().verticalExaggeration).toBe(50)
  s.setVerticalExaggeration(Number.NaN)
  expect(useViewer.getState().verticalExaggeration).toBe(50)
  s.setVerticalExaggeration(0)
  expect(useViewer.getState().verticalExaggeration).toBe(0.1)
})

test('boundsOf', () => {
  expect(boundsOf(new Float64Array([1, 2, 3, -1, 5, 0]))).toEqual({ min: [-1, 2, 0], max: [1, 5, 3] })
})

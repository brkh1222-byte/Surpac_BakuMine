import { expect, test } from 'vitest'
import { decimalsFor, formatCoord, formatLength, niceStep } from './format'

test('niceStep rounds up to 1-2-5 steps', () => {
  expect(niceStep(0.7)).toBe(1)
  expect(niceStep(1)).toBe(1)
  expect(niceStep(1.2)).toBe(2)
  expect(niceStep(3)).toBe(5)
  expect(niceStep(7)).toBe(10)
  expect(niceStep(130)).toBe(200)
  expect(niceStep(0.03)).toBeCloseTo(0.05, 12)
  expect(niceStep(0)).toBe(1)
  expect(niceStep(Number.NaN)).toBe(1)
})

test('decimalsFor', () => {
  expect(decimalsFor(100)).toBe(0)
  expect(decimalsFor(1)).toBe(0)
  expect(decimalsFor(0.5)).toBe(1)
  expect(decimalsFor(0.02)).toBe(2)
})

test('formatCoord groups digits and keeps decimals', () => {
  expect(formatCoord(5_300_000)).toBe('5 300 000')
  expect(formatCoord(500_012.346, 2)).toBe('500 012.35')
  expect(formatCoord(999)).toBe('999')
  expect(formatCoord(-1250)).toBe('−1 250')
  expect(formatCoord(-0.001, 1)).toBe('0.0')
  expect(formatCoord(Number.NaN)).toBe('–')
})

test('formatLength', () => {
  expect(formatLength(250)).toBe('250 m')
  expect(formatLength(5000)).toBe('5 km')
  expect(formatLength(0.05)).toBe('0.05 m')
})

import { describe, expect, it } from 'vitest'
import { cutRock, hazeAt, HAZE_LENGTH, reliefShade, SUN } from './terrain'

const unit = (v: readonly number[]) => {
  const l = Math.hypot(v[0], v[1], v[2])
  return [v[0] / l, v[1] / l, v[2] / l]
}

describe('terrain look', () => {
  it('keeps the brightness of flat ground and shades slopes by the sun', () => {
    const sun = unit(SUN)
    expect(reliefShade([0, 0, 1], sun)).toBeCloseTo(1, 12)
    // A slope of 30 degrees facing the sun (south-east) and one facing away (north-west).
    const c = Math.cos(Math.PI / 6)
    const s = Math.sin(Math.PI / 6)
    const towards = [s * Math.SQRT1_2, -s * Math.SQRT1_2, c]
    const away = [-s * Math.SQRT1_2, s * Math.SQRT1_2, c]
    // Sun 45 degrees up: cos of the angle to the normal is cos(45 -/+ 30).
    expect(reliefShade(towards, sun)).toBeCloseTo(1 + 0.85 * (Math.cos(Math.PI / 12) - Math.SQRT1_2), 12)
    expect(reliefShade(away, sun)).toBeCloseTo(1 + 0.85 * (Math.cos((75 * Math.PI) / 180) - Math.SQRT1_2), 12)
    expect(reliefShade(towards, sun)).toBeGreaterThan(1)
    expect(reliefShade(away, sun)).toBeLessThan(1)
    // Limits: a wall facing away is not black, strength cannot blow the picture out.
    expect(reliefShade([-Math.SQRT1_2, Math.SQRT1_2, 0], sun)).toBe(0.5)
    expect(reliefShade(sun, sun, 10)).toBe(1.35)
  })

  it('hazes by 1 - exp(-d / L)', () => {
    expect(hazeAt(0)).toBe(0)
    expect(hazeAt(-5)).toBe(0)
    expect(hazeAt(HAZE_LENGTH)).toBeCloseTo(1 - 1 / Math.E, 12)
    expect(hazeAt(1000)).toBeLessThan(0.09) // across the site the picture keeps its colours
  })

  it('turns cut ground into rock between 0.2 and 1 m of cut', () => {
    expect(cutRock(0)).toBe(0)
    expect(cutRock(0.2)).toBe(0)
    expect(cutRock(0.6)).toBeCloseTo(0.5, 12)
    expect(cutRock(1)).toBe(1)
    expect(cutRock(12)).toBe(1)
    expect(cutRock(null)).toBe(0)
    expect(cutRock(undefined)).toBe(0)
    expect(cutRock(Number.NaN)).toBe(0)
  })
})

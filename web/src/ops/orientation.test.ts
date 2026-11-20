import { describe, expect, it } from 'vitest'
import { describePlane, parsePlane, planeAngles, planeAxes, trendPlunge, type Vec3 } from './orientation'

const RAD = Math.PI / 180
/** Unit vector plunging `plunge` below horizontal towards `azimuth`. */
const direction = (azimuth: number, plunge: number): Vec3 => [
  Math.sin(azimuth * RAD) * Math.cos(plunge * RAD),
  Math.cos(azimuth * RAD) * Math.cos(plunge * RAD),
  -Math.sin(plunge * RAD),
]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const sameAxis = (a: Vec3, b: Vec3) => Math.abs(Math.abs(dot(a, b)) - 1) < 1e-9

describe('plane to ellipsoid axes', () => {
  it('a plane dipping 69 towards 216: major along strike, semi-major down dip, minor = pole', () => {
    const { major, semiMajor, minor } = planeAxes({ dip: 69, dip_azimuth: 216, pitch: 0 })
    expect(Math.abs(major[2])).toBeLessThan(1e-12) // horizontal
    expect(sameAxis(major, direction(306, 0))).toBe(true)
    expect(sameAxis(major, direction(126, 0))).toBe(true)
    expect(sameAxis(semiMajor, direction(216, 69))).toBe(true)
    expect(sameAxis(minor, direction(36, 21))).toBe(true)
    expect(minor[2]).toBeGreaterThan(0)
    // the pole is perpendicular to the strike line and to the dip line
    expect(Math.abs(dot(minor, direction(126, 0)))).toBeLessThan(1e-12)
    expect(Math.abs(dot(minor, direction(216, 69)))).toBeLessThan(1e-12)
    expect(trendPlunge(major)).toEqual({ azimuth: 126, plunge: 0 })
    expect(trendPlunge(semiMajor)).toEqual({ azimuth: 216, plunge: 69 })
    expect(trendPlunge(minor)).toEqual({ azimuth: 36, plunge: 21 })
    expect(describePlane({ dip: 69, dip_azimuth: 216, pitch: 0 })).toBe('Axes (azimuth/plunge): major 126/00, semi-major 216/69, minor 036/21')
  })

  it('gives the same rotation angles as the Python helper', () => {
    // values of bakumine.core.estimation.orientation.angles_from_plane (tests/estimation/test_orientation.py)
    expect(planeAngles({ dip: 69, dip_azimuth: 216, pitch: 0 })).toEqual([126, 0, -69])
    const pitched = planeAngles({ dip: 69, dip_azimuth: 216, pitch: 30 })
    expect(pitched[0]).toBeCloseTo(137.689769, 5)
    expect(pitched[1]).toBeCloseTo(-27.826143, 5)
    expect(pitched[2]).toBeCloseTo(-66.09478, 5)
    expect(planeAngles({ dip: 69, dip_azimuth: 216, pitch: 90 })).toEqual([216, -69, 0])
    expect(planeAngles({ dip: 0, dip_azimuth: 0, pitch: 0 })).toEqual([270, 0, 0])
    // a vertical major axis (gimbal lock) still gives finite angles
    expect(planeAngles({ dip: 90, dip_azimuth: 40, pitch: 90 }).every(Number.isFinite)).toBe(true)
  })

  it('a pitch turns the major axis in the plane, from strike towards down dip', () => {
    for (const pitch of [0, 25, 90, 140]) {
      const plane = { dip: 55, dip_azimuth: 80, pitch }
      const { major, semiMajor, minor } = planeAxes(plane)
      expect(dot(major, direction(80 - 90, 0))).toBeCloseTo(Math.cos(pitch * RAD), 12)
      expect(dot(major, direction(80, 55))).toBeCloseTo(Math.sin(pitch * RAD), 12)
      expect(Math.abs(dot(major, semiMajor))).toBeLessThan(1e-12)
      expect(Math.abs(dot(major, minor))).toBeLessThan(1e-12)
      expect(Math.abs(dot(semiMajor, minor))).toBeLessThan(1e-12)
    }
  })

  it('parses the three fields', () => {
    expect(parsePlane('69', '216', '')).toEqual({ dip: 69, dip_azimuth: 216, pitch: 0 })
    expect(parsePlane(' 69 ', '216', '15')).toEqual({ dip: 69, dip_azimuth: 216, pitch: 15 })
    expect(parsePlane('', '216', '')).toMatch(/dip/i)
    expect(parsePlane('95', '216', '')).toMatch(/0 to 90/)
    expect(parsePlane('60', 'north', '')).toMatch(/azimuth/)
    expect(parsePlane('60', '10', 'x')).toMatch(/Pitch/)
  })
})

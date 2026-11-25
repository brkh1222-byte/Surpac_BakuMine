import { describe, expect, it } from 'vitest'
import {
  blastCentre, depthBelowCollars, DUST_DRAG_S, DUST_RISE, DUST_WIND, dustCentre, dustRadius, EQUIPMENT_MARGIN_M, equipmentSites, fragmentOffset,
  fragmentSize, hash01, machineIdle, SHAKE_DECAY_MS, shakeEnvelope,
} from './realism'
import { heightSampler } from './topography'

describe('site-view rules', () => {
  it('hash01 is repeatable, in [0, 1) and spread evenly', () => {
    let sum = 0
    const bins = new Array(10).fill(0)
    for (let i = 0; i < 20000; i++) {
      const h = hash01(i, 7)
      expect(h).toBeGreaterThanOrEqual(0)
      expect(h).toBeLessThan(1)
      sum += h
      bins[Math.floor(h * 10)]++
    }
    expect(hash01(123, 7)).toBe(hash01(123, 7))
    expect(hash01(123, 7)).not.toBe(hash01(123, 8))
    expect(sum / 20000).toBeCloseTo(0.5, 1)
    for (const b of bins) expect(Math.abs(b - 2000)).toBeLessThan(200) // 4.5 sigma of a fair bin
  })

  it('fragment sizes invert the Rosin-Rammler curve', () => {
    const curve = { xc_m: 0.3, n: 1.7 }
    for (const u of [0.05, 0.5, 0.8, 0.95]) {
      const x = fragmentSize(curve, u)
      expect(1 - Math.exp(-((x / curve.xc_m) ** curve.n))).toBeCloseTo(u, 12)
    }
    // The median is x50 = xc (ln 2)^(1/n).
    expect(fragmentSize(curve, 0.5)).toBeCloseTo(0.3 * Math.log(2) ** (1 / 1.7), 12)
    expect(Number.isFinite(fragmentSize(curve, 1))).toBe(true)
    expect(fragmentSize(curve, 0)).toBeGreaterThan(0)
  })

  it('depth is measured below the nearest collar', () => {
    const collars = [[0, 0, 100], [50, 0, 110]]
    expect(depthBelowCollars([2, 1, 98.5], collars)).toBeCloseTo(1.5, 12)
    expect(depthBelowCollars([49, 0, 98.5], collars)).toBeCloseTo(11.5, 12)
    expect(depthBelowCollars([1, 1, 101], collars)).toBeCloseTo(-1, 12)
  })

  it('a fragment drifts from its parcel in flight and meets the ground with it', () => {
    const out = [0, 0, 0]
    const d0 = [0.2, -0.1, 0.4]
    const dv = [1, 2, -0.5]
    fragmentOffset(d0, dv, 0, 2, out)
    expect(out).toEqual([0.2, -0.1, 0.4])
    fragmentOffset(d0, dv, 1, 2, out) // half way: no fading yet
    expect(out[0]).toBeCloseTo(1.2, 12)
    expect(out[1]).toBeCloseTo(1.9, 12)
    expect(out[2]).toBeCloseTo(-0.1, 12)
    fragmentOffset(d0, dv, 2, 2, out) // landing: horizontal drift kept, height offset gone
    expect(out[0]).toBeCloseTo(2.2, 12)
    expect(out[1]).toBeCloseTo(3.9, 12)
    expect(out[2]).toBeCloseTo(0, 12)
    const frozen = [0, 0, 0]
    fragmentOffset(d0, dv, 9, 2, frozen) // after landing: frozen
    expect(frozen[0]).toBeCloseTo(out[0], 12)
    expect(frozen[1]).toBeCloseTo(out[1], 12)
  })

  it('dust follows dv/dt = (wind - v) / tau, checked by stepping the equation', () => {
    const p0 = [10, 20, 30]
    const v0 = [8, -3, 12]
    const wind = [DUST_WIND[0], DUST_WIND[1], DUST_RISE]
    // Independent check: classical Runge-Kutta on x' = v, v' = (wind - v) / tau.
    const x = [...p0]
    const v = [...v0]
    const h = 1e-3
    for (let step = 0; step < 4000; step++) {
      for (let j = 0; j < 3; j++) {
        const a = (u: number) => (wind[j] - u) / DUST_DRAG_S
        const k1v = a(v[j])
        const k2v = a(v[j] + (h / 2) * k1v)
        const k3v = a(v[j] + (h / 2) * k2v)
        const k4v = a(v[j] + h * k3v)
        x[j] += (h / 6) * (v[j] + 2 * (v[j] + (h / 2) * k1v) + 2 * (v[j] + (h / 2) * k2v) + (v[j] + h * k3v))
        v[j] += (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
      }
    }
    const out = [0, 0, 0]
    dustCentre(p0, v0, 4, out)
    for (let j = 0; j < 3; j++) expect(out[j]).toBeCloseTo(x[j], 6)
    dustCentre(p0, v0, 0, out)
    expect(out).toEqual(p0)
    expect(dustRadius(1, 2, 4)).toBe(5)
    expect(dustRadius(1, 2, -1)).toBe(1)
  })

  it('the excavators stand behind the face, outside the clearance radius', () => {
    const holes = [
      { collar: [100, 200, 50] }, { collar: [140, 200, 50] }, { collar: [100, 230, 52] }, { collar: [140, 230, 52] },
    ]
    const bundle = {
      timeline: { holes }, muckpile: { origin: [0, 0] }, flyrock: { clearance_m: 400 }, design: { face_azimuth: 180 },
    } as unknown as Parameters<typeof equipmentSites>[0]
    const centre = blastCentre(bundle)
    expect(centre).toEqual([120, 215, 51])
    const sites = equipmentSites(bundle)
    expect(sites).toHaveLength(2)
    for (const s of sites) {
      expect(Math.hypot(s.x - centre[0], s.y - centre[1])).toBeGreaterThanOrEqual(400 + EQUIPMENT_MARGIN_M)
      expect(s.y).toBeGreaterThan(centre[1] + 400) // face looks south (180): behind is north
      // Roughly facing the blast.
      const bearing = Math.atan2(centre[1] - s.y, centre[0] - s.x)
      expect(Math.abs(Math.atan2(Math.sin(s.heading - bearing), Math.cos(s.heading - bearing)))).toBeLessThan(0.5)
    }
    expect(Math.hypot(sites[0].x - sites[1].x, sites[0].y - sites[1].y)).toBeCloseTo(18, 9)
    // Face to the east (90): behind is west.
    const east = equipmentSites({ ...bundle, design: { face_azimuth: 90 } } as unknown as Parameters<typeof equipmentSites>[0])
    for (const s of east) expect(s.x).toBeLessThan(centre[0] - 400)
  })
})

describe('camera tremble', () => {
  it('starts when the wave arrives, decays, sums over holes and weakens with distance', () => {
    const cp = 5000
    // 100 m from the blast: the wave of a hole fired at 50 ms arrives at 50 + 100 / 5000 s = 70 ms.
    expect(shakeEnvelope([50], 69.9, 100, cp)).toBe(0)
    expect(shakeEnvelope([50], 70, 100, cp)).toBeCloseTo(1, 12)
    expect(shakeEnvelope([50], 70 + SHAKE_DECAY_MS, 100, cp)).toBeCloseTo(Math.exp(-1), 12)
    // Two holes: the sum of their decays, limited to 1.
    const two = shakeEnvelope([0, 100], 300, 0, cp)
    expect(two).toBeCloseTo(Math.exp(-300 / SHAKE_DECAY_MS) + Math.exp(-200 / SHAKE_DECAY_MS), 12)
    expect(shakeEnvelope([0, 1, 2, 3], 3, 0, cp)).toBe(1)
    // Beyond 150 m it falls as 150 / distance; the arrival is later too.
    expect(shakeEnvelope([0], 160, 800, cp)).toBeCloseTo(150 / 800, 12)
    expect(shakeEnvelope([0], 159, 800, cp)).toBe(0)
    expect(shakeEnvelope([], 1000, 100, cp)).toBe(0)
  })
})

describe('waiting machines', () => {
  it('swing and curl stay small, repeat with the time and differ between machines', () => {
    const deg = Math.PI / 180
    let swing = 0
    let curl = 0
    for (let t = 0; t <= 60; t += 0.05) {
      const pose = machineIdle(t, 2.3)
      swing = Math.max(swing, Math.abs(pose.swing))
      curl = Math.max(curl, Math.abs(pose.curl))
    }
    expect(swing).toBeLessThanOrEqual(7 * deg + 1e-12)
    expect(swing).toBeGreaterThan(6.9 * deg)
    expect(curl).toBeLessThanOrEqual(9 * deg + 1e-12)
    expect(curl).toBeGreaterThan(8.9 * deg)
    expect(machineIdle(3.2, 0)).toEqual(machineIdle(3.2, 0))
    expect(machineIdle(3.2, 0).swing).not.toBeCloseTo(machineIdle(3.2, 2.3).swing, 3)
    expect(machineIdle(0, 0).swing).toBe(0)
  })
})

describe('height lookup on a surface', () => {
  const plane = (x: number, y: number) => 3 + 0.2 * x - 0.1 * y

  it('is bilinear on a complete grid: exact on a plane, NaN outside', () => {
    const p: number[] = []
    for (let i = 0; i < 6; i++) for (let j = 0; j < 4; j++) p.push(1000 + i * 10, 500 + j * 25, plane(1000 + i * 10, 500 + j * 25))
    // Vertex order must not matter.
    const shuffled: number[] = []
    for (const k of [7, 0, 23, 3, 12, 5, 18, 1, 9, 22, 2, 14, 6, 19, 4, 11, 21, 8, 16, 10, 13, 20, 15, 17]) shuffled.push(...p.slice(k * 3, k * 3 + 3))
    const z = heightSampler(shuffled)
    for (const [x, y] of [[1000, 500], [1013.7, 541.2], [1050, 575], [1049.9, 500.1]]) expect(z(x, y)).toBeCloseTo(plane(x, y), 9)
    expect(z(999.9, 520)).toBeNaN()
    expect(z(1020, 575.1)).toBeNaN()
    // A saddle z = x y on the unit cell is reproduced too (bilinear, not planar).
    const saddle = heightSampler([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    expect(saddle(0.25, 0.5)).toBeCloseTo(0.125, 12)
  })

  it('falls back to nearby vertices on scattered points: exact at a vertex, close on a plane', () => {
    const p: number[] = []
    let seed = 1
    const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646
    for (let i = 0; i < 400; i++) {
      const x = random() * 100
      const y = random() * 100
      p.push(x, y, plane(x, y))
    }
    const z = heightSampler(p)
    expect(z(p[30], p[31])).toBe(p[32])
    let worst = 0
    for (let i = 0; i < 200; i++) {
      const x = 10 + random() * 80
      const y = 10 + random() * 80
      worst = Math.max(worst, Math.abs(z(x, y) - plane(x, y)))
    }
    expect(worst).toBeLessThan(1.5) // slope 0.22 over the 5 m point spacing: inverse-distance is not exact
    expect(z(500, 500)).toBeNaN()
    expect(heightSampler([])(0, 0)).toBeNaN()
  })
})

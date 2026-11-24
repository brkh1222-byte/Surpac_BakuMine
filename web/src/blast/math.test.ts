import { expect, test } from 'vitest'
import { makeDemoBundle } from './fixture'
import {
  activeWaves, advance, autoSpeed, contourSegments, countAtOrBefore, detonationSplit, firePulse, landedFraction, lastFire, logScale,
  particlePosition, Phase, ppvAtRadius, ppvColor, PPV_STOPS, radiusAtPpv, ramp, sampleGrid, sliderToTime, sortedFinite, timelineEnd,
  timeToSlider, toHex, validateBundle, waveRadius,
} from './math'
import type { ThrowParticles, WaveSource } from './types'

test('wave radius: zero until the source fires, then cp * dt', () => {
  expect(waveRadius(5000, 10, 25)).toBe(0)
  expect(waveRadius(5000, 25, 25)).toBe(0)
  // 5000 m/s for 4 ms = 20 m
  expect(waveRadius(5000, 29, 25)).toBeCloseTo(20, 12)
})

test('PPV follows the scaled-distance law (hand-worked) and its inverse', () => {
  // k = 1140, beta = 1.6, Q = 100 kg, R = 100 m: SD = 10, PPV = 1140 * 10^-1.6 = 28.635...
  expect(ppvAtRadius(1140, 1.6, 100, 100)).toBeCloseTo(1140 / 10 ** 1.6, 10)
  expect(ppvAtRadius(1140, 1.6, 100, 100)).toBeCloseTo(28.6354, 3)
  // Doubling the distance divides the PPV by 2^beta.
  expect(ppvAtRadius(1140, 1.6, 100, 200) / ppvAtRadius(1140, 1.6, 100, 100)).toBeCloseTo(2 ** -1.6, 12)
  expect(ppvAtRadius(1140, 1.6, 100, 0)).toBe(Infinity)
  expect(ppvAtRadius(1140, 1.6, 0, 50)).toBe(0)
  expect(radiusAtPpv(1140, 1.6, 100, ppvAtRadius(1140, 1.6, 100, 73))).toBeCloseTo(73, 9)
})

test('active waves: fired, above the cut-off, newest first when capped', () => {
  const sources: WaveSource[] = [0, 10, 20, 30].map((t, i) => ({ hole: `h${i}`, t_ms: t, centre: [0, 0, 0], charge_kg: 100 }))
  const w = { cp_m_s: 5000, k: 1140, beta: 1.6 }
  expect(activeWaves(sources, 5, w, 10, 16).map((a) => a.index)).toEqual([0])
  expect(activeWaves(sources, 5, w, 10, 16)[0].radius).toBeCloseTo(25, 12)
  expect(activeWaves(sources, 25, w, 10, 2).map((a) => a.index)).toEqual([2, 1])
  // R(10 mm/s) = 10 * 114^(1/1.6) = 193 m = 38.6 ms: at t = 45 the first front has faded.
  expect(activeWaves(sources, 45, w, 10, 16).map((a) => a.index)).toEqual([1, 2, 3])
})

test('log colour scale', () => {
  expect(logScale(1, 1, 1000)).toBe(0)
  expect(logScale(1000, 1, 1000)).toBe(1)
  expect(logScale(10, 1, 1000)).toBeCloseTo(1 / 3, 12)
  expect(logScale(0.1, 1, 1000)).toBe(0)
  expect(logScale(1e6, 1, 1000)).toBe(1)
  expect(logScale(NaN, 1, 1000)).toBe(0)
  expect(toHex(ppvColor(1))).toBe(PPV_STOPS[0])
  expect(toHex(ppvColor(5000))).toBe(PPV_STOPS[PPV_STOPS.length - 1])
  expect(toHex(ramp(['#000000', '#ffffff'], 0.5))).toBe('#808080')
})

test('fire pulse: off before, 1 at the instant, 1/e after 50 ms', () => {
  expect(firePulse(99, 100)).toBe(0)
  expect(firePulse(100, 100)).toBe(1)
  expect(firePulse(150, 100)).toBeCloseTo(Math.exp(-1), 12)
})

const ONE: ThrowParticles = {
  p0: [[10, 20, 5]],
  v0: [[3, 4, 9.81]],
  t0_ms: [100],
  // Up at 9.81 m/s under g = 9.81: back at the launch height after exactly 2 s.
  t_land_ms: [2100],
  p_land: [[16, 28, 5]],
  p_rest: [[17, 28, 4]],
  size_m: [0.2],
  kind: [0],
}

test('particle: in place before launch, ballistic in flight, eased to rest after landing', () => {
  const out = [0, 0, 0]
  expect(particlePosition(ONE, 0, 50, 9.81, out)).toBe(Phase.InPlace)
  expect(out).toEqual([10, 20, 5])
  // Apex after 1 s: z = 5 + 9.81 - 9.81/2
  expect(particlePosition(ONE, 0, 1100, 9.81, out)).toBe(Phase.Flying)
  expect(out[0]).toBeCloseTo(13, 12)
  expect(out[1]).toBeCloseTo(24, 12)
  expect(out[2]).toBeCloseTo(5 + 9.81 / 2, 12)
  // Just before landing the ballistic path meets p_land.
  particlePosition(ONE, 0, 2100 - 1e-6, 9.81, out)
  expect(out[0]).toBeCloseTo(16, 6)
  expect(out[2]).toBeCloseTo(5, 6)
  expect(particlePosition(ONE, 0, 2100, 9.81, out, 400)).toBe(Phase.Settling)
  expect(out).toEqual([16, 28, 5])
  particlePosition(ONE, 0, 2300, 9.81, out, 400)
  expect(out[0]).toBeCloseTo(16.5, 12)
  expect(out[2]).toBeCloseTo(4.5, 12)
  expect(particlePosition(ONE, 0, 2500, 9.81, out, 400)).toBe(Phase.AtRest)
  expect(out).toEqual([17, 28, 4])
})

test('terrain blend factor = fraction of particles landed', () => {
  const landed = sortedFinite([300, 100, null, 200, 400])
  expect(Array.from(landed)).toEqual([100, 200, 300, 400])
  expect(countAtOrBefore(landed, 99)).toBe(0)
  expect(countAtOrBefore(landed, 200)).toBe(2)
  expect(landedFraction(landed, 0)).toBe(0)
  expect(landedFraction(landed, 250)).toBe(0.5)
  expect(landedFraction(landed, 1e9)).toBe(1)
  expect(landedFraction([], 0)).toBe(0)
  expect(landedFraction([], 5)).toBe(1)
})

test('timeline extents include the last landing and its settling', () => {
  const base = {
    timeline: { t_end_ms: 500, holes: [{ t_fire_ms: 0 }, { t_fire_ms: 168 }] },
    throw: { g: 9.81, particles: { ...ONE } },
  } as unknown as Parameters<typeof timelineEnd>[0]
  expect(timelineEnd(base, 400)).toBe(2500)
  expect(lastFire(base)).toBe(168)
  const long = { ...base, timeline: { ...base.timeline, t_end_ms: 9000 } }
  expect(timelineEnd(long, 400)).toBe(9000)
})

test('player clock: advance, stop at the end, loop, auto speed', () => {
  expect(advance(100, 0.016, 0.05, 1000, false)).toEqual({ t: 100 + 16 * 0.05, playing: true })
  expect(advance(999, 0.1, 1, 1000, false)).toEqual({ t: 1000, playing: false })
  expect(advance(999, 0.1, 1, 1000, true)).toEqual({ t: 0, playing: true })
  expect(autoSpeed(100, 168)).toBe(0.05)
  expect(autoSpeed(300, 168)).toBe(1)
})

test('timeline slider: half for the detonation, half for the flight, and back', () => {
  const split = detonationSplit(168, 3000)
  expect(split).toBe(228)
  expect(sliderToTime(0.25, split, 3000)).toBeCloseTo(114, 12)
  expect(sliderToTime(0.5, split, 3000)).toBeCloseTo(228, 12)
  expect(sliderToTime(0.75, split, 3000)).toBeCloseTo(228 + 1386, 12)
  for (const t of [0, 50, 228, 1000, 3000]) expect(sliderToTime(timeToSlider(t, split, 3000), split, 3000)).toBeCloseTo(t, 9)
  // No flight after the detonation: plain linear.
  expect(sliderToTime(0.5, 100, 100)).toBe(50)
})

test('grid sampling is bilinear and clamps at the edge', () => {
  const grid = { origin: [100, 200] as [number, number], spacing: 2, shape: [2, 3] as [number, number] }
  // z = x-index * 10 + y-index, row-major [ix * ny + iy]
  const z = [0, 1, 2, 10, 11, 12]
  expect(sampleGrid(grid, z, 100, 200)).toBe(0)
  expect(sampleGrid(grid, z, 102, 204)).toBe(12)
  expect(sampleGrid(grid, z, 101, 201)).toBeCloseTo(5.5, 12)
  expect(sampleGrid(grid, z, 500, 0)).toBe(10)
})

test('iso-line of a plane is the straight line where it crosses the level', () => {
  const grid = { origin: [0, 0] as [number, number], spacing: 1, shape: [4, 3] as [number, number] }
  const values: number[] = []
  const z: number[] = []
  for (let ix = 0; ix < 4; ix++) for (let iy = 0; iy < 3; iy++) {
    values.push(ix * 10)
    z.push(100 + ix)
  }
  const seg = contourSegments(grid, values, z, 15)
  // Two cells in y, one segment each, all points at x = 1.5 and z = 101.5.
  expect(seg.length).toBe(2 * 6)
  for (let i = 0; i < seg.length; i += 3) {
    expect(seg[i]).toBeCloseTo(1.5, 12)
    expect(seg[i + 2]).toBeCloseTo(101.5, 12)
  }
  expect(contourSegments(grid, values, z, 99)).toEqual([])
})

test('bundle validation: the demo passes; wrong shapes are named', () => {
  const bundle = makeDemoBundle()
  expect(validateBundle(bundle)).toEqual([])
  expect(validateBundle(JSON.parse(JSON.stringify(bundle)))).toEqual([])
  expect(validateBundle(null)).toHaveLength(1)
  expect(validateBundle({ ...bundle, format: 'x' })[0]).toMatch(/format/)
  const { waves: _waves, ...noWaves } = bundle
  expect(validateBundle(noWaves)).toContain('"waves" is missing')

  const short = structuredClone(bundle)
  short.throw.particles.t_land_ms.pop()
  expect(validateBundle(short).join('\n')).toMatch(/throw\.particles\.t_land_ms has \d+ entries, expected \d+/)
  const grid = structuredClone(bundle)
  grid.muckpile.z_after.push(0)
  expect(validateBundle(grid).join('\n')).toMatch(/muckpile\.z_after/)
  const blocks = structuredClone(bundle)
  blocks.damage.blocks.cls.pop()
  expect(validateBundle(blocks).join('\n')).toMatch(/damage\.blocks\.cls/)
})

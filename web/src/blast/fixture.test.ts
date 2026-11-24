import { expect, test } from 'vitest'
import { apiAbsent, BlastApiError, detailText, loadResult, simulate } from './api'
import { chargeBins, DEFAULT_DESIGN, DEMO_HOLES_PER_ROW, DEMO_ROWS, demoDelay, makeDemoBundle, maxInstantaneousCharge } from './fixture'
import { particlePosition, ppvAtRadius, sampleGrid, timelineEnd, validateBundle } from './math'
import { productFromSearch, searchForProduct } from './product'

const bundle = makeDemoBundle()

test('the demo bundle is deterministic and valid', () => {
  expect(validateBundle(bundle)).toEqual([])
  expect(JSON.stringify(makeDemoBundle())).toBe(JSON.stringify(bundle))
  expect(bundle.timeline.holes).toHaveLength(DEMO_ROWS * DEMO_HOLES_PER_ROW)
  expect(bundle.throw.particles.t0_ms.length).toBeGreaterThanOrEqual(3000)
  // Finite numbers everywhere a number is promised (JSON has no NaN).
  expect(JSON.stringify(bundle)).not.toMatch(/NaN|Infinity/)
  expect(JSON.parse(JSON.stringify(bundle))).toEqual(bundle)
})

test('charge per hole = pi/4 d^2 L rho, and the totals add up', () => {
  const h = bundle.timeline.holes[0]
  const length = h.charge_top[2] - h.charge_bottom[2]
  expect(h.charge_kg).toBeCloseTo((Math.PI / 4) * 0.115 ** 2 * length * 850, 2)
  const s = bundle.summary
  expect(s.explosive_kg).toBeCloseTo(h.charge_kg * 40, 0)
  expect(s.powder_factor_kg_m3).toBeCloseTo(s.explosive_kg / s.volume_m3, 2)
  expect(s.tonnes).toBeCloseTo(s.volume_m3 * 2.7, 0)
})

test('row-by-row timing, MIC and the charge envelope', () => {
  const holes = bundle.timeline.holes
  for (const h of holes) expect(h.t_fire_ms).toBe(h.row * 42)
  expect(bundle.summary.duration_ms).toBe(4 * 42)
  // A whole row fires together: MIC = 8 holes.
  expect(bundle.summary.mic_kg).toBeCloseTo(8 * holes[0].charge_kg, 0)
  expect(maxInstantaneousCharge([{ t_fire_ms: 0, charge_kg: 5 }, { t_fire_ms: 7.9, charge_kg: 7 }, { t_fire_ms: 8, charge_kg: 20 }])).toBe(27)
  const envelope = bundle.envelope!
  expect(envelope.bins).toHaveLength(5)
  expect(Math.max(...envelope.bins.map((b) => b.kg))).toBeCloseTo(envelope.mic_kg, 0)
  expect(chargeBins([{ id: 'a', t_fire_ms: 0, charge_kg: 1 }, { id: 'b', t_fire_ms: 9, charge_kg: 2 }, { id: 'c', t_fire_ms: 15, charge_kg: 3 }])).toEqual([
    { t_ms: 0, kg: 1, holes: ['a'] },
    { t_ms: 8, kg: 5, holes: ['b', 'c'] },
  ])
  expect(demoDelay('hole_by_hole', 1, 3, 8, 17, 42)).toBe(42 + 51)
  expect(demoDelay('v_cut', 0, 0, 8, 10, 42)).toBe(35)
  expect(demoDelay('echelon', 2, 3, 8, 17, 42)).toBe(210)
})

test('particles land where their ballistic path says, in front of the face', () => {
  const p = bundle.throw.particles
  const out = [0, 0, 0]
  let forward = 0
  let flyrock = 0
  for (let i = 0; i < p.t0_ms.length; i += 7) {
    particlePosition(p, i, p.t_land_ms[i] - 1e-6, bundle.throw.g, out)
    expect(Math.hypot(out[0] - p.p_land[i][0], out[1] - p.p_land[i][1], out[2] - p.p_land[i][2])).toBeLessThan(0.02)
    expect(p.t_land_ms[i]).toBeGreaterThan(p.t0_ms[i])
  }
  for (let i = 0; i < p.t0_ms.length; i++) {
    // Face azimuth 180: thrown south (y decreases).
    if (p.kind[i] === 0 && p.p_land[i][1] < p.p0[i][1]) forward++
    if (p.kind[i] === 1) flyrock++
  }
  expect(forward).toBe(3000)
  expect(flyrock).toBeGreaterThan(0)
  expect(timelineEnd(bundle)).toBeGreaterThanOrEqual(Math.max(...p.t_land_ms))
})

test('muckpile: swell x dug-out volume, and fragments rest on it', () => {
  const m = bundle.muckpile
  const cell = m.spacing ** 2
  let net = 0
  let dug = 0
  for (let i = 0; i < m.z_before.length; i++) {
    const d = m.z_after[i]! - m.z_before[i]!
    net += d * cell
    if (d < 0) dug += -d * cell
  }
  // Net gain = (swell - 1) x the dug-out volume, which is close to the design volume.
  expect(net).toBeGreaterThan(0)
  expect(net / bundle.summary.volume_m3).toBeGreaterThan(0.2)
  expect(net / bundle.summary.volume_m3).toBeLessThan(0.4)
  expect(dug).toBeGreaterThan(0)
  const p = bundle.throw.particles
  for (let i = 0; i < 3000; i += 11) {
    expect(p.p_rest[i][2]).toBeCloseTo(sampleGrid(m, m.z_after, p.p_rest[i][0], p.p_rest[i][1]), 2)
  }
})

test('PPV surface, damage and monitors are consistent with their laws', () => {
  const g = bundle.ppv_surface
  expect(g.z).toHaveLength(g.shape[0] * g.shape[1])
  expect(Math.min(...(g.arrival_ms as number[]))).toBeGreaterThanOrEqual(0)
  const m = bundle.monitors[0]
  expect(m.ppv_mm_s).toBeCloseTo(ppvAtRadius(bundle.waves.k, bundle.waves.beta, m.charge_kg, m.distance_m), 0)
  expect(m.ok).toBe(m.ppv_mm_s <= m.limit_mm_s!)
  const b = bundle.damage.blocks
  expect(b.blasted.some(Boolean)).toBe(true)
  expect(b.cls.some((c) => c === 0)).toBe(true)
  b.cls.forEach((c, i) => {
    if (c === 0) expect(b.t_break_ms[i]).toBeNull()
    else expect(b.t_break_ms[i]).toBeGreaterThan(0)
  })
  const f = bundle.fragmentation
  // Rosin-Rammler identity: half passes at x50.
  expect(1 - Math.exp(-Math.pow(f.x50_m / f.xc_m, f.n))).toBeCloseTo(0.5, 2)
  expect(bundle.summary.p80_m).toBeGreaterThan(f.x50_m)
  expect(bundle.flyrock.clearance_m).toBeCloseTo(bundle.flyrock.max_range_m * bundle.flyrock.safety_factor, 0)
})

test('design parameters change the demo', () => {
  const other = makeDemoBundle({ ...DEFAULT_DESIGN, bench_height: 12, face_azimuth: 90, timing: { type: 'hole_by_hole', inter_hole_ms: 17, inter_row_ms: 42 } })
  expect(validateBundle(other)).toEqual([])
  expect(other.summary.volume_m3).toBeGreaterThan(bundle.summary.volume_m3)
  expect(other.summary.mic_kg).toBeLessThan(bundle.summary.mic_kg)
  // Face looks east: fragments go east.
  const p = other.throw.particles
  expect(p.p_land[0][0]).toBeGreaterThan(p.p0[0][0])
})

test('API client: a served bundle is used; 404, network failure and a bad bundle fall back to the demo', async () => {
  const ok = (async () => new Response(JSON.stringify(bundle), { status: 200 })) as typeof fetch
  const served = await loadResult(DEFAULT_DESIGN, undefined, ok)
  expect(served.source).toBe('api')
  expect(served.reason).toBeNull()

  const missing = (async () => new Response(JSON.stringify({ detail: 'no blast result' }), { status: 404 })) as typeof fetch
  const fallback = await loadResult(DEFAULT_DESIGN, undefined, missing)
  expect(fallback.source).toBe('demo')
  expect(fallback.reason).toMatch(/404/)
  expect(validateBundle(fallback.bundle)).toEqual([])

  const down = (async () => {
    throw new TypeError('fetch failed')
  }) as typeof fetch
  expect((await simulate(DEFAULT_DESIGN, undefined, down)).source).toBe('demo')

  const bad = (async () => new Response(JSON.stringify({ format: 'bakumine-blast-result', version: 1 }), { status: 200 })) as typeof fetch
  const rejected = await loadResult(DEFAULT_DESIGN, undefined, bad)
  expect(rejected.source).toBe('demo')
  expect(rejected.reason).toMatch(/not a valid bundle/)
  await expect(simulate(DEFAULT_DESIGN, undefined, bad)).rejects.toThrow(/not a valid bundle/)

  // A design the API refuses is an error with the API's words, not a reason to show the demo.
  const refused = (async () => new Response(JSON.stringify({ detail: 'the polygon is outside the topography' }), { status: 400 })) as typeof fetch
  await expect(simulate(DEFAULT_DESIGN, undefined, refused)).rejects.toThrow('400: the polygon is outside the topography')
  expect(detailText([{ loc: ['body', 'bench_height'], msg: 'Input should be greater than 0' }])).toBe('bench_height: Input should be greater than 0')
  expect(apiAbsent(new BlastApiError('x', 404))).toBe(true)
  expect(apiAbsent(new BlastApiError('x', 400))).toBe(false)

  let sent: unknown = null
  const echo = (async (_url: RequestInfo | URL, init?: RequestInit) => {
    sent = JSON.parse(String(init?.body))
    return new Response(JSON.stringify(bundle), { status: 200 })
  }) as typeof fetch
  expect((await simulate(DEFAULT_DESIGN, undefined, echo)).source).toBe('api')
  expect(sent).toEqual(DEFAULT_DESIGN)
})

test('product switch lives in the URL', () => {
  expect(productFromSearch('')).toBe('modeller')
  expect(productFromSearch('?app=blast&blast=demo')).toBe('blast')
  expect(searchForProduct('?theme=light', 'blast')).toBe('?theme=light&app=blast')
  expect(searchForProduct('?app=blast&blast=demo&theme=light', 'modeller')).toBe('?theme=light')
  expect(searchForProduct('?app=blast', 'modeller')).toBe('')
})

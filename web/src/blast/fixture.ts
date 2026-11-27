/**
 * Demo result bundle, built in the browser so the tab works without the API
 * (`?app=blast&blast=demo`, and the fallback when `/api/v1/blast/*` is not there).
 *
 * It is a stand-in with the right shape and plausible magnitudes, NOT the engineering
 * model: the real numbers come from `bakumine.blast` (tested against published cases).
 * The formulas below are the same families (rules of thumb, Kuz-Ram, scaled distance,
 * Richards-Moore, ballistics) in their shortest form. Deterministic: same design, same
 * bundle.
 *
 * Geometry: a bench of `bench_height` above a floor at z = 1200, free face looking along
 * `face_azimuth`, 5 rows x 8 holes behind the crest. Local frame: a = along the face,
 * b = out of the face (b < 0 is the bench, b > 0 the lower floor in front).
 */
import { ppvAtRadius, radiusAtPpv, sampleGrid } from './math'
import {
  BUNDLE_FORMAT,
  BUNDLE_VERSION,
  type BlastBundle,
  type DesignMonitor,
  type DesignParams,
  type EnvelopeBin,
  type GridSpec,
  type MonitorResult,
  type TimelineHole,
  type TimingType,
  type Xyz,
} from './types'

export const DEMO_ROWS = 5
export const DEMO_HOLES_PER_ROW = 8
/** Where the demo bench sits (mine-like coordinates, so the local origin is exercised). */
export const DEMO_ANCHOR: Xyz = [512300, 5291800, 1200]

export const DEFAULT_DESIGN: DesignParams = {
  name: 'Demo bench',
  polygon: null,
  bench_height: 10,
  diameter: 0.115,
  burden: null,
  spacing: null,
  pattern: 'staggered',
  face_azimuth: 180,
  explosive: 'ANFO',
  stemming: null,
  subdrill: null,
  timing: { type: 'row_by_row', inter_hole_ms: 17, inter_row_ms: 42 },
  rock: { density: 2700, ucs_mpa: 100, tensile_mpa: 10, youngs_gpa: 50, poisson: 0.25, rock_factor: null },
  monitors: [],
}

const EXPLOSIVES = {
  ANFO: { density: 850, rws: 100 },
  Emulsion: { density: 1150, rws: 115 },
} as const

const G = 9.81
const SWELL = 1.3
/** Typical published site constants, uncalibrated (USBM RI 8507 style upper bound). */
const VIB_K = 1140
const VIB_BETA = 1.6
/** Holmberg-Persson style near field, point-charge form: PPV = K * Q^alpha / R^beta. */
const HP = { k: 700, alpha: 0.7, beta: 1.5 }
const DAMAGE_THRESHOLDS = [700, 1500, 4000]
const FLYROCK_K = 27
const FLYROCK_SF = 2
const AIR = { k: 185_000, a: -1.2 }

/** Small deterministic PRNG (mulberry32), uniform in [0, 1). */
export function prng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const round = (v: number, digits = 3) => {
  const f = 10 ** digits
  return Math.round(v * f) / f
}

/** Firing time of hole `i` of row `r`, ms. */
export function demoDelay(type: TimingType, r: number, i: number, perRow: number, interHole: number, interRow: number): number {
  switch (type) {
    case 'row_by_row':
      return r * interRow
    case 'hole_by_hole':
      return r * interRow + i * interHole
    case 'echelon':
      return (r + i) * interRow
    case 'v_cut':
      return r * interRow + Math.abs(i - (perRow - 1) / 2) * interHole
  }
}

/** Largest charge fired within any `windowMs` window (brute force). */
export function maxInstantaneousCharge(holes: readonly { t_fire_ms: number; charge_kg: number }[], windowMs = 8): number {
  let best = 0
  for (const a of holes) {
    let sum = 0
    for (const b of holes) if (b.t_fire_ms >= a.t_fire_ms && b.t_fire_ms < a.t_fire_ms + windowMs) sum += b.charge_kg
    if (sum > best) best = sum
  }
  return best
}

/** Charge per consecutive `windowMs` window (empty windows left out). */
export function chargeBins(holes: readonly { id: string; t_fire_ms: number; charge_kg: number }[], windowMs = 8): EnvelopeBin[] {
  const bins = new Map<number, EnvelopeBin>()
  for (const h of holes) {
    const k = Math.floor(h.t_fire_ms / windowMs)
    const bin = bins.get(k) ?? { t_ms: k * windowMs, kg: 0, holes: [] }
    bin.kg = round(bin.kg + h.charge_kg)
    bin.holes.push(h.id)
    bins.set(k, bin)
  }
  return [...bins.values()].sort((a, b) => a.t_ms - b.t_ms)
}

/** Monitors of the demo when the design names none: one near, one far, behind the bench. */
export function demoMonitors(faceAzimuth: number): DesignMonitor[] {
  const az = (faceAzimuth * Math.PI) / 180
  const f = [Math.sin(az), Math.cos(az)]
  const u = [Math.cos(az), -Math.sin(az)]
  const at = (a: number, b: number): Xyz => [
    round(DEMO_ANCHOR[0] + a * u[0] + b * f[0], 2),
    round(DEMO_ANCHOR[1] + a * u[1] + b * f[1], 2),
    DEMO_ANCHOR[2] + 10,
  ]
  return [
    { name: 'Workshop', xyz: at(60, -140), limit_mm_s: 50 },
    { name: 'Village', xyz: at(-120, -210), limit_mm_s: 10 },
  ]
}

export function makeDemoBundle(design: DesignParams = DEFAULT_DESIGN, seed = 20261125): BlastBundle {
  const rand = prng(seed)
  const H = Math.max(1, design.bench_height)
  const d = Math.max(0.03, design.diameter)
  const explosive = EXPLOSIVES[design.explosive] ?? EXPLOSIVES.ANFO
  // Rules of thumb (Ash): burden ~ 25-30 diameters, spacing 1.15 B (staggered) or 1.0 B.
  const B = design.burden ?? round((design.explosive === 'Emulsion' ? 30 : 26) * d, 2)
  const S = design.spacing ?? round((design.pattern === 'staggered' ? 1.15 : 1.0) * B, 2)
  const stemming = Math.min(H * 0.8, design.stemming ?? round(0.8 * B, 2))
  const subdrill = design.subdrill ?? round(0.3 * B, 2)
  const floorZ = DEMO_ANCHOR[2]
  const topZ = floorZ + H
  const rows = DEMO_ROWS
  const perRow = DEMO_HOLES_PER_ROW

  // ---- local frame
  const az = (design.face_azimuth * Math.PI) / 180
  const fx = Math.sin(az)
  const fy = Math.cos(az)
  const ux = Math.cos(az)
  const uy = -Math.sin(az)
  const world = (a: number, b: number): [number, number] => [DEMO_ANCHOR[0] + a * ux + b * fx, DEMO_ANCHOR[1] + a * uy + b * fy]
  const local = (x: number, y: number): [number, number] => {
    const dx = x - DEMO_ANCHOR[0]
    const dy = y - DEMO_ANCHOR[1]
    return [dx * ux + dy * uy, dx * fx + dy * fy]
  }
  const stagger = design.pattern === 'staggered' ? S / 2 : 0
  const width = (perRow - 1) * S + stagger + S
  const a0 = -width / 2
  const a1 = width / 2
  const depth = rows * B
  /** Horizontal run of the (steep) face. */
  const faceRun = H * 0.18

  /** Ground before the blast. */
  const zBefore = (_a: number, b: number) => (b <= 0 ? topZ : b < faceRun ? topZ - (H * b) / faceRun : floorZ)
  /** Ground with the block dug out to the floor: new face behind the last row, side walls. */
  const zDugOut = (a: number, b: number) => {
    const before = zBefore(a, b)
    const back = Math.min(1, Math.max(0, (-depth - b) / faceRun))
    const side = Math.min(1, Math.max(0, Math.max(a0 - a, a - a1) / faceRun))
    return Math.min(before, floorZ + H * Math.max(back, side))
  }

  // ---- holes
  const holeLength = H + subdrill
  const chargeLength = holeLength - stemming
  const chargeKg = (Math.PI / 4) * d * d * chargeLength * explosive.density
  const holes: TimelineHole[] = []
  const holeLocal: [number, number][] = []
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < perRow; i++) {
      const a = a0 + S / 2 + i * S + (r % 2 === 1 ? stagger : 0)
      const b = -B * (r + 1)
      const [x, y] = world(a, b)
      holeLocal.push([a, b])
      holes.push({
        id: `R${r + 1}-${String(i + 1).padStart(2, '0')}`,
        row: r,
        t_fire_ms: round(demoDelay(design.timing.type, r, i, perRow, design.timing.inter_hole_ms, design.timing.inter_row_ms), 2),
        collar: [x, y, topZ],
        toe: [x, y, round(floorZ - subdrill)],
        charge_top: [x, y, round(topZ - stemming)],
        charge_bottom: [x, y, round(floorZ - subdrill)],
        charge_kg: round(chargeKg),
        diameter_m: d,
      })
    }
  }
  const duration = Math.max(...holes.map((h) => h.t_fire_ms))
  const mic = maxInstantaneousCharge(holes)

  // ---- quantities
  const volume = width * depth * H
  const tonnes = (volume * design.rock.density) / 1000
  const explosiveKg = chargeKg * holes.length
  const drilled = holeLength * holes.length
  const pf = explosiveKg / volume

  // ---- fragmentation (Kuz-Ram, shortest form)
  const A = design.rock.rock_factor ?? 7
  const x50 = (A * Math.pow(pf, -0.8) * Math.pow(chargeKg, 1 / 6) * Math.pow(115 / explosive.rws, 19 / 20)) / 100
  const n = Math.min(2.2, Math.max(0.7,
    (2.2 - (14 * B) / (d * 1000)) * Math.sqrt((1 + S / B) / 2) * (chargeLength / H) * (design.pattern === 'staggered' ? 1.1 : 1),
  ))
  const xc = x50 / Math.pow(Math.LN2, 1 / n)
  const passingAt = (x: number) => 1 - Math.exp(-Math.pow(x / xc, n))
  const sizes: number[] = []
  for (let i = 0; i <= 40; i++) sizes.push(round(0.005 * Math.pow(3 / 0.005, i / 40), 4))
  const p80 = xc * Math.pow(Math.log(5), 1 / n)

  // ---- waves
  const E = design.rock.youngs_gpa * 1e9
  const nu = Math.min(0.45, Math.max(0.05, design.rock.poisson))
  const cp = Math.sqrt((E * (1 - nu)) / (design.rock.density * (1 + nu) * (1 - 2 * nu)))
  const cs = Math.sqrt(E / (2 * design.rock.density * (1 + nu)))
  const centres: Xyz[] = holes.map((h) => [h.collar[0], h.collar[1], (h.charge_top[2] + h.charge_bottom[2]) / 2])

  // ---- PPV on the surface
  const pad = 180
  const corners = [world(a0 - pad, -depth - pad), world(a1 + pad, -depth - pad), world(a0 - pad, pad), world(a1 + pad, pad)]
  const gridOver = (pts: [number, number][], spacing: number): GridSpec => {
    const x0 = Math.floor(Math.min(...pts.map((p) => p[0])) / spacing) * spacing
    const y0 = Math.floor(Math.min(...pts.map((p) => p[1])) / spacing) * spacing
    const x1 = Math.max(...pts.map((p) => p[0]))
    const y1 = Math.max(...pts.map((p) => p[1]))
    return { origin: [x0, y0], spacing, shape: [Math.ceil((x1 - x0) / spacing) + 1, Math.ceil((y1 - y0) / spacing) + 1] }
  }
  const ppvGrid = gridOver(corners, 6)
  const surfZ: number[] = []
  const surfPpv: number[] = []
  const surfArrival: number[] = []
  for (let ix = 0; ix < ppvGrid.shape[0]; ix++) {
    for (let iy = 0; iy < ppvGrid.shape[1]; iy++) {
      const x = ppvGrid.origin[0] + ix * ppvGrid.spacing
      const y = ppvGrid.origin[1] + iy * ppvGrid.spacing
      const [a, b] = local(x, y)
      const z = zBefore(a, b)
      let ppv = 0
      let arrival = Infinity
      for (let h = 0; h < holes.length; h++) {
        const c = centres[h]
        const R = Math.max(1, Math.hypot(x - c[0], y - c[1], z - c[2]))
        ppv = Math.max(ppv, ppvAtRadius(VIB_K, VIB_BETA, holes[h].charge_kg, R))
        arrival = Math.min(arrival, holes[h].t_fire_ms + (R / cp) * 1000)
      }
      surfZ.push(round(z))
      surfPpv.push(round(ppv))
      surfArrival.push(round(arrival))
    }
  }

  // ---- damage blocks (world-aligned cubes around the block)
  const size = 2
  const blockCorners = [world(a0 - 6, -depth - 8), world(a1 + 6, -depth - 8), world(a0 - 6, 0), world(a1 + 6, 0)]
  const blockGrid = gridOver(blockCorners, size)
  const blocks: BlastBundle['damage']['blocks'] = { centre: [], size: [], ppv_mm_s: [], cls: [], t_break_ms: [], blasted: [] }
  for (let ix = 0; ix + 1 < blockGrid.shape[0]; ix++) {
    for (let iy = 0; iy + 1 < blockGrid.shape[1]; iy++) {
      const x = blockGrid.origin[0] + (ix + 0.5) * size
      const y = blockGrid.origin[1] + (iy + 0.5) * size
      const [a, b] = local(x, y)
      if (a < a0 - 6 || a > a1 + 6 || b < -depth - 8 || b > 0) continue
      for (let z = floorZ - size / 2; z < topZ; z += size) {
        let ppv = 0
        let tBreak = Infinity
        for (let h = 0; h < holes.length; h++) {
          const hole = holes[h]
          const zc = Math.min(hole.charge_top[2], Math.max(hole.charge_bottom[2], z))
          const R = Math.max(0.4, Math.hypot(x - hole.collar[0], y - hole.collar[1], z - zc))
          const v = (HP.k * Math.pow(hole.charge_kg, HP.alpha)) / Math.pow(R, HP.beta)
          if (v > ppv) {
            ppv = v
            tBreak = hole.t_fire_ms + (R / cp) * 1000 + 1
          }
        }
        const cls = ppv >= DAMAGE_THRESHOLDS[2] ? 3 : ppv >= DAMAGE_THRESHOLDS[1] ? 2 : ppv >= DAMAGE_THRESHOLDS[0] ? 1 : 0
        const blasted = a >= a0 && a <= a1 && b >= -depth && z > floorZ
        blocks.centre.push([x, y, z])
        blocks.size.push([size, size, size])
        blocks.ppv_mm_s.push(round(ppv, 1))
        blocks.cls.push(blasted ? Math.max(cls, 2) : cls)
        blocks.t_break_ms.push(cls > 0 || blasted ? round(tBreak, 2) : null)
        blocks.blasted.push(blasted)
      }
    }
  }
  const backbreak = Math.pow((HP.k * Math.pow(chargeKg, HP.alpha)) / DAMAGE_THRESHOLDS[1], 1 / HP.beta)

  // ---- flyrock (Richards and Moore)
  const linear = chargeKg / chargeLength
  const flyBase = (FLYROCK_K * FLYROCK_K) / G
  const faceBurst = flyBase * Math.pow(Math.sqrt(linear) / B, 2.6)
  const cratering = flyBase * Math.pow(Math.sqrt(linear) / stemming, 2.6)
  const maxRange = Math.max(faceBurst, cratering)
  const clearance = maxRange * FLYROCK_SF

  // ---- throw
  const p: BlastBundle['throw']['particles'] = { p0: [], v0: [], t0_ms: [], t_land_ms: [], p_land: [], p_rest: [], size_m: [], kind: [] }
  const muckCount = 3000
  for (let k = 0; k < muckCount; k++) {
    const h = Math.floor(rand() * holes.length)
    const hole = holes[h]
    const [ha, hb] = holeLocal[h]
    // A point of the hole's own burden: between the hole and the row in front.
    const a = ha + (rand() - 0.5) * S
    const b = Math.min(-0.05, hb + rand() * B)
    const z0 = floorZ + 0.3 + rand() * (H - 0.3)
    const speed = (11 - 1.3 * hole.row) * (0.55 + 0.6 * rand())
    const up = ((8 + 25 * rand()) * Math.PI) / 180
    const swing = ((rand() - 0.5) * 24 * Math.PI) / 180
    const va = Math.sin(swing) * speed * Math.cos(up)
    const vb = Math.cos(swing) * speed * Math.cos(up)
    const vz = speed * Math.sin(up)
    const flight = (vz + Math.sqrt(vz * vz + 2 * G * (z0 - floorZ))) / G
    const [x0, y0] = world(a, b)
    const [xl, yl] = world(a + va * flight, b + vb * flight)
    const t0 = hole.t_fire_ms + 5 + 10 * rand()
    p.p0.push([x0, y0, z0])
    p.v0.push([va * ux + vb * fx, va * uy + vb * fy, vz])
    p.t0_ms.push(round(t0, 2))
    p.t_land_ms.push(round(t0 + flight * 1000, 2))
    p.p_land.push([xl, yl, floorZ])
    p.p_rest.push([xl, yl, floorZ])
    p.size_m.push(round(Math.min(1.5, Math.max(0.02, xc * Math.pow(-Math.log(1 - rand()), 1 / n))), 3))
    p.kind.push(0)
  }
  const flyCount = 36
  for (let k = 0; k < flyCount; k++) {
    const h = Math.floor(rand() * perRow)
    const hole = holes[h]
    const [ha, hb] = holeLocal[h]
    const range = maxRange * (0.15 + 0.6 * rand())
    const up = ((45 + (rand() - 0.5) * 16) * Math.PI) / 180
    const speed = Math.sqrt((G * range) / Math.sin(2 * up))
    const swing = ((rand() - 0.5) * 110 * Math.PI) / 180
    const va = Math.sin(swing) * speed * Math.cos(up)
    const vb = Math.cos(swing) * speed * Math.cos(up)
    const vz = speed * Math.sin(up)
    let flight = (vz + Math.sqrt(vz * vz + 2 * G * H)) / G
    let ground = floorZ
    if (zBefore(ha + va * flight, hb + vb * flight) > floorZ + H / 2) {
      flight = (2 * vz) / G
      ground = topZ
    }
    const [x0, y0] = world(ha, hb)
    const [xl, yl] = world(ha + va * flight, hb + vb * flight)
    const t0 = hole.t_fire_ms + 3 + 6 * rand()
    p.p0.push([x0, y0, topZ])
    p.v0.push([va * ux + vb * fx, va * uy + vb * fy, vz])
    p.t0_ms.push(round(t0, 2))
    p.t_land_ms.push(round(t0 + flight * 1000, 2))
    p.p_land.push([xl, yl, ground])
    p.p_rest.push([xl, yl, ground])
    p.size_m.push(round(0.08 + 0.25 * rand(), 3))
    p.kind.push(1)
  }

  // ---- muckpile heightmap
  const reach = 45
  const margin = 12
  const muckCorners = [world(a0 - margin, -depth - margin), world(a1 + margin, -depth - margin), world(a0 - margin, reach), world(a1 + margin, reach)]
  const muckGrid = gridOver(muckCorners, 1.5)
  const [mnx, mny] = muckGrid.shape
  const before: number[] = new Array(mnx * mny)
  const dug: number[] = new Array(mnx * mny)
  let heap = new Float64Array(mnx * mny)
  for (let ix = 0; ix < mnx; ix++) {
    for (let iy = 0; iy < mny; iy++) {
      const [a, b] = local(muckGrid.origin[0] + ix * muckGrid.spacing, muckGrid.origin[1] + iy * muckGrid.spacing)
      before[ix * mny + iy] = zBefore(a, b)
      dug[ix * mny + iy] = zDugOut(a, b)
    }
  }
  for (let k = 0; k < muckCount; k++) {
    const ix = Math.round((p.p_land[k][0] - muckGrid.origin[0]) / muckGrid.spacing)
    const iy = Math.round((p.p_land[k][1] - muckGrid.origin[1]) / muckGrid.spacing)
    if (ix >= 0 && ix < mnx && iy >= 0 && iy < mny) heap[ix * mny + iy] += 1
  }
  // Box blur: fragments pile up at the angle of repose, not as spikes.
  for (let pass = 0; pass < 4; pass++) {
    const next = new Float64Array(mnx * mny)
    for (let ix = 0; ix < mnx; ix++) {
      for (let iy = 0; iy < mny; iy++) {
        let sum = 0
        let count = 0
        for (let dx = -2; dx <= 2; dx++) {
          for (let dy = -2; dy <= 2; dy++) {
            const jx = ix + dx
            const jy = iy + dy
            if (jx < 0 || jx >= mnx || jy < 0 || jy >= mny) continue
            sum += heap[jx * mny + jy]
            count++
          }
        }
        next[ix * mny + iy] = sum / count
      }
    }
    heap = next
  }
  // Scale the heap so that muck volume = swell x dug-out volume (on this grid, exactly).
  const cell = muckGrid.spacing * muckGrid.spacing
  let dugVolume = 0
  let heapVolume = 0
  for (let i = 0; i < heap.length; i++) {
    dugVolume += (before[i] - dug[i]) * cell
    heapVolume += heap[i] * cell
  }
  const scale = heapVolume > 0 ? (SWELL * dugVolume) / heapVolume : 0
  const after = dug.map((z, i) => round(z + heap[i] * scale))
  const zBeforeList = before.map((z) => round(z))
  for (let k = 0; k < muckCount; k++) {
    p.p_rest[k] = [p.p_land[k][0], p.p_land[k][1], round(sampleGrid(muckGrid, after, p.p_land[k][0], p.p_land[k][1]))]
  }
  const tEnd = Math.max(...p.t_land_ms) + 500

  // ---- airblast and monitors
  const dbl = (pa: number) => 20 * Math.log10(pa / 2e-5)
  const airPa = (R: number, Q: number) => AIR.k * Math.pow(R / Math.cbrt(Q), AIR.a)
  const rings = [133, 125, 120].map((level) => {
    const pa = 2e-5 * Math.pow(10, level / 20)
    return { radius_m: round(Math.cbrt(mic) * Math.pow(pa / AIR.k, 1 / AIR.a), 1), pa: round(pa, 2), dbl: level }
  })
  const [bx, by] = world(0, -depth / 2)
  const blastCentre: Xyz = [bx, by, floorZ + H / 2]
  const wanted = design.monitors.length > 0 ? design.monitors : demoMonitors(design.face_azimuth)
  const monitors: MonitorResult[] = wanted.map((m) => {
    const distance = Math.hypot(m.xyz[0] - blastCentre[0], m.xyz[1] - blastCentre[1], m.xyz[2] - blastCentre[2])
    const scaled = distance / Math.sqrt(mic)
    const ppv = ppvAtRadius(VIB_K, VIB_BETA, mic, distance)
    const air = dbl(airPa(Math.max(distance, 1), mic))
    return {
      name: m.name,
      xyz: m.xyz,
      distance_m: round(distance, 1),
      charge_kg: round(mic, 1),
      scaled_distance: round(scaled, 2),
      ppv_mm_s: round(ppv, 2),
      limit_mm_s: m.limit_mm_s,
      ok: ppv <= m.limit_mm_s,
      airblast_dbl: round(air, 1),
      airblast_limit_dbl: 120,
      airblast_ok: air <= 120,
    }
  })

  return {
    format: BUNDLE_FORMAT,
    version: BUNDLE_VERSION,
    design: {
      name: design.name,
      bench_height: H,
      floor_z: floorZ,
      burden: B,
      spacing: S,
      pattern: design.pattern,
      face_azimuth: design.face_azimuth,
      polygon: [world(a0, -depth), world(a1, -depth), world(a1, 0), world(a0, 0)],
      notes: { source: 'browser demo fixture, not the engineering model' },
    },
    summary: {
      volume_m3: round(volume, 1),
      tonnes: round(tonnes, 1),
      explosive_kg: round(explosiveKg, 1),
      powder_factor_kg_m3: round(pf),
      powder_factor_kg_t: round(explosiveKg / tonnes),
      drilled_m: round(drilled, 1),
      holes: holes.length,
      x50_m: round(x50),
      uniformity_n: round(n),
      p80_m: round(p80),
      oversize_fraction: round(1 - passingAt(1), 4),
      mic_kg: round(mic, 1),
      duration_ms: duration,
      flyrock_max_m: round(maxRange, 1),
      clearance_m: round(clearance, 1),
      backbreak_m: round(backbreak, 2),
      models: {
        demo_fixture: { calibrated: false, note: 'generated in the browser; shapes and magnitudes only' },
        kuz_ram: { rock_factor: A, calibrated: false },
        vibration_scaled_distance: { k: VIB_K, beta: VIB_BETA, calibrated: false },
        holmberg_persson: { ...HP, calibrated: false },
        flyrock_richards_moore: { k: FLYROCK_K, safety_factor: FLYROCK_SF, calibrated: false },
        airblast_cube_root: { k: AIR.k, a: AIR.a, calibrated: false },
        throw_ballistic: { swell: SWELL, calibrated: false },
      },
    },
    timeline: { t_end_ms: round(tEnd, 1), holes },
    waves: {
      cp_m_s: round(cp, 1),
      cs_m_s: round(cs, 1),
      k: VIB_K,
      beta: VIB_BETA,
      sources: holes.map((h, i) => ({ hole: h.id, t_ms: h.t_fire_ms, centre: centres[i], charge_kg: h.charge_kg })),
    },
    ppv_surface: { ...ppvGrid, z: surfZ, ppv_mm_s: surfPpv, arrival_ms: surfArrival },
    damage: { classes: ['intact', 'damaged', 'fractured', 'crushed'], thresholds_mm_s: DAMAGE_THRESHOLDS, blocks },
    fragmentation: { sizes_m: sizes, passing: sizes.map((x) => round(passingAt(x), 5)), x50_m: round(x50), n: round(n), xc_m: round(xc) },
    throw: { g: G, particles: p },
    muckpile: { ...muckGrid, z_before: zBeforeList, z_after: after },
    flyrock: {
      face_burst_m: round(faceBurst, 1),
      cratering_m: round(cratering, 1),
      stemming_ejection_m: round(cratering, 1),
      max_range_m: round(maxRange, 1),
      clearance_m: round(clearance, 1),
      safety_factor: FLYROCK_SF,
    },
    airblast: { k: AIR.k, a: AIR.a, rings },
    monitors,
    envelope: { window_ms: 8, bins: chargeBins(holes, 8), mic_kg: round(mic, 1) },
  }
}

/** Radius at which the demo vibration law drops to `ppv` for the bundle's MIC (for tests / hints). */
export function demoRadiusAtPpv(micKg: number, ppv: number): number {
  return radiusAtPpv(VIB_K, VIB_BETA, micKg, ppv)
}

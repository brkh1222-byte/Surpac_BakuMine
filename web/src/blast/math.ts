/**
 * Pure animation math of the Blast BakuMine tab. No three.js, no DOM: everything here is
 * tested in math.test.ts against hand-worked numbers.
 *
 * The viewer does not model anything: it replays the result bundle with the formulas the
 * bundle names (plan_blast.md section 3).
 */
import { BUNDLE_FORMAT, BUNDLE_VERSION, type BlastBundle, type GridSpec, type ThrowParticles, type WaveSource } from './types'

export type Rgb = [number, number, number]

/** After landing a fragment slides from `p_land` to `p_rest` over this long. */
export const SETTLE_MS = 400
/** A fired hole glows for about this long (1/e time of the pulse). */
export const FLASH_MS = 50

// ---- waves ---------------------------------------------------------------------------

/** Wavefront radius in metres at `tMs` for a source fired at `t0Ms`; 0 before it fires. */
export function waveRadius(cpMs: number, tMs: number, t0Ms: number): number {
  return tMs <= t0Ms ? 0 : (cpMs * (tMs - t0Ms)) / 1000
}

/** Scaled-distance law: PPV = k * (R / sqrt(Q)) ** -beta, mm/s. Infinity at R = 0. */
export function ppvAtRadius(k: number, beta: number, chargeKg: number, radius: number): number {
  if (!(radius > 0)) return Infinity
  if (!(chargeKg > 0)) return 0
  return k * Math.pow(radius / Math.sqrt(chargeKg), -beta)
}

/** Inverse of `ppvAtRadius`: the radius at which the PPV has dropped to `ppv`. */
export function radiusAtPpv(k: number, beta: number, chargeKg: number, ppv: number): number {
  if (!(chargeKg > 0) || !(ppv > 0)) return 0
  return Math.sqrt(chargeKg) * Math.pow(k / ppv, 1 / beta)
}

export interface ActiveWave {
  /** Index into `sources`. */
  index: number
  radius: number
  ppv: number
}

/**
 * The wavefronts worth drawing at `tMs`: fired, and still above the PPV cut-off. When there
 * are more than `cap`, the most recently fired win (the old ones are large and faint).
 */
export function activeWaves(
  sources: readonly WaveSource[],
  tMs: number,
  waves: { cp_m_s: number; k: number; beta: number },
  cutoff: number,
  cap: number,
): ActiveWave[] {
  const out: ActiveWave[] = []
  for (let index = 0; index < sources.length; index++) {
    const s = sources[index]
    const radius = waveRadius(waves.cp_m_s, tMs, s.t_ms)
    if (radius <= 0) continue
    const ppv = ppvAtRadius(waves.k, waves.beta, s.charge_kg, radius)
    if (ppv < cutoff) continue
    out.push({ index, radius, ppv })
  }
  if (out.length > cap) {
    out.sort((a, b) => sources[b.index].t_ms - sources[a.index].t_ms)
    out.length = Math.max(0, cap)
  }
  return out
}

// ---- colour scales -------------------------------------------------------------------

/** Position of `value` on a log scale from `min` to `max`, clamped to 0..1 (NaN -> 0). */
export function logScale(value: number, min: number, max: number): number {
  if (!(value > 0) || !(min > 0) || !(max > min)) return 0
  const t = Math.log(value / min) / Math.log(max / min)
  return t <= 0 ? 0 : t >= 1 ? 1 : t
}

function hex(h: string): Rgb {
  const n = parseInt(h.slice(1), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

export function toHex(rgb: Rgb): string {
  const byte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255)
  return '#' + ((1 << 24) | (byte(rgb[0]) << 16) | (byte(rgb[1]) << 8) | byte(rgb[2])).toString(16).slice(1)
}

/** Piecewise-linear ramp through evenly spaced stops, sRGB 0..1. */
export function ramp(stops: readonly string[], t: number): Rgb {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(x))
  const a = hex(stops[i])
  const b = hex(stops[i + 1])
  const f = x - i
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** Vibration: calm blue to alarm red. */
export const PPV_STOPS = ['#2c4fb8', '#2b9fc4', '#3cb878', '#e3d23c', '#ee8a2c', '#d8322b'] as const
/** Firing time: early = light, late = deep violet. */
export const TIMING_STOPS = ['#ffe08a', '#f2994a', '#d6537a', '#7d4fc4', '#3f5bd0'] as const
/** Legend range of every PPV colour in the tab, mm/s. */
export const PPV_RANGE = { min: 1, max: 1000 } as const

export function ppvColor(ppv: number, min: number = PPV_RANGE.min, max: number = PPV_RANGE.max): Rgb {
  return ramp(PPV_STOPS, logScale(ppv, min, max))
}

export function timingColor(tFireMs: number, durationMs: number): Rgb {
  return ramp(TIMING_STOPS, durationMs > 0 ? tFireMs / durationMs : 0)
}

/** Damage class colours: intact (not drawn), damaged, fractured, crushed. */
export const DAMAGE_COLORS = ['#7d8790', '#e8cf3a', '#ef8a2e', '#d9352c'] as const

// ---- holes ---------------------------------------------------------------------------

/** Emissive pulse of a hole: 0 before it fires, 1 at the instant, decaying as exp(-dt/50 ms). */
export function firePulse(tMs: number, tFireMs: number, decayMs: number = FLASH_MS): number {
  if (!(tMs >= tFireMs)) return 0
  return Math.exp(-(tMs - tFireMs) / decayMs)
}

// ---- throw ---------------------------------------------------------------------------

/** InPlace: not launched yet (still part of the bench); Settling: landed, sliding to rest. */
export const Phase = { InPlace: 0, Flying: 1, Settling: 2, AtRest: 3 } as const
export type Phase = (typeof Phase)[keyof typeof Phase]

export function smoothstep(x: number): number {
  const t = x <= 0 ? 0 : x >= 1 ? 1 : x
  return t * t * (3 - 2 * t)
}

/**
 * Position of particle `i` at `tMs`, written to `out`. Before launch it sits at p0; in
 * flight p = p0 + v0*dt - g/2*dt^2 (z only); after `t_land_ms` it eases from p_land to
 * p_rest over `settleMs`.
 */
export function particlePosition(
  p: ThrowParticles,
  i: number,
  tMs: number,
  g: number,
  out: number[] | Float64Array,
  settleMs: number = SETTLE_MS,
): Phase {
  const t0 = p.t0_ms[i]
  const tLand = p.t_land_ms[i]
  if (!(tMs > t0)) {
    const a = p.p0[i]
    out[0] = a[0]
    out[1] = a[1]
    out[2] = a[2]
    return Phase.InPlace
  }
  if (tMs < tLand) {
    const dt = (tMs - t0) / 1000
    const a = p.p0[i]
    const v = p.v0[i]
    out[0] = a[0] + v[0] * dt
    out[1] = a[1] + v[1] * dt
    out[2] = a[2] + v[2] * dt - 0.5 * g * dt * dt
    return Phase.Flying
  }
  const a = p.p_land[i]
  const b = p.p_rest[i]
  const e = settleMs > 0 ? smoothstep((tMs - tLand) / settleMs) : 1
  out[0] = a[0] + (b[0] - a[0]) * e
  out[1] = a[1] + (b[1] - a[1]) * e
  out[2] = a[2] + (b[2] - a[2]) * e
  return e >= 1 ? Phase.AtRest : Phase.Settling
}

/** Finite values in ascending order (for `countAtOrBefore`). */
export function sortedFinite(values: readonly (number | null)[]): Float64Array {
  const out = Float64Array.from(values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v)))
  return out.sort()
}

/** How many entries of the ascending array are <= t (binary search). */
export function countAtOrBefore(sorted: ArrayLike<number>, t: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (sorted[mid] <= t) lo = mid + 1
    else hi = mid
  }
  return lo
}

/**
 * Terrain blend factor: the fraction of particles that have landed by `tMs`. The ground
 * mesh shows z_before + f * (z_after - z_before). With no particles the terrain switches
 * at t = 0.
 */
export function landedFraction(sortedLandMs: ArrayLike<number>, tMs: number): number {
  if (sortedLandMs.length === 0) return tMs > 0 ? 1 : 0
  return countAtOrBefore(sortedLandMs, tMs) / sortedLandMs.length
}

// ---- timeline ------------------------------------------------------------------------

/** End of the animation: nothing fires, flies or settles after it. */
export function timelineEnd(bundle: Pick<BlastBundle, 'timeline' | 'throw'>, settleMs: number = SETTLE_MS): number {
  let end = Number.isFinite(bundle.timeline.t_end_ms) ? bundle.timeline.t_end_ms : 0
  for (const h of bundle.timeline.holes) if (h.t_fire_ms + FLASH_MS > end) end = h.t_fire_ms + FLASH_MS
  for (const t of bundle.throw.particles.t_land_ms) if (t + settleMs > end) end = t + settleMs
  return Math.max(end, 1)
}

/** Time of the last detonation. */
export function lastFire(bundle: Pick<BlastBundle, 'timeline'>): number {
  let last = 0
  for (const h of bundle.timeline.holes) if (h.t_fire_ms > last) last = h.t_fire_ms
  return last
}

/**
 * "Auto" playback: slow motion while holes fire and the waves cross the bench, real time
 * for the rock flight that follows.
 */
export function autoSpeed(tMs: number, lastFireMs: number, slow = 0.05, tailMs = 60): number {
  return tMs < lastFireMs + tailMs ? slow : 1
}

/** One step of the player clock. Returns the new time and whether playback goes on. */
export function advance(tMs: number, dtSeconds: number, speed: number, endMs: number, loop: boolean): { t: number; playing: boolean } {
  const t = tMs + dtSeconds * 1000 * speed
  if (t < endMs) return { t, playing: true }
  return loop ? { t: 0, playing: true } : { t: endMs, playing: false }
}

export function formatMs(tMs: number): string {
  return tMs >= 10000 ? `${(tMs / 1000).toFixed(2)} s` : `${tMs.toFixed(tMs < 100 ? 1 : 0)} ms`
}

// ---- grids ---------------------------------------------------------------------------

export function gridIndex(ix: number, iy: number, ny: number): number {
  return ix * ny + iy
}

/** Bilinear sample of node values at world (x, y); clamped to the grid edge. NaN if no data. */
export function sampleGrid(grid: GridSpec, values: readonly (number | null)[], x: number, y: number): number {
  const [nx, ny] = grid.shape
  if (nx < 1 || ny < 1) return NaN
  const fx = Math.min(nx - 1, Math.max(0, (x - grid.origin[0]) / grid.spacing))
  const fy = Math.min(ny - 1, Math.max(0, (y - grid.origin[1]) / grid.spacing))
  const ix = Math.min(Math.max(nx - 2, 0), Math.floor(fx))
  const iy = Math.min(Math.max(ny - 2, 0), Math.floor(fy))
  const ix1 = Math.min(nx - 1, ix + 1)
  const iy1 = Math.min(ny - 1, iy + 1)
  const u = fx - ix
  const v = fy - iy
  const at = (a: number, b: number) => values[a * ny + b] ?? NaN
  return (1 - u) * ((1 - v) * at(ix, iy) + v * at(ix, iy1)) + u * ((1 - v) * at(ix1, iy) + v * at(ix1, iy1))
}

/**
 * Iso-line of `values` at `level` by marching squares (ambiguous cells are split by the
 * cell mean). Returns line segments as flat world triples [x1, y1, z1, x2, y2, z2, ...],
 * z interpolated from `z`. Cells touching a missing value are skipped.
 */
export function contourSegments(
  grid: GridSpec,
  values: readonly (number | null)[],
  z: readonly (number | null)[],
  level: number,
): number[] {
  const [nx, ny] = grid.shape
  const out: number[] = []
  const [x0, y0] = grid.origin
  const h = grid.spacing
  // Corner order: 0 = (ix, iy), 1 = (ix+1, iy), 2 = (ix+1, iy+1), 3 = (ix, iy+1).
  const cx = [0, 1, 1, 0]
  const cy = [0, 0, 1, 1]
  const v = [0, 0, 0, 0]
  const zz = [0, 0, 0, 0]
  for (let ix = 0; ix + 1 < nx; ix++) {
    for (let iy = 0; iy + 1 < ny; iy++) {
      let ok = true
      for (let c = 0; c < 4; c++) {
        const i = (ix + cx[c]) * ny + iy + cy[c]
        const value = values[i]
        const height = z[i]
        if (typeof value !== 'number' || !Number.isFinite(value) || typeof height !== 'number' || !Number.isFinite(height)) {
          ok = false
          break
        }
        v[c] = value
        zz[c] = height
      }
      if (!ok) continue
      const code = (v[0] >= level ? 1 : 0) | (v[1] >= level ? 2 : 0) | (v[2] >= level ? 4 : 0) | (v[3] >= level ? 8 : 0)
      if (code === 0 || code === 15) continue
      /** Crossing point on edge e (e joins corner e and corner e+1). */
      const cross = (e: number) => {
        const a = e
        const b = (e + 1) % 4
        const f = (level - v[a]) / (v[b] - v[a])
        out.push(
          x0 + (ix + cx[a] + (cx[b] - cx[a]) * f) * h,
          y0 + (iy + cy[a] + (cy[b] - cy[a]) * f) * h,
          zz[a] + (zz[b] - zz[a]) * f,
        )
      }
      const seg = (e1: number, e2: number) => {
        cross(e1)
        cross(e2)
      }
      const high = (v[0] + v[1] + v[2] + v[3]) / 4 >= level
      switch (code) {
        case 1: case 14: seg(3, 0); break
        case 2: case 13: seg(0, 1); break
        case 4: case 11: seg(1, 2); break
        case 8: case 7: seg(2, 3); break
        case 3: case 12: seg(3, 1); break
        case 6: case 9: seg(0, 2); break
        case 5:
          if (high) { seg(0, 1); seg(2, 3) } else { seg(3, 0); seg(1, 2) }
          break
        case 10:
          if (high) { seg(3, 0); seg(1, 2) } else { seg(0, 1); seg(2, 3) }
          break
      }
    }
  }
  return out
}

// ---- validation ----------------------------------------------------------------------

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

/**
 * Structural check of a result bundle: the sections exist and parallel arrays have the
 * same length. Returns the problems found; an empty list means the viewer can draw it.
 */
export function validateBundle(value: unknown): string[] {
  const errors: string[] = []
  if (!isObject(value)) return ['the bundle is not a JSON object']
  if (value.format !== BUNDLE_FORMAT) errors.push(`format is ${JSON.stringify(value.format)}, expected "${BUNDLE_FORMAT}"`)
  if (value.version !== BUNDLE_VERSION) errors.push(`version is ${JSON.stringify(value.version)}, expected ${BUNDLE_VERSION}`)
  const sections = ['design', 'summary', 'timeline', 'waves', 'ppv_surface', 'damage', 'fragmentation', 'throw', 'muckpile', 'flyrock', 'airblast']
  for (const key of sections) if (!isObject(value[key])) errors.push(`"${key}" is missing`)
  if (!Array.isArray(value.monitors)) errors.push('"monitors" is missing')
  if (errors.length > 0) return errors
  const b = value as unknown as BlastBundle

  const list = (what: string, v: unknown): number => {
    if (!Array.isArray(v)) {
      errors.push(`${what} is not an array`)
      return -1
    }
    return v.length
  }
  const same = (what: string, arrays: Record<string, unknown>, expected?: number) => {
    let n = expected
    for (const [name, array] of Object.entries(arrays)) {
      const len = list(`${what}.${name}`, array)
      if (len < 0) continue
      if (n === undefined) n = len
      else if (len !== n) errors.push(`${what}.${name} has ${len} entries, expected ${n}`)
    }
  }
  const grid = (what: string, g: GridSpec, arrays: Record<string, unknown>) => {
    if (!Array.isArray(g.origin) || g.origin.length !== 2) errors.push(`${what}.origin must be [x0, y0]`)
    if (!(g.spacing > 0)) errors.push(`${what}.spacing must be > 0`)
    if (!Array.isArray(g.shape) || g.shape.length !== 2 || !g.shape.every((n) => Number.isInteger(n) && n >= 0)) {
      errors.push(`${what}.shape must be [nx, ny]`)
      return
    }
    same(what, arrays, g.shape[0] * g.shape[1])
  }

  if (list('timeline.holes', b.timeline.holes) >= 0) {
    b.timeline.holes.forEach((h, i) => {
      for (const key of ['collar', 'toe', 'charge_top', 'charge_bottom'] as const) {
        if (!Array.isArray(h[key]) || h[key].length !== 3) errors.push(`timeline.holes[${i}].${key} must be [x, y, z]`)
      }
      if (!Number.isFinite(h.t_fire_ms)) errors.push(`timeline.holes[${i}].t_fire_ms is not a number`)
    })
  }
  if (!Number.isFinite(b.timeline.t_end_ms)) errors.push('timeline.t_end_ms is not a number')
  for (const key of ['cp_m_s', 'k', 'beta'] as const) if (!Number.isFinite(b.waves[key])) errors.push(`waves.${key} is not a number`)
  list('waves.sources', b.waves.sources)
  grid('ppv_surface', b.ppv_surface, { z: b.ppv_surface.z, ppv_mm_s: b.ppv_surface.ppv_mm_s, arrival_ms: b.ppv_surface.arrival_ms })
  grid('muckpile', b.muckpile, { z_before: b.muckpile.z_before, z_after: b.muckpile.z_after })
  if (isObject(b.damage.blocks)) {
    const k = b.damage.blocks
    same('damage.blocks', { centre: k.centre, size: k.size, ppv_mm_s: k.ppv_mm_s, cls: k.cls, t_break_ms: k.t_break_ms, blasted: k.blasted })
  } else errors.push('"damage.blocks" is missing')
  list('damage.classes', b.damage.classes)
  same('fragmentation', { sizes_m: b.fragmentation.sizes_m, passing: b.fragmentation.passing })
  if (isObject(b.throw.particles)) {
    const p = b.throw.particles
    same('throw.particles', {
      p0: p.p0, v0: p.v0, t0_ms: p.t0_ms, t_land_ms: p.t_land_ms, p_land: p.p_land, p_rest: p.p_rest, size_m: p.size_m, kind: p.kind,
    })
  } else errors.push('"throw.particles" is missing')
  if (!Number.isFinite(b.throw.g)) errors.push('throw.g is not a number')
  list('airblast.rings', b.airblast.rings)
  return errors
}

// ---- timeline slider -----------------------------------------------------------------

/**
 * The detonation takes a fraction of a second, the rock flight seconds: on a linear
 * slider the firing sequence would be a few pixels. The slider therefore gives its first
 * half to [0, split] (the detonation) and its second half to [split, end] (the flight).
 */
export function sliderToTime(u: number, splitMs: number, endMs: number): number {
  const x = u <= 0 ? 0 : u >= 1 ? 1 : u
  if (!(splitMs > 0) || splitMs >= endMs) return x * endMs
  return x <= 0.5 ? x * 2 * splitMs : splitMs + (x - 0.5) * 2 * (endMs - splitMs)
}

export function timeToSlider(tMs: number, splitMs: number, endMs: number): number {
  const t = tMs <= 0 ? 0 : tMs >= endMs ? endMs : tMs
  if (!(endMs > 0)) return 0
  if (!(splitMs > 0) || splitMs >= endMs) return t / endMs
  return t <= splitMs ? (t / splitMs) * 0.5 : 0.5 + ((t - splitMs) / (endMs - splitMs)) * 0.5
}

/** Where the slider switches from the detonation to the flight. */
export function detonationSplit(lastFireMs: number, endMs: number): number {
  return Math.min(endMs, lastFireMs + 60)
}

/**
 * Pure rules of the "site view" of a blast: how one simulated parcel of rock is drawn as
 * several fragments, which fragments are soil, how dust moves, where the equipment stands.
 *
 * The simulation (throw.py) moves a few thousand parcels; each stands for many tonnes. The
 * viewer draws every parcel as a small cluster of fragments around the parcel's own path.
 * This adds no physics: the cluster follows the parcel, it only spreads. What is assumed
 * here is listed where it is used and in research_blast.md part 6.
 */
import type { BlastBundle, Fragmentation } from '../types'

/** Deterministic pseudo-random number in [0, 1) from two integers (no state, same on every frame). */
export function hash01(i: number, k: number): number {
  let h = (Math.imul(i | 0, 0x9e3779b1) ^ Math.imul((k | 0) + 0x7f4a7c15, 0x85ebca6b)) >>> 0
  h ^= h >>> 15
  h = Math.imul(h, 0x2c1b3c6d) >>> 0
  h ^= h >>> 12
  h = Math.imul(h, 0x297a2d39) >>> 0
  h ^= h >>> 15
  return (h >>> 0) / 4294967296
}

/**
 * Fragment size at cumulative passing `u` of the Rosin-Rammler curve the simulation
 * predicted: P(x) = 1 - exp(-(x / xc)^n), so x = xc (-ln(1 - u))^(1 / n).
 */
export function fragmentSize(fragmentation: Pick<Fragmentation, 'xc_m' | 'n'>, u: number): number {
  const p = Math.min(0.999, Math.max(0.001, u))
  return fragmentation.xc_m * (-Math.log(1 - p)) ** (1 / fragmentation.n)
}

/** Thickness of the soil layer drawn at the top of the bench, m. Assumed: no soil log is wired to the blast. */
export const SOIL_THICKNESS_M = 1.5
/** Soil clods are drawn between these sizes, m (assumed). */
export const SOIL_SIZE_M: readonly [number, number] = [0.03, 0.16]

/** Depth of the point `p` below the collar of the nearest hole in plan, m (negative above it). */
export function depthBelowCollars(p: readonly number[], collars: readonly (readonly number[])[]): number {
  let best = Infinity
  let z = p[2]
  for (const c of collars) {
    const d = (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2
    if (d < best) {
      best = d
      z = c[2]
    }
  }
  return z - p[2]
}

/**
 * Offset of a fragment from its parcel, `dt` seconds after the launch of a flight that
 * lasts `flight` seconds: its place inside the parcel (`d0`) plus the drift of its own
 * small velocity difference (`dv`), frozen at landing. The vertical part fades out over
 * the last fifth of the flight, so the fragment meets the ground where the parcel does.
 */
export function fragmentOffset(d0: readonly number[], dv: readonly number[], dt: number, flight: number, out: number[] | Float64Array): void {
  const t = Math.min(Math.max(dt, 0), Math.max(flight, 0))
  out[0] = d0[0] + dv[0] * t
  out[1] = d0[1] + dv[1] * t
  const f = flight > 0 ? t / flight : 1
  const x = Math.min(1, Math.max(0, (f - 0.8) / 0.2))
  out[2] = (d0[2] + dv[2] * t) * (1 - x * x * (3 - 2 * x))
}

// ---- dust --------------------------------------------------------------------------

/** Time constant of the air drag that stops a dust puff, s (assumed). */
export const DUST_DRAG_S = 1.1
/** Wind that carries the dust, m/s east / north (assumed: light breeze). */
export const DUST_WIND: readonly [number, number] = [1.6, 0.7]
/** Slow rise of warm dusty air, m/s (assumed). */
export const DUST_RISE = 0.5

/**
 * Centre of a dust puff `dt` seconds after its birth: a start velocity `v` braked by linear
 * drag (x = v tau (1 - exp(-dt / tau)), the solution of dv/dt = -v / tau), plus the wind and
 * the rise acting with the same lag (dt - tau (1 - exp(-dt / tau))).
 */
export function dustCentre(p0: readonly number[], v: readonly number[], dt: number, out: number[] | Float64Array, tau = DUST_DRAG_S): void {
  const t = Math.max(0, dt)
  const lag = tau * (1 - Math.exp(-t / tau))
  out[0] = p0[0] + v[0] * lag + DUST_WIND[0] * (t - lag)
  out[1] = p0[1] + v[1] * lag + DUST_WIND[1] * (t - lag)
  out[2] = p0[2] + v[2] * lag + DUST_RISE * (t - lag)
}

/** Radius of a puff: it spreads like turbulent diffusion, r = r0 + growth * sqrt(dt). */
export function dustRadius(r0: number, growth: number, dt: number): number {
  return r0 + growth * Math.sqrt(Math.max(0, dt))
}

/** The player runs this long after the last fragment has settled, so the dust can drift, ms. */
export const DUST_TAIL_MS = 7000

// ---- shake and sound ---------------------------------------------------------------

/** Speed of sound in air used for the delay of the rumble, m/s (about 15 degrees C). */
export const SOUND_SPEED_M_S = 340
/** The tremble of one hole dies away with this time constant, ms (chosen for effect). */
export const SHAKE_DECAY_MS = 140

/**
 * Strength (0..1) of the camera tremble at player time `tMs`, `distance` metres from the
 * blast: every hole adds exp(-(t - arrival) / SHAKE_DECAY_MS) from its arrival
 * `t_fire + distance / cp` on; the sum is limited to 1 and weakened with distance beyond
 * 150 m (factor 150 / distance).
 */
export function shakeEnvelope(firesMs: readonly number[], tMs: number, distance: number, cp: number): number {
  const travel = cp > 0 ? (distance / cp) * 1000 : 0
  let sum = 0
  for (const fire of firesMs) {
    const dt = tMs - fire - travel
    if (dt >= 0) sum += Math.exp(-dt / SHAKE_DECAY_MS)
  }
  return Math.min(1, sum) * Math.min(1, 150 / Math.max(distance, 1e-9))
}

// ---- equipment ---------------------------------------------------------------------

/** Equipment stands this far outside the clearance radius, m. */
export const EQUIPMENT_MARGIN_M = 25

export interface EquipmentSite {
  /** World plan position. */
  x: number
  y: number
  /** Heading of the machine's front, radians counter-clockwise from east. */
  heading: number
  /** Swing of the upper structure against the tracks, radians. */
  swing: number
}

/** Centre of the blast in plan: the mean of the hole collars (the muckpile origin without holes). */
export function blastCentre(bundle: Pick<BlastBundle, 'timeline' | 'muckpile'>): [number, number, number] {
  const holes = bundle.timeline.holes
  if (holes.length === 0) return [bundle.muckpile.origin[0], bundle.muckpile.origin[1], 0]
  let x = 0
  let y = 0
  let z = 0
  for (const h of holes) {
    x += h.collar[0]
    y += h.collar[1]
    z += h.collar[2]
  }
  return [x / holes.length, y / holes.length, z / holes.length]
}

/**
 * Where the two excavators wait: behind the blast (opposite to the free face, the side the
 * rock is not thrown to), `EQUIPMENT_MARGIN_M` outside the clearance radius of the flyrock
 * model, 18 m apart, turned towards the blast.
 */
export function equipmentSites(bundle: Pick<BlastBundle, 'timeline' | 'muckpile' | 'flyrock' | 'design'>): EquipmentSite[] {
  const [cx, cy] = blastCentre(bundle)
  const azimuth = typeof bundle.design.face_azimuth === 'number' ? bundle.design.face_azimuth : 180
  const a = (azimuth * Math.PI) / 180
  // Unit vector the face looks along (azimuth clockwise from north), and its left-hand normal.
  const fx = Math.sin(a)
  const fy = Math.cos(a)
  const d = Math.max(0, bundle.flyrock.clearance_m) + EQUIPMENT_MARGIN_M
  const sites: EquipmentSite[] = []
  for (const [side, swing] of [
    [-9, 0.5],
    [9, -0.35],
  ] as const) {
    const x = cx - fx * d - fy * side
    const y = cy - fy * d + fx * side
    sites.push({ x, y, heading: Math.atan2(cy - y, cx - x) + (side > 0 ? 0.25 : -0.4), swing })
  }
  return sites
}

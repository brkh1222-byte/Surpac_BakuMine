/**
 * Blast BakuMine data contract: the result bundle written by `bakumine blast simulate`
 * (plan_blast.md section 3, `bakumine-blast-result` version 1) and the design request the
 * tab sends to `POST /api/v1/blast/simulate`.
 *
 * World coordinates: x = east, y = north, z = up, metres. Times are milliseconds (`*_ms`),
 * PPV is mm/s (`*_mm_s`). Grids are row-major with x outermost: index = ix * ny + iy,
 * node (ix, iy) at origin + (ix, iy) * spacing.
 */

export type Xyz = [number, number, number]
export type Xy = [number, number]

export const BUNDLE_FORMAT = 'bakumine-blast-result'
export const BUNDLE_VERSION = 1

/** Constants of one named model; `calibrated: false` = published typical values. */
export interface ModelInfo {
  calibrated: boolean
  [constant: string]: unknown
}

export interface BlastSummary {
  volume_m3: number
  tonnes: number
  explosive_kg: number
  powder_factor_kg_m3: number
  powder_factor_kg_t: number
  drilled_m: number
  holes: number
  x50_m: number
  uniformity_n: number
  p80_m: number
  oversize_fraction: number
  mic_kg: number
  duration_ms: number
  flyrock_max_m: number
  clearance_m: number
  backbreak_m: number
  models: Record<string, ModelInfo>
}

export interface TimelineHole {
  id: string
  row: number
  t_fire_ms: number
  collar: Xyz
  toe: Xyz
  charge_top: Xyz
  charge_bottom: Xyz
  charge_kg: number
  diameter_m: number
}

export interface BlastTimeline {
  t_end_ms: number
  holes: TimelineHole[]
}

export interface WaveSource {
  hole: string
  t_ms: number
  centre: Xyz
  charge_kg: number
}

/** radius(t) = cp * (t - t_ms); PPV(R) = k * (R / sqrt(charge_kg)) ** -beta [mm/s]. */
export interface BlastWaves {
  cp_m_s: number
  cs_m_s: number
  k: number
  beta: number
  sources: WaveSource[]
}

export interface GridSpec {
  origin: Xy
  spacing: number
  shape: [number, number]
}

export interface PpvSurface extends GridSpec {
  z: (number | null)[]
  ppv_mm_s: (number | null)[]
  arrival_ms: (number | null)[]
}

export interface DamageBlocks {
  centre: Xyz[]
  size: Xyz[]
  ppv_mm_s: number[]
  /** Index into `BlastDamage.classes`. */
  cls: number[]
  /** null = never breaks (intact). */
  t_break_ms: (number | null)[]
  blasted: boolean[]
}

export interface BlastDamage {
  classes: string[]
  thresholds_mm_s: number[]
  blocks: DamageBlocks
}

export interface Fragmentation {
  sizes_m: number[]
  passing: number[]
  x50_m: number
  n: number
  xc_m: number
}

export interface ThrowParticles {
  p0: Xyz[]
  v0: Xyz[]
  t0_ms: number[]
  t_land_ms: number[]
  p_land: Xyz[]
  p_rest: Xyz[]
  size_m: number[]
  /** 0 = muck, 1 = flyrock. */
  kind: number[]
}

/** p(t) = p0 + v0*dt - g/2*dt^2 (z) for t0 <= t <= t_land, then eased to p_rest. */
export interface BlastThrow {
  g: number
  particles: ThrowParticles
}

export interface Muckpile extends GridSpec {
  z_before: (number | null)[]
  z_after: (number | null)[]
}

export interface Flyrock {
  face_burst_m: number
  cratering_m: number
  stemming_ejection_m: number
  max_range_m: number
  clearance_m: number
  safety_factor: number
}

export interface AirblastRing {
  radius_m: number
  pa: number
  dbl: number
}

export interface Airblast {
  k: number
  a: number
  rings: AirblastRing[]
}

export interface MonitorResult {
  name: string
  xyz: Xyz
  distance_m: number
  charge_kg: number
  scaled_distance: number
  ppv_mm_s: number
  limit_mm_s: number | null
  ok: boolean | null
  airblast_dbl: number | null
  airblast_limit_dbl: number | null
  airblast_ok: boolean | null
}

export interface EnvelopeBin {
  /** Start of the window. */
  t_ms: number
  kg: number
  holes: string[]
}

/** Charge fired per `window_ms` window; the tallest bin is the MIC. */
export interface ChargeEnvelope {
  window_ms: number
  bins: EnvelopeBin[]
  mic_kg: number
}

/** `BlastDesign.to_dict()`; the viewer only reads the fields listed here. */
export interface BundleDesign {
  name: string
  bench_height?: number
  floor_z?: number
  burden?: number | null
  spacing?: number | null
  pattern?: string
  face_azimuth?: number
  polygon?: Xy[]
  [key: string]: unknown
}

export interface BlastBundle {
  format: typeof BUNDLE_FORMAT
  version: number
  design: BundleDesign
  summary: BlastSummary
  timeline: BlastTimeline
  waves: BlastWaves
  ppv_surface: PpvSurface
  damage: BlastDamage
  fragmentation: Fragmentation
  throw: BlastThrow
  muckpile: Muckpile
  flyrock: Flyrock
  airblast: Airblast
  monitors: MonitorResult[]
  /** Optional: older bundles do not have it. */
  envelope?: ChargeEnvelope
}

// ---- design request ------------------------------------------------------------------

export type PatternType = 'staggered' | 'square'
export type ExplosiveName = 'ANFO' | 'Emulsion'
export type TimingType = 'row_by_row' | 'echelon' | 'v_cut' | 'hole_by_hole'

export interface DesignMonitor {
  name: string
  xyz: Xyz
  limit_mm_s: number
}

/** Body of `POST /api/v1/blast/simulate`. null = "let the rules of thumb decide". */
export interface DesignParams {
  name: string
  /** Plan outline of the block to blast; null = the demo bench. */
  polygon: Xy[] | null
  bench_height: number
  diameter: number
  burden: number | null
  spacing: number | null
  pattern: PatternType
  /** Direction the free face looks (the throw direction), degrees clockwise from north. */
  face_azimuth: number
  explosive: ExplosiveName
  stemming: number | null
  subdrill: number | null
  timing: { type: TimingType; inter_hole_ms: number; inter_row_ms: number }
  rock: {
    density: number
    ucs_mpa: number
    tensile_mpa: number
    youngs_gpa: number
    poisson: number
    rock_factor: number | null
  }
  monitors: DesignMonitor[]
}

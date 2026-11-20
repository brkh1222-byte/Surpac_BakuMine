/** Request bodies of the dialogs, built from form state (pure, unit-tested). */
import { parsePlane } from './orientation'

export function nums(text: string): number[] | null {
  const out: number[] = []
  for (const item of text.split(/[\s,;]+/).filter(Boolean)) {
    const value = Number(item)
    if (!Number.isFinite(value)) return null
    out.push(value)
  }
  return out
}

export interface ConstraintForm {
  type: 'none' | 'solid' | 'dtm' | 'attribute'
  /** Object id of the solid or DTM. */
  ref: string
  /** inside / outside (solid), above / below (DTM). */
  side: string
  /** DTM: blocks beyond the surface footprint count too. */
  outside: boolean
  expression: string
}

export const EMPTY_CONSTRAINT: ConstraintForm = { type: 'none', ref: '', side: '', outside: false, expression: '' }

export interface StepForm {
  attribute: string
  value: string
  first: ConstraintForm
  /** How the second constraint joins the first. */
  combine: 'and' | 'and not'
  second: ConstraintForm
}

export function constraintDict(c: ConstraintForm): Record<string, unknown> | null {
  switch (c.type) {
    case 'solid':
      return { type: 'solid', solid: c.ref, side: c.side || 'inside' }
    case 'dtm':
      return { type: 'dtm', surface: c.ref, side: c.side || 'above', outside: c.outside }
    case 'attribute':
      return { type: 'attribute', expression: c.expression }
    default:
      return null
  }
}

/** A step's constraint: first, optionally combined with the second (and / and not). */
export function stepConstraint(step: StepForm): Record<string, unknown> | null {
  const a = constraintDict(step.first)
  const b = constraintDict(step.second)
  if (!a || !b) return a ?? b
  return { type: 'and', children: [a, step.combine === 'and not' ? { type: 'not', child: b } : b] }
}

export function buildCodeRequest(
  attributes: { name: string; kind: string; background: string; unit: string }[],
  steps: StepForm[],
  options: { subblock: boolean; merge: boolean; maxLevel: string },
): Record<string, unknown> | string {
  const outSteps: Record<string, unknown>[] = []
  for (const [i, s] of steps.entries()) {
    if (!s.attribute.trim()) return `Step ${i + 1}: give the attribute`
    if (s.value.trim() === '') return `Step ${i + 1}: give the value`
    for (const c of [s.first, s.second]) {
      if ((c.type === 'solid' || c.type === 'dtm') && !c.ref) return `Step ${i + 1}: pick the solid or surface`
      if (c.type === 'attribute' && !c.expression.trim()) return `Step ${i + 1}: give the expression`
    }
    const step: Record<string, unknown> = { attribute: s.attribute.trim(), value: codeValue(s.value) }
    const constraint = stepConstraint(s)
    if (constraint) step.constraint = constraint
    // only geometric steps (solids, surfaces) can be sub-blocked; expression steps never
    const geometric = [s.first, s.second].every((c) => c.type !== 'attribute') && constraint !== null
    if (options.subblock && geometric) {
      step.subblock = true
      if (options.maxLevel.trim()) step.max_level = Number(options.maxLevel)
    }
    outSteps.push(step)
  }
  if (outSteps.length === 0) return 'Add at least one step'
  return {
    attributes: attributes
      .filter((a) => a.name.trim())
      .map((a) => {
        const out: Record<string, unknown> = { name: a.name.trim(), kind: a.kind }
        if (a.kind === 'category' && a.background.trim()) out.background = a.background.trim()
        if (a.kind !== 'category' && a.unit.trim()) out.unit = a.unit.trim()
        return out
      }),
    steps: outSteps,
    merge: options.merge,
  }
}

/** Numbers stay numbers; anything else is text or an expression, as in the CLI's coding file. */
function codeValue(text: string): string | number {
  const t = text.trim()
  const n = Number(t)
  return t !== '' && Number.isFinite(n) ? n : t
}

export interface PassForm {
  name: string
  radiusFactor: string
  minSamples: string
  maxSamples: string
  maxPerHole: string
}

/** How an orientation is entered: rotation angles, or the dip / dip azimuth / pitch of a plane. */
export interface OrientationForm {
  /** 'angles' (bearing, plunge, dip) or 'plane'; a structure may also be 'search' (same as the search ellipsoid). */
  orientation?: 'search' | 'angles' | 'plane'
  angles?: string
  dip?: string
  dipAzimuth?: string
  pitch?: string
}

export interface StructureForm extends OrientationForm {
  type: string
  sill: string
  /** Major, semi-major, minor range; blank = the radii of the search ellipsoid. */
  ranges?: string
}

export const MAX_STRUCTURES = 3
export const STRUCTURE_TYPES = ['spherical', 'exponential', 'gaussian']

/** `{angles}` or `{plane}` of an orientation form, `{}` for 'search'; a string is an error. */
export function orientationSpec(f: OrientationForm, what: string): Record<string, unknown> | string {
  if (f.orientation === 'search') return {}
  if (f.orientation === 'plane') {
    const plane = parsePlane(f.dip ?? '', f.dipAzimuth ?? '', f.pitch ?? '')
    return typeof plane === 'string' ? `${what}: ${plane}` : { plane }
  }
  const angles = nums(f.angles ?? '')
  if (!angles || angles.length !== 3) return `${what}: three numbers (bearing, plunge, dip)`
  return { angles }
}

export interface EstimateForm extends OrientationForm {
  attribute: string
  method: 'nn' | 'idw' | 'ok'
  power: string
  radii: string
  /** Search ellipsoid angles (bearing, plunge, dip); used unless `orientation` is 'plane'. */
  angles: string
  nugget: string
  structures: StructureForm[]
  domainName: string
  blockWhere: string
  sampleColumn: string
  sampleValues: string
  passes: PassForm[]
}

export function buildEstimateSpec(f: EstimateForm): Record<string, unknown> | string {
  if (!f.attribute) return 'Pick the grade attribute'
  const radii = nums(f.radii)
  if (!radii || radii.length !== 3 || radii.some((r) => r <= 0)) return 'Radii: three positive numbers (major, semi-major, minor)'
  const plane = f.orientation === 'plane'
  const orientation = orientationSpec({ ...f, orientation: plane ? 'plane' : 'angles' }, plane ? 'Search ellipsoid' : 'Angles')
  if (typeof orientation === 'string') return orientation
  const spec: Record<string, unknown> = {
    attribute: f.attribute,
    method: f.method,
    ellipsoid: { radii, ...orientation },
  }
  if (f.method === 'idw') spec.power = Number(f.power) || 2
  if (f.method === 'ok') {
    const used = f.structures.map((s, i) => ({ s, label: `Structure ${i + 1}` })).filter(({ s }) => s.sill.trim() !== '')
    if (used.length === 0) return 'Kriging needs at least one variogram structure with a sill'
    if (used.length > MAX_STRUCTURES) return `At most ${MAX_STRUCTURES} variogram structures`
    const structures: Record<string, unknown>[] = []
    for (const { s, label } of used) {
      if (!Number.isFinite(Number(s.sill)) || Number(s.sill) < 0) return 'Sill must be a number'
      const out: Record<string, unknown> = { type: s.type, sill: Number(s.sill) }
      if ((s.ranges ?? '').trim()) {
        const ranges = nums(s.ranges ?? '')
        if (!ranges || ranges.length !== 3 || ranges.some((r) => r <= 0)) return `${label}: ranges are three positive numbers (major, semi-major, minor)`
        out.ranges = ranges
      }
      // a structure follows the search ellipsoid unless it is given its own orientation
      const own = orientationSpec({ ...s, orientation: s.orientation ?? 'search' }, label)
      if (typeof own === 'string') return own
      structures.push({ ...out, ...own })
    }
    spec.variogram = { nugget: Number(f.nugget) || 0, structures }
  }
  const domain: Record<string, unknown> = { name: f.domainName.trim() || 'all' }
  if (f.blockWhere.trim()) domain.block_constraint = { type: 'attribute', expression: f.blockWhere.trim() }
  if (f.sampleColumn.trim() && f.sampleValues.trim()) {
    const values = f.sampleValues.split(',').map((v) => v.trim()).filter(Boolean)
    domain.sample_filter = values.length === 1 ? { column: f.sampleColumn.trim(), equals: values[0] } : { column: f.sampleColumn.trim(), in: values }
  }
  spec.domain = domain
  const passes: Record<string, unknown>[] = []
  for (const [i, p] of f.passes.entries()) {
    const factor = Number(p.radiusFactor)
    if (!(factor > 0)) return `Pass ${i + 1}: radius factor must be positive`
    const pass: Record<string, unknown> = {
      name: p.name.trim() || `pass ${i + 1}`,
      radius_factor: factor,
      min_samples: Number(p.minSamples) || 1,
      max_samples: Number(p.maxSamples) || 16,
    }
    if (p.maxPerHole.trim()) pass.max_per_hole = Number(p.maxPerHole)
    passes.push(pass)
  }
  if (passes.length === 0) return 'Add at least one pass'
  spec.passes = passes
  return spec
}

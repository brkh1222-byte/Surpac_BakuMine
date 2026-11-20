/**
 * Resource classification (`POST /api/v1/ops/models/{id}/classify`, see
 * bakumine/api/ops_classify.py): the request builder of the dialog and the call. The rules
 * are defined in bakumine/core/estimation/classification.py.
 */
import { postJson } from './intercepts'

/** Criterion names of the server, in the order the dialog lists them. */
export const CRITERIA = [
  'max_pass',
  'min_samples',
  'min_holes',
  'max_hole_distance',
  'max_nearest_distance',
  'max_mean_distance',
  'max_kriging_variance',
  'min_slope',
  'min_efficiency',
] as const

const HOLE_CRITERIA = ['min_holes', 'max_hole_distance']

/** Per-class drillhole support, written among the criteria: "min_holes=3, hole_radius=50". */
export const CLASS_SUPPORT = { hole_radius: 'radius', hole_nth: 'nth' } as const

export interface ClassRow {
  /** free text: Measured, Indicated, B, C1, P1, ... */
  label: string
  /** "max_pass=1, min_samples=8"; blank = every estimated block that is left */
  criteria: string
}

export interface ClassifyForm {
  model: string
  attribute: string
  output: string
  /** composites object for the drillhole support; blank = none */
  composites: string
  /** radius of the hole count; blank = do not compute the drillhole support */
  holeRadius: string
  holeNth: string
  /** composites that count for the drillhole support: "domain=MIN" or "domain=MIN|OX"; blank = all */
  holeFilter?: string
  classes: ClassRow[]
  smooth: boolean
  minAgree: string
  maxSame: string
  iterations: string
}

export interface ClassifyResult {
  attribute: string
  output: string
  n_blocks: number
  n_estimated: number
  n_classified: number
  n_unclassified: number
  n_not_estimated: number
  n_smoothed: number
  classes: { label: string; blocks: number; volume: number }[]
  written: string[]
}

export const JORC_CLASSES: ClassRow[] = [
  { label: 'Measured', criteria: 'max_pass=1, min_samples=8, min_holes=3' },
  { label: 'Indicated', criteria: 'max_pass=2, min_holes=2' },
  { label: 'Inferred', criteria: '' },
]

/** "max_pass=1, min_samples = 8" -> {max_pass: 1, min_samples: 8}, or a message. */
export function parseCriteria(text: string): Record<string, number> | string {
  return parseNamed(text, CRITERIA)
}

/** Criteria and the class's own drillhole support ("hole_radius=75, hole_nth=2"), or a message. */
export function parseClass(text: string): { criteria: Record<string, number>; hole_support?: Record<string, number> } | string {
  const all = parseNamed(text, [...CRITERIA, ...Object.keys(CLASS_SUPPORT)])
  if (typeof all === 'string') return all
  const criteria: Record<string, number> = {}
  const support: Record<string, number> = {}
  for (const [name, value] of Object.entries(all)) {
    if (name in CLASS_SUPPORT) support[CLASS_SUPPORT[name as keyof typeof CLASS_SUPPORT]] = value
    else criteria[name] = value
  }
  if ('radius' in support && !(support.radius > 0)) return 'hole_radius must be positive'
  if ('nth' in support && (!Number.isInteger(support.nth) || support.nth < 1)) return 'hole_nth: a whole number from 1'
  return Object.keys(support).length > 0 ? { criteria, hole_support: support } : { criteria }
}

/** "domain=MIN" -> {column, equals}; "domain=MIN|OX" -> {column, in}; blank -> null; else a message. */
export function parseSampleFilter(text: string): Record<string, unknown> | null | string {
  if (!text.trim()) return null
  const match = /^\s*([^=\s]+)\s*=\s*(.+?)\s*$/.exec(text)
  if (!match) return 'Sample filter: write column=value (several values: column=A|B)'
  const values = match[2].split('|').map((v) => v.trim()).filter(Boolean)
  if (values.length === 0) return 'Sample filter: give a value'
  return values.length === 1 ? { column: match[1], equals: values[0] } : { column: match[1], in: values }
}

function parseNamed(text: string, known: readonly string[]): Record<string, number> | string {
  const out: Record<string, number> = {}
  for (const item of text.split(/[,;\n]+/).map((s) => s.trim()).filter(Boolean)) {
    const match = /^([a-z_]+)\s*(?:=|<=|>=|:)\s*(\S+)$/.exec(item)
    if (!match) return `“${item}”: write name=value`
    const [, name, value] = match
    if (!known.includes(name)) return `“${name}” is not a criterion (${CRITERIA.join(', ')})`
    if (name in out) return `“${name}” is given twice`
    const number = Number(value)
    if (!Number.isFinite(number)) return `${name}: “${value}” is not a number`
    out[name] = number
  }
  return out
}

function integer(text: string, what: string, min: number, max: number): number | string {
  const value = Number(text.trim())
  if (text.trim() === '' || !Number.isInteger(value) || value < min || value > max) return `${what}: a whole number ${min}–${max}`
  return value
}

/** The request body, or a message saying what is missing. */
export function buildClassifyRequest(f: ClassifyForm): { params: Record<string, unknown>; composites?: string } | string {
  if (!f.model) return 'Create and estimate a block model first'
  if (!f.attribute) return 'Pick the estimated grade attribute'
  const output = f.output.trim()
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(output)) return 'Class attribute: letters, digits and _, starting with a letter'
  if (output === f.attribute) return 'The class attribute needs another name than the grade'
  const rows = f.classes.filter((c) => c.label.trim() || c.criteria.trim())
  if (rows.length === 0) return 'Add at least one class'
  const categories: { label: string; criteria: Record<string, number>; hole_support?: Record<string, number> }[] = []
  let usesHoles = false
  let ownSupport = false
  for (const row of rows) {
    const label = row.label.trim()
    if (!label) return 'Every class needs a label'
    if (categories.some((c) => c.label === label)) return `Class “${label}” is listed twice`
    const parsed = parseClass(row.criteria)
    if (typeof parsed === 'string') return `${label}: ${parsed}`
    const { criteria } = parsed
    usesHoles ||= HOLE_CRITERIA.some((name) => name in criteria)
    if (parsed.hole_support) {
      if (!f.holeRadius.trim() && !('radius' in parsed.hole_support)) return `${label}: give hole_radius (there is no general hole radius)`
      ownSupport = true
    }
    categories.push({ label, ...parsed })
  }
  const filter = parseSampleFilter(f.holeFilter ?? '')
  if (typeof filter === 'string') return filter
  const params: Record<string, unknown> = { attribute: f.attribute, output, categories }
  const body: { params: Record<string, unknown>; composites?: string } = { params }
  if (f.holeRadius.trim()) {
    const radius = Number(f.holeRadius)
    if (!(radius > 0)) return 'Hole radius must be positive'
    const nth = integer(f.holeNth, 'Nth hole', 1, 1000)
    if (typeof nth === 'string') return nth
    if (!f.composites) return 'Pick the composites for the drillhole support'
    params.hole_support = filter ? { radius, nth, sample_filter: filter } : { radius, nth }
    body.composites = f.composites
  } else if (ownSupport) {
    if (!f.composites) return 'Pick the composites for the drillhole support'
    if (filter) for (const c of categories) if (c.hole_support) (c.hole_support as Record<string, unknown>).sample_filter = filter
    body.composites = f.composites
  } else if (usesHoles && f.composites) {
    return 'Give the hole radius for the drillhole support'
  }
  if (f.smooth) {
    const smoothing: Record<string, number> = {}
    const fields: [string, string, string, number, number][] = [
      ['iterations', f.iterations, 'Smoothing passes', 1, 100],
      ['min_agree', f.minAgree, 'Neighbours that must agree', 1, 6],
      ['max_same', f.maxSame, 'Neighbours of the own class', 0, 5],
    ]
    for (const [key, text, what, min, max] of fields) {
      const value = integer(text, what, min, max)
      if (typeof value === 'string') return value
      smoothing[key] = value
    }
    params.smoothing = smoothing
  }
  return body
}

export const runClassify = (model: string, body: unknown) =>
  postJson<ClassifyResult>(`/models/${encodeURIComponent(model)}/classify`, body)

/** "Measured 1 204 blocks (12.5 %)" per class, of the estimated blocks. */
export function describeClasses(result: ClassifyResult): string[] {
  const share = (n: number) => (result.n_estimated > 0 ? ` (${((100 * n) / result.n_estimated).toFixed(1)} %)` : '')
  const lines = result.classes.map((c) => `${c.label}: ${c.blocks} blocks${share(c.blocks)}`)
  if (result.n_unclassified > 0) lines.push(`no class: ${result.n_unclassified} blocks${share(result.n_unclassified)}`)
  return lines
}

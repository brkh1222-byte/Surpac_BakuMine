/**
 * Geology models in the viewer: the response types of the geology endpoints
 * (bakumine/api/routes_geology.py, operations_geology.py) and the pure logic of the
 * object tree and the Geology panel. No React and no three.js here.
 *
 *   GET  /api/v1/geology/{id}             -> GeologyDetail
 *   GET  /api/v1/geology/{id}/provenance  -> GeologyProvenance
 *   POST /api/v1/ops/geology/code         -> CodeResult
 *
 * The provenance record splits every input of a model into what was measured on site,
 * what the software assumed, and what is still missing. The panel shows it in that order.
 */
import { API_BASE } from '../api/client'

// ---- ids and the object tree ---------------------------------------------------------

export type ChildKind = 'unit' | 'fault' | 'surface'

export interface ChildId {
  /** Id of the geology model. It may itself contain dots. */
  parent: string
  kind: ChildKind
  /** Unit code, fault name, or surface stem ("base_OXIDE"). */
  name: string
}

const CHILD_KINDS: readonly string[] = ['unit', 'fault', 'surface']

export const GROUP_OF_KIND: Record<ChildKind, string> = { unit: 'Units', fault: 'Faults', surface: 'Surfaces' }

/** Order of the groups under a model; a group not listed here comes after them. */
export const GROUP_ORDER: readonly string[] = ['Units', 'Faults', 'Surfaces']

/**
 * Splits `<model>.unit.<CODE>`, `<model>.fault.<NAME>` or `<model>.surface.<stem>`.
 * Read from the right: the last part holds no dot, the model id may. Null for any other id.
 */
export function parseChildId(id: string): ChildId | null {
  const last = id.lastIndexOf('.')
  if (last <= 0 || last === id.length - 1) return null
  const before = id.lastIndexOf('.', last - 1)
  if (before <= 0) return null
  const kind = id.slice(before + 1, last)
  if (!CHILD_KINDS.includes(kind)) return null
  return { parent: id.slice(0, before), kind: kind as ChildKind, name: id.slice(last + 1) }
}

/** What the tree needs of an object: a manifest entry and a registered scene object both fit. */
export interface TreeEntry {
  id: string
  name: string
  category: string
  parent?: string
  group?: string
  /** Position in the project's object list, when known. */
  order?: number
}

export interface GeologyGroup<T> {
  name: string
  members: T[]
}

export interface GeologyNode<T> {
  model: T
  groups: GeologyGroup<T>[]
}

export interface GeologyTree<T> {
  models: GeologyNode<T>[]
  /** Geology meshes whose model is not in the list (listed, so nothing disappears). */
  orphans: T[]
}

/**
 * The geology objects of a list as models with their groups. Parent and group come from
 * the entry; an entry without them falls back to its id. Members are in project order
 * (`order`) when they carry one, else in list order.
 */
export function geologyTree<T extends TreeEntry>(objects: readonly T[]): GeologyTree<T> {
  const mine = objects.filter((o) => o.category === 'geology')
  const ids = new Set(mine.map((o) => o.id))
  const nodes = new Map<string, GeologyNode<T>>()
  const children: { object: T; parent: string; group: string }[] = []
  const orphans: T[] = []
  for (const object of mine) {
    const parsed = parseChildId(object.id)
    // An id that looks like a child is one only when its model is listed too.
    const parent = object.parent ?? (parsed && ids.has(parsed.parent) ? parsed.parent : undefined)
    if (parent === undefined) nodes.set(object.id, { model: object, groups: [] })
    else children.push({ object, parent, group: object.group ?? (parsed ? GROUP_OF_KIND[parsed.kind] : 'Other') })
  }
  for (const { object, parent, group } of children) {
    const node = nodes.get(parent)
    if (!node) {
      orphans.push(object)
      continue
    }
    let entry = node.groups.find((g) => g.name === group)
    if (!entry) node.groups.push((entry = { name: group, members: [] }))
    entry.members.push(object)
  }
  const rank = (name: string) => {
    const i = GROUP_ORDER.indexOf(name)
    return i < 0 ? GROUP_ORDER.length : i
  }
  for (const node of nodes.values()) {
    node.groups.sort((a, b) => rank(a.name) - rank(b.name))
    for (const group of node.groups) group.members.sort((a, b) => (a.order ?? Infinity) - (b.order ?? Infinity) || 0)
  }
  return { models: [...nodes.values()], orphans }
}

/** State of a group's tick box from its members' own toggles. */
export function groupState(members: readonly { visible: boolean }[]): 'all' | 'some' | 'none' {
  const on = members.filter((m) => m.visible).length
  return on === 0 ? 'none' : on === members.length ? 'all' : 'some'
}

// ---- provenance ----------------------------------------------------------------------

export type ProvenanceKind = 'measured' | 'assumed' | 'missing'

export interface ProvenanceEntry {
  item: string
  kind: string
  source: string
  used_for: string
  detail?: string
  stage?: string
  task?: string
  count?: number | null
  value?: number | string | null
}

export interface ProvenanceRecord {
  format?: string
  version?: number
  object?: { kind?: string; name?: string; created?: string; bakumine_version?: string }
  entries: ProvenanceEntry[]
}

export interface GeologyProvenance {
  build: ProvenanceRecord
  /** One record per coded block model, by the model's folder name. */
  coding: Record<string, ProvenanceRecord>
}

/** The three sections, in the order they are shown. */
export const PROVENANCE_SECTIONS: readonly { kind: ProvenanceKind; title: string }[] = [
  { kind: 'measured', title: 'Real data used' },
  { kind: 'assumed', title: 'Assumed by the software' },
  { kind: 'missing', title: 'Inputs you still need to provide' },
]

/** Stages of the record format, in pipeline order. */
export const STAGE_ORDER: readonly string[] = ['units', 'picks', 'surfaces', 'lithology', 'faults', 'coding', 'density', 'topography', 'extent']

/** Filter value of the entries that carry no stage. */
export const NO_STAGE = '(none)'

const STAGE_LABELS: Record<string, string> = {
  units: 'Units',
  picks: 'Picks',
  surfaces: 'Surfaces',
  lithology: 'Rock units',
  faults: 'Faults',
  coding: 'Block coding',
  density: 'Density',
  topography: 'Topography',
  extent: 'Extent and grid',
  [NO_STAGE]: 'Other defaults',
}

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}

function stageOf(entry: ProvenanceEntry): string {
  return entry.stage ? entry.stage : NO_STAGE
}

/**
 * Stages that occur in the records, in pipeline order; unknown stages after them, the
 * entries without a stage last. Empty when no entry has a stage (nothing to filter by).
 */
export function provenanceStages(records: readonly (ProvenanceRecord | null | undefined)[]): string[] {
  const seen = new Set<string>()
  for (const record of records) for (const entry of record?.entries ?? []) seen.add(stageOf(entry))
  if (seen.size === 0 || (seen.size === 1 && seen.has(NO_STAGE))) return []
  const known = STAGE_ORDER.filter((s) => seen.has(s))
  const unknown = [...seen].filter((s) => s !== NO_STAGE && !STAGE_ORDER.includes(s)).sort()
  return [...known, ...unknown, ...(seen.has(NO_STAGE) ? [NO_STAGE] : [])]
}

export interface ProvenanceSection {
  kind: ProvenanceKind
  title: string
  /** Entries of this kind in the whole record. */
  total: number
  /** Entries of this kind that pass the stage filter, in record order. */
  entries: ProvenanceEntry[]
}

export interface ProvenanceView {
  sections: ProvenanceSection[]
  /** Entries whose kind is none of the three: counted, so that none is dropped silently. */
  otherKinds: number
}

/** The record as its three sections; `stage` null = every stage. */
export function provenanceView(record: ProvenanceRecord | null | undefined, stage: string | null = null): ProvenanceView {
  const entries = record?.entries ?? []
  const sections = PROVENANCE_SECTIONS.map(({ kind, title }) => {
    const all = entries.filter((e) => e.kind === kind)
    return { kind, title, total: all.length, entries: stage === null ? all : all.filter((e) => stageOf(e) === stage) }
  })
  const kinds: readonly string[] = PROVENANCE_SECTIONS.map((s) => s.kind)
  return { sections, otherKinds: entries.filter((e) => !kinds.includes(e.kind)).length }
}

export function provenanceCounts(record: ProvenanceRecord | null | undefined): Record<ProvenanceKind, number> {
  const [measured, assumed, missing] = provenanceView(record).sections.map((s) => s.total)
  return { measured, assumed, missing }
}

/** "Real data used (34)", or "Real data used (10 of 34)" while a stage filter is on. */
export function sectionHeading(section: ProvenanceSection, filtered: boolean): string {
  const count = filtered ? `${section.entries.length} of ${section.total}` : String(section.total)
  return `${section.title} (${count})`
}

/** The value of an entry as text; null when it has none. */
export function entryValue(entry: ProvenanceEntry): string | null {
  const { value } = entry
  if (value === null || value === undefined || value === '') return null
  if (typeof value === 'number') return Number.isFinite(value) ? formatNumber(value, Number.isInteger(value) ? 0 : 2) : null
  return String(value)
}

/** The count of an entry as text; '' when it has none (zero is a count and is shown). */
export function entryCount(entry: ProvenanceEntry): string {
  return typeof entry.count === 'number' && Number.isFinite(entry.count) ? formatNumber(entry.count, 0) : ''
}

// ---- the model summary ---------------------------------------------------------------

export interface GeologyGrid {
  origin: number[]
  spacing: number[]
  shape: number[]
}

export interface GeologyUnit {
  code: string
  name: string
  kind: string
  colour: string | null
  /** Child id of the unit's mesh; null for a unit without one (the cover). */
  mesh: string | null
}

export interface GeologyFault {
  name: string
  source: string
  dip: number | null
  dip_azimuth: number | null
  blocks: string[]
  misfit?: Record<string, unknown> | null
  mesh: string | null
}

export interface SurfaceDiagnostics {
  holes?: number
  used?: Record<string, number>
  holes_without_information?: number
  conflicts?: number
  method?: string
  depth_min?: number | null
  depth_max?: number | null
  depth_mean?: number | null
  extrapolated_share?: number | null
  n_nested?: number
}

export interface GeologySurface {
  /** "base:OXIDE" */
  name: string
  role: string
  code: string
  colour: string | null
  mesh: string | null
  diagnostics?: SurfaceDiagnostics | null
}

export interface UnitVolume {
  volume: number | null
  inside_hull: number | null
  extrapolated: number | null
  extrapolated_share: number | null
}

export interface VolumeDiagnostics {
  method?: string
  cell_volume?: number
  hull?: boolean
  units: Record<string, UnitVolume>
}

export interface ThrowResult {
  fault: string
  upper: string
  lower: string
  blocks?: string[]
  throw: number | null
  spread?: number | null
  /** "measured" or "assumed", as the build decided. */
  kind?: string
  note?: string
  at?: string
  pooled?: string[]
}

export interface ResidualDiagnostics {
  points?: number
  step?: number
  /** Share of the points along the holes where the model gives the logged unit, 0..1. */
  reproduced?: number | null
  note?: string
  /** Not in the record today; shown when the build gives one. */
  kind?: string
  units?: Record<string, { points: number; reproduced: number | null }>
}

export interface GeologyDiagnostics {
  grid?: { nodes?: number; resolution_requested?: number | null; coarsened?: boolean }
  inputs?: {
    holes?: number
    holes_with_trace?: number
    holes_logged?: number
    lithology_intervals?: number
    lithology_intervals_mapped?: number
    rock_intervals?: number
  }
  throw?: ThrowResult[]
  residuals?: ResidualDiagnostics | null
  volumes?: VolumeDiagnostics | null
  pooled_blocks?: unknown[]
  cross_validation?: { run?: boolean; note?: string }
}

export interface GeologyDetail {
  id: string
  name: string
  model_name?: string
  created?: string
  grid: GeologyGrid
  bounds?: { min: number[]; max: number[] }
  layers: { lith?: string[]; weathering?: string[]; fault_block?: string[] }
  /** Fault blocks. */
  blocks: string[]
  pooled_blocks: unknown[]
  units: GeologyUnit[]
  faults: GeologyFault[]
  surfaces: GeologySurface[]
  max_extrapolation?: number | null
  diagnostics: GeologyDiagnostics | null
  coded_models?: string[]
}

/** Numbers as in the report table: thousands separated by spaces, null as an en dash. */
export function formatNumber(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–'
  return value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).replace(/,/g, ' ')
}

/** A share 0..1 as a percentage with one decimal; null as an en dash. */
export function formatShare(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–'
  return `${(value * 100).toFixed(1)} %`
}

/** "81 x 81 x 63 nodes at 5 m" (three spacings when they differ). */
export function gridText(grid: GeologyGrid | null | undefined): string {
  if (!grid || !Array.isArray(grid.shape) || grid.shape.length === 0) return '–'
  const spacing = grid.spacing ?? []
  const same = spacing.length > 0 && spacing.every((s) => s === spacing[0])
  const step = spacing.length === 0 ? '' : ` at ${same ? spacing[0] : spacing.join(' x ')} m`
  return `${grid.shape.join(' x ')} nodes${step}`
}

/** A pooled-blocks entry as text, whatever its shape (names, or lists of names). */
export function pooledText(pooled: readonly unknown[] | null | undefined): string {
  if (!pooled || pooled.length === 0) return 'none'
  return pooled
    .map((p) => (Array.isArray(p) ? p.join(' + ') : typeof p === 'object' && p !== null ? Object.values(p).flat().join(' + ') : String(p)))
    .join('; ')
}

export interface SummaryRow {
  label: string
  text: string
}

export function summaryRows(detail: GeologyDetail): SummaryRow[] {
  const inputs = detail.diagnostics?.inputs
  const rows: SummaryRow[] = [
    { label: 'Grid', text: gridText(detail.grid) + (detail.diagnostics?.grid?.coarsened ? ' (coarsened)' : '') },
    { label: 'Units', text: `${detail.units.length}: ${detail.units.map((u) => u.code).join(', ')}` },
    { label: 'Fault blocks', text: detail.blocks.length > 0 ? `${detail.blocks.length}: ${detail.blocks.join(', ')}` : 'none' },
    { label: 'Pooled blocks', text: pooledText(detail.pooled_blocks) },
  ]
  if (detail.faults.length > 0) rows.push({ label: 'Faults', text: detail.faults.map((f) => f.name).join(', ') })
  if (detail.surfaces.length > 0) rows.push({ label: 'Surfaces', text: detail.surfaces.map((s) => s.name).join(', ') })
  if (inputs?.holes !== undefined) {
    const intervals = inputs.lithology_intervals !== undefined ? `, ${formatNumber(inputs.lithology_intervals)} logged intervals` : ''
    rows.push({ label: 'Drilling', text: `${formatNumber(inputs.holes)} holes${intervals}` })
  }
  return rows
}

// ---- volumes, throw, residuals -------------------------------------------------------

export interface VolumeRow {
  unit: string
  volume: number | null
  insideHull: number | null
  extrapolated: number | null
  share: number | null
}

const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/**
 * One row per unit, in the order of `order` (the lith layer), then any other unit of the
 * record. A figure the record does not give stays null: it is never filled in from the
 * other columns.
 */
export function volumeRows(volumes: VolumeDiagnostics | null | undefined, order: readonly string[] = []): VolumeRow[] {
  const units = volumes?.units ?? {}
  const codes = [...order.filter((c) => c in units), ...Object.keys(units).filter((c) => !order.includes(c))]
  return codes.map((unit) => ({
    unit,
    volume: numberOrNull(units[unit].volume),
    insideHull: numberOrNull(units[unit].inside_hull),
    extrapolated: numberOrNull(units[unit].extrapolated),
    share: numberOrNull(units[unit].extrapolated_share),
  }))
}

export const VOLUME_COLUMNS = ['Unit', 'Whole box, m3', 'Inside drilled hull, m3', 'Extrapolated, m3', 'Extrapolated share'] as const

/** The cells of a volume row, in the order of `VOLUME_COLUMNS`. */
export function volumeCells(row: VolumeRow): string[] {
  return [row.unit, formatNumber(row.volume), formatNumber(row.insideHull), formatNumber(row.extrapolated), formatShare(row.share)]
}

export interface ThrowRow {
  fault: string
  contact: string
  /** "26.2 ± 1.0 m" */
  throwText: string
  /** "measured", "assumed", or "not stated" when the record gives no kind. */
  basis: string
  note: string
}

export function throwRows(results: readonly ThrowResult[] | null | undefined): ThrowRow[] {
  return (results ?? []).map((t) => {
    const value = numberOrNull(t.throw)
    const spread = numberOrNull(t.spread)
    return {
      fault: t.fault,
      contact: `${t.upper} / ${t.lower}`,
      throwText: value === null ? '–' : `${value.toFixed(1)}${spread === null ? '' : ` ± ${spread.toFixed(1)}`} m`,
      basis: t.kind === 'measured' || t.kind === 'assumed' ? t.kind : 'not stated',
      note: t.note ?? '',
    }
  })
}

export interface ResidualRow {
  unit: string
  points: number | null
  reproduced: number | null
}

export function residualRows(residuals: ResidualDiagnostics | null | undefined, order: readonly string[] = []): ResidualRow[] {
  const units = residuals?.units ?? {}
  const codes = [...order.filter((c) => c in units), ...Object.keys(units).filter((c) => !order.includes(c))]
  return codes.map((unit) => ({ unit, points: numberOrNull(units[unit].points), reproduced: numberOrNull(units[unit].reproduced) }))
}

// ---- block coding --------------------------------------------------------------------

export interface CodeRequest {
  model: string
  geology: string
}

export interface CodeResult {
  model: string
  geology: string
  n_blocks: number
  attributes: Record<string, string>
  skipped: Record<string, string>
  n_split: number
  n_merged: number
  volumes: { layer: string; category: string; volume: number | null }[]
  tonnage: { unit: string; blocks: number; volume: number | null; tonnes: number | null; density: number | null }[] | null
  provenance?: Partial<Record<ProvenanceKind, number>>
}

// ---- requests ------------------------------------------------------------------------

type FetchFn = typeof fetch

async function json<T>(url: string, init: RequestInit | undefined, fetchFn: FetchFn): Promise<T> {
  let response: Response
  try {
    response = await fetchFn(url, init)
  } catch {
    throw new Error('The API is not reachable. Start it with `make dev`.')
  }
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    const detail = (body as { detail?: unknown } | null)?.detail
    // A refused request (400) says which field is wrong: show the server's text as it is.
    throw new Error(typeof detail === 'string' && detail ? detail : `Request failed (${response.status})`)
  }
  return (await response.json()) as T
}

export function fetchGeology(id: string, fetchFn: FetchFn = fetch): Promise<GeologyDetail> {
  return json(`${API_BASE}/geology/${encodeURIComponent(id)}`, undefined, fetchFn)
}

export function fetchGeologyProvenance(id: string, fetchFn: FetchFn = fetch): Promise<GeologyProvenance> {
  return json(`${API_BASE}/geology/${encodeURIComponent(id)}/provenance`, undefined, fetchFn)
}

/** Codes the block model with the layers of the geology model (changes the model; undo is in the job log). */
export function codeBlockModel(body: CodeRequest, fetchFn: FetchFn = fetch): Promise<CodeResult> {
  return json(
    `${API_BASE}/ops/geology/code`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    fetchFn,
  )
}

// ---- taking the composition off again -------------------------------------------------

export interface UncodeResult {
  model: string
  /** Attributes dropped from the block model. */
  removed: string[]
  /** Attribute the coding wrote that is still in the model -> why. */
  kept: Record<string, string>
  n_merged: number
  n_blocks: number
  /** Geology models whose record of the coding was deleted with it. */
  geology: string[]
}

export interface RemoveGeologyResult {
  removed: string
  name: string
  path: string
  /** False when the project keeps no undo copies: the model is gone for good. */
  undoable: boolean
  /** Block models coded from it; they keep their attributes. */
  coded_models: string[]
}

function post<T>(path: string, body: unknown, fetchFn: FetchFn): Promise<T> {
  return json(`${API_BASE}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, fetchFn)
}

/** Drops the attributes the geology coding wrote to the block model (undo is in the job log). */
export function uncodeBlockModel(model: string, fetchFn: FetchFn = fetch): Promise<UncodeResult> {
  return post('/ops/geology/uncode', { model }, fetchFn)
}

/** Takes the geology model and its files out of the project (undo is in the job log). */
export function removeGeology(geology: string, fetchFn: FetchFn = fetch): Promise<RemoveGeologyResult> {
  return post('/ops/geology/remove', { geology }, fetchFn)
}

/** What the confirmation says before a geology model is removed. */
export function removeGeologyWarning(name: string, codedModels: readonly string[]): string[] {
  const lines = [`Remove "${name}" from the project? Its units, faults and surfaces leave the scene and its files are deleted.`]
  if (codedModels.length > 0) {
    lines.push(
      `Block models coded from it (${codedModels.join(', ')}) keep their attributes, without the dictionary colours and names. Use "Remove coding" to take those off too.`,
    )
  }
  lines.push('Undo in the job log brings the model back while its undo copy is kept.')
  return lines
}

/** One line about a finished removal of a geology model. */
export function removeGeologyNote(result: RemoveGeologyResult): string {
  return result.undoable
    ? `Removed "${result.name}". Undo is in the job log.`
    : `Removed "${result.name}". The project keeps no undo copies: it cannot be brought back.`
}

/** What a finished "Remove coding" says: the attributes dropped, the blocks, the ones kept. */
export function uncodeNotes(result: UncodeResult): string[] {
  const lines = [
    `Removed ${result.removed.join(', ') || 'nothing'}. ${formatNumber(result.n_blocks)} blocks (${formatNumber(result.n_merged)} sets of sub-blocks merged back). Undo is in the job log.`,
  ]
  for (const [name, reason] of Object.entries(result.kept)) lines.push(`Kept ${name}: ${reason}.`)
  return lines
}

/** A block model of the project summary with the attributes its geology coding wrote. */
export interface CodedModelInfo {
  geology_coding?: { geology: string | null; attributes: { name: string; role: string; created: boolean }[] } | null
}

/** Why "Remove coding" cannot run on a model, or null when it can. */
export function uncodeBlocked(model: CodedModelInfo | null | undefined): string | null {
  if (!model) return 'Choose a block model.'
  const coding = model.geology_coding
  if (!coding) return 'This block model has no geology coding on record.'
  if (!coding.attributes.some((a) => a.created)) return 'The coding made no attribute of its own in this model: nothing to remove.'
  return null
}

/** What "Remove coding" would do to a model: the attributes it drops and the ones it keeps. */
export function uncodePreview(model: CodedModelInfo | null | undefined): string | null {
  const coding = model?.geology_coding
  if (!coding || uncodeBlocked(model)) return null
  const dropped = coding.attributes.filter((a) => a.created).map((a) => a.name)
  const kept = coding.attributes.filter((a) => !a.created).map((a) => a.name)
  return `Remove coding drops ${dropped.join(', ')}${kept.length ? ` and keeps ${kept.join(', ')} (the model had ${kept.length > 1 ? 'them' : 'it'} before)` : ''}.`
}

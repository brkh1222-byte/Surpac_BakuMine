/**
 * Pure helpers of the Import panel: which sheets a wildcard selects, and the request body
 * of a source (only what the user set; everything else is detected by the server).
 */
import type { DroppedRows, DuplicateReport, HoleFilter, HoleFilterReport, HoleIdRule, MergeReport, OverlapReport, SampleDepthCheck, SourceReport, SourceSpec } from './api'

/** What the panel keeps per uploaded file. Empty strings mean "detect". */
export interface SourceForm {
  sheets: string[] | null
  headerRow: string
  headerRows: string
  holeFromSheet: boolean
  sampleDepth: boolean
  sampleLength: string
  samplePatterns: string
  /** how far below the hole bottom the derived depths may end ('' = 5) */
  sampleTolerance: string
  qc: 'separate' | 'off'
  /** the column map once the user edited it; null = as suggested */
  map: Record<string, string> | null
  previewSheet: string | null
  /** what this file does differently from the panel's settings ('' = as set there) */
  dip: '' | 'auto' | 'negative_down' | 'positive_down'
  lengthUnit: '' | 'm' | 'ft' | 'ftUS'
  order: '' | 'auto' | 'xyz' | 'yxz'
  crs: string
}

export const emptyForm = (): SourceForm => ({
  sheets: null,
  headerRow: '',
  headerRows: '',
  holeFromSheet: false,
  sampleDepth: false,
  sampleLength: '1',
  samplePatterns: '',
  sampleTolerance: '',
  qc: 'separate',
  map: null,
  previewSheet: null,
  dip: '',
  lengthUnit: '',
  order: '',
  crs: '',
})

/** Sheet names matching a wildcard pattern (`*` any text, `?` one character), in order. */
export function matchSheets(names: string[], pattern: string): string[] {
  const text = pattern.trim()
  if (!text) return []
  const escaped = text.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.')
  const re = new RegExp(`^${escaped}$`)
  return names.filter((n) => re.test(n))
}

/** A typed number that is 0 or more; null when the field is empty or not such a number. */
export function nonNegative(text: string): number | null {
  if (text.trim() === '') return null
  const value = Number(text.trim().replace(',', '.'))
  return Number.isFinite(value) && value >= 0 ? value : null
}

/** The `merge` of an import request; the interval tolerance only when it was typed. */
export function mergeRule(precedence: string, ignore: string, duplicates: string, intervalTolerance: string): Record<string, unknown> {
  const near = nonNegative(intervalTolerance)
  return { precedence, ignore_columns: holeList(ignore), duplicate_intervals: duplicates, ...(near !== null ? { interval_tolerance: near } : {}) }
}

/** "; 3 intervals written differently in the other file (from / to within 0.5) taken as one" */
export function nearText(report: MergeReport): string {
  if (!report.near_intervals) return ''
  return `; ${report.near_intervals} interval${report.near_intervals > 1 ? 's' : ''} written differently in the other file (from / to within ${report.interval_tolerance ?? 0.5}) taken as one`
}

/** "assay: 2 intervals still overlap in 1 hole, e.g. SYN-01 10–11.3 (a.csv) with 10–11 (b.csv); ..." */
export function overlapsText(table: string, report: OverlapReport): string {
  const first = report.rows[0]
  const example = first ? `, e.g. ${first.hole_id} ${first.depth_from}–${first.depth_to} (${first.source}) with ${first.overlaps_from}–${first.overlaps_to} (${first.overlaps_source})` : ''
  return `${table}: ${report.count} pair${report.count > 1 ? 's' : ''} of intervals still overlap in ${report.holes} hole${report.holes > 1 ? 's' : ''}${example}; listed in ${report.listing}. Mineralised intervals refuse overlapping samples: raise the interval tolerance or fix the files`
}

/** The recipe source a form describes. `interval`: the table has from / to. */
export function sourceSpec(form: SourceForm, interval: boolean, forPreview = false): SourceSpec {
  const spec: SourceSpec = {}
  if (form.sheets) spec.sheets = form.sheets
  const row = Number.parseInt(form.headerRow, 10)
  if (form.headerRow.trim() !== '' && Number.isFinite(row) && row >= 0) spec.header_row = row
  if (form.headerRows === '1' || form.headerRows === '2') spec.header_rows = Number(form.headerRows)
  if (form.holeFromSheet) spec.hole_id_from_sheet = true
  if (form.map) spec.map = form.map
  if (interval) {
    if (form.sampleDepth) {
      const length = Number.parseFloat(form.sampleLength)
      const patterns = form.samplePatterns.split('\n').map((p) => p.trim()).filter(Boolean)
      const slack = nonNegative(form.sampleTolerance)
      spec.sample_depth = { length: Number.isFinite(length) && length > 0 ? length : 1, ...(patterns.length ? { patterns } : {}), ...(slack !== null ? { depth_tolerance: slack } : {}) }
    }
    if (form.qc === 'off') spec.qc = { mode: 'off' }
  }
  const options: Record<string, string> = {}
  if (form.dip && !interval) options.dip_convention = form.dip
  if (form.lengthUnit) options.length_unit = form.lengthUnit
  if (form.order && !interval) options.coordinate_order = form.order
  if (form.crs.trim() && !interval) options.crs = form.crs.trim()
  if (Object.keys(options).length) spec.options = options
  if (forPreview && form.previewSheet) spec.preview_sheet = form.previewSheet
  return spec
}

/** "collar: 1 total row, 2 without a hole id, 1 collar without coordinates (SYN-09)" */
export function droppedText(table: string, dropped: DroppedRows): string {
  const parts: string[] = []
  if (dropped.total_rows) parts.push(`${dropped.total_rows} total / summary row${dropped.total_rows > 1 ? 's' : ''}`)
  if (dropped.rows_without_hole_id) parts.push(`${dropped.rows_without_hole_id} without a hole id`)
  if (dropped.collars_without_coordinates) {
    const holes = dropped.holes_without_coordinates ?? []
    parts.push(`${dropped.collars_without_coordinates} without coordinates${holes.length ? ` (${holes.slice(0, 8).join(', ')}${holes.length > 8 ? ', …' : ''})` : ''}`)
  }
  return `${table}: rows dropped: ${parts.join(', ')}`
}

/** " (au 3, ag 1); not counted: sample_id 2140; full list in conflicts_assay.csv" */
export function conflictsText(report: MergeReport): string {
  const counts = (values?: Record<string, number>) => Object.entries(values ?? {}).map(([c, n]) => `${c} ${n}`).join(', ')
  const by = counts(report.by_column)
  const apart = counts(report.informational)
  return `${by ? ` (${by})` : ''}${apart ? `; not counted (informational): ${apart}` : ''}${report.listing ? `; every difference is in ${report.listing}` : ''}`
}

/** "; NOT imported, deeper than the hole: SYN-04 (to 4020, hole 250)" */
export function beyondText(check: SampleDepthCheck): string {
  const beyond = check.beyond_hole ?? []
  if (!beyond.length) return ''
  return `; NOT imported, deeper than the hole: ${beyond.slice(0, 6).map((b) => `${b.hole_id} (to ${b.derived_to}, hole ${b.max_depth})`).join(', ')}${beyond.length > 6 ? ', …' : ''}`
}

/** What the sample-number rule was refused for: the file's own intervals, or another file's. */
export function contradictedText(check: SampleDepthCheck): string {
  const names = (holes: { hole_id: string }[]) => `${holes.slice(0, 6).map((h) => h.hole_id).join(', ')}${holes.length > 6 ? ', …' : ''}`
  let text = ''
  const own = check.contradicted ?? []
  if (own.length) {
    const e = own[0].example
    text += `; NOT given depths (${check.contradicted_rows ?? 0} rows): the file's own from / to show that the sample numbers are not depths in ${names(own)} (sample ${e.sample_id} would end at ${e.derived_to}, the file says ${e.depth_to})`
  }
  const other = check.contradicts_sources ?? []
  if (other.length) {
    const e = other[0].example
    text += `; NOT imported (${check.contradicts_sources_rows ?? 0} rows): the depths read from sample numbers overlap the intervals of another file in ${names(other)} (sample ${e.sample_id} read as ${e.depth_from}–${e.depth_to}, ${e.other_file} has ${e.other_from}–${e.other_to}); turn the rule off for this file or give a pattern`
  }
  return text
}

/** "assay: 12 intervals in 3 holes were there more than once (keep first): 12 rows removed ..." */
export function duplicatesText(table: string, report: DuplicateReport): string {
  const first = report.examples[0]
  const example = first ? `, e.g. ${first.hole_id} ${first.depth_from}–${first.depth_to}` : ''
  return `${table}: ${report.intervals} interval${report.intervals > 1 ? 's' : ''} in ${report.holes} hole${report.holes > 1 ? 's' : ''} more than once (${report.rule.replace(/_/g, ' ')}): ${report.rows_removed} row${report.rows_removed > 1 ? 's' : ''} removed${example}; listed in ${report.listing}`
}

/** "gis.xlsx: dip convention positive_down, length unit ft" (what was decided for a file) */
export function sourceText(entry: SourceReport): string {
  const said = Object.entries(entry).filter(([k]) => k !== 'file').map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`)
  return `${entry.file}: ${said.join(', ')}`
}

/** Hole names / wildcard patterns typed with commas or on lines of their own. */
export function holeList(text: string): string[] {
  return [...new Set(text.split(/[,\n;]/).map((p) => p.trim()).filter(Boolean))]
}

/** The `holes` of an import request; null when nothing is filtered. */
export function holeFilter(include: string, exclude: string): HoleFilter | null {
  const filter: HoleFilter = {}
  if (holeList(include).length) filter.include = holeList(include)
  if (holeList(exclude).length) filter.exclude = holeList(exclude)
  return filter.include || filter.exclude ? filter : null
}

export function holeFilterText(report: HoleFilterReport): string {
  const kept = Object.entries(report.tables).map(([t, c]) => `${t} ${c.rows_kept} of ${c.rows_kept + c.rows_dropped} rows`).join(', ')
  const unmatched = report.unmatched.length ? `; matched no hole: ${report.unmatched.join(', ')}` : ''
  return `Hole filter: kept ${kept}${unmatched}`
}

export function holeIdRule(pattern: string, template: string): HoleIdRule | null {
  return pattern.trim() ? { pattern: pattern.trim(), template: template.trim() || '{0}' } : null
}

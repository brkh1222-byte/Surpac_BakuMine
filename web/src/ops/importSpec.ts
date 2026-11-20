/**
 * Pure helpers of the Import panel: which sheets a wildcard selects, and the request body
 * of a source (only what the user set; everything else is detected by the server).
 */
import type { DroppedRows, DuplicateReport, HoleFilter, HoleFilterReport, HoleIdRule, MergeReport, SourceReport, SourceSpec } from './api'

/** What the panel keeps per uploaded file. Empty strings mean "detect". */
export interface SourceForm {
  sheets: string[] | null
  headerRow: string
  headerRows: string
  holeFromSheet: boolean
  sampleDepth: boolean
  sampleLength: string
  samplePatterns: string
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
      spec.sample_depth = { length: Number.isFinite(length) && length > 0 ? length : 1, ...(patterns.length ? { patterns } : {}) }
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

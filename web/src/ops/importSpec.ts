/**
 * Pure helpers of the Import panel: which sheets a wildcard selects, and the request body
 * of a source (only what the user set; everything else is detected by the server).
 */
import type { HoleIdRule, SourceSpec } from './api'

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
  if (forPreview && form.previewSheet) spec.preview_sheet = form.previewSheet
  return spec
}

export function holeIdRule(pattern: string, template: string): HoleIdRule | null {
  return pattern.trim() ? { pattern: pattern.trim(), template: template.trim() || '{0}' } : null
}

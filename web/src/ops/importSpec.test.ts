import { describe, expect, it } from 'vitest'
import { beyondText, conflictsText, contradictedText, droppedText, duplicatesText, emptyForm, holeFilter, holeFilterText, holeIdRule, matchSheets, mergeRule, nearText, nonNegative, overlapsText, sourceSpec, sourceText } from './importSpec'

describe('import panel helpers', () => {
  it('selects sheets by wildcard, in workbook order', () => {
    const names = ['SYN_01', 'SYN-002', 'Summary', 'SYN_3', 'S.1']
    expect(matchSheets(names, 'SYN*')).toEqual(['SYN_01', 'SYN-002', 'SYN_3'])
    expect(matchSheets(names, 'SYN_??')).toEqual(['SYN_01'])
    expect(matchSheets(names, 'S.1')).toEqual(['S.1'])
    expect(matchSheets(names, '*')).toEqual(names)
    expect(matchSheets(names, '  ')).toEqual([])
  })

  it('sends only what the user set', () => {
    expect(sourceSpec(emptyForm(), true)).toEqual({})
    const form = { ...emptyForm(), sheets: ['A', 'B'], headerRow: '3', headerRows: '2', holeFromSheet: true, sampleDepth: true, sampleLength: '1.5', samplePatterns: ' a(?P<to>\\d+) \n\n', qc: 'off' as const, previewSheet: 'B' }
    expect(sourceSpec(form, true)).toEqual({
      sheets: ['A', 'B'],
      header_row: 3,
      header_rows: 2,
      hole_id_from_sheet: true,
      sample_depth: { length: 1.5, patterns: ['a(?P<to>\\d+)'] },
      qc: { mode: 'off' },
    })
    // a collar table has no sample rule or QC; the preview may name the sheet to show
    expect(sourceSpec(form, false, true)).toEqual({ sheets: ['A', 'B'], header_row: 3, header_rows: 2, hole_id_from_sheet: true, preview_sheet: 'B' })
    expect(sourceSpec({ ...emptyForm(), headerRow: '0', sampleDepth: true, sampleLength: 'x' }, true)).toEqual({ header_row: 0, sample_depth: { length: 1 } })
  })

  it('sends the tolerances only when they were typed', () => {
    expect(nonNegative('')).toBeNull()
    expect(nonNegative(' 0 ')).toBe(0)
    expect(nonNegative('0,3')).toBe(0.3)
    expect(nonNegative('-1')).toBeNull()
    expect(nonNegative('x')).toBeNull()
    expect(mergeRule('first', 'sample_id, lab', 'keep_first', '')).toEqual({ precedence: 'first', ignore_columns: ['sample_id', 'lab'], duplicate_intervals: 'keep_first' })
    expect(mergeRule('last', '', 'mean', '0')).toEqual({ precedence: 'last', ignore_columns: [], duplicate_intervals: 'mean', interval_tolerance: 0 })
    expect(mergeRule('first', '', 'error', '1.5').interval_tolerance).toBe(1.5)
    // the slack of the sample-number rule belongs to one file
    expect(sourceSpec({ ...emptyForm(), sampleDepth: true, sampleTolerance: '2.5' }, true)).toEqual({ sample_depth: { length: 1, depth_tolerance: 2.5 } })
    expect(sourceSpec({ ...emptyForm(), sampleDepth: false, sampleTolerance: '2.5' }, true)).toEqual({})
  })

  it('says which intervals were taken as one and which still overlap', () => {
    const merged = { precedence: 'first', keys: [], merged: 4, filled: 0, conflicts: 0, rows: [] }
    expect(nearText(merged)).toBe('')
    expect(nearText({ ...merged, near_intervals: 2, interval_tolerance: 0.5 })).toBe('; 2 intervals written differently in the other file (from / to within 0.5) taken as one')
    const row = { hole_id: 'SYN-01', depth_from: 10, depth_to: 11.3, source: 'a.csv', overlaps_from: 10, overlaps_to: 11, overlaps_source: 'b.csv', overlap: 1 }
    expect(overlapsText('assay', { count: 1, holes: 1, listing: 'overlaps_assay.csv', rows: [row] }))
      .toBe('assay: 1 pair of intervals still overlap in 1 hole, e.g. SYN-01 10–11.3 (a.csv) with 10–11 (b.csv); listed in overlaps_assay.csv. Mineralised intervals refuse overlapping samples: raise the interval tolerance or fix the files')
  })

  it('says why the sample-number rule was refused', () => {
    const check = { rows: 10, duplicates: 0, backwards: 0, gaps: 0, overlaps: 0, unmatched: 0, unmatched_examples: [] }
    expect(contradictedText(check)).toBe('')
    expect(contradictedText({ ...check, contradicted: [{ hole_id: 'SYN-02', agreed: 1, compared: 9, example: { sample_id: '17026', derived_to: 26, depth_to: 24.6 } }], contradicted_rows: 4 }))
      .toBe("; NOT given depths (4 rows): the file's own from / to show that the sample numbers are not depths in SYN-02 (sample 17026 would end at 26, the file says 24.6)")
    expect(contradictedText({ ...check, contradicts_sources: [{ hole_id: 'SYN-01', overlapping: 2, rows: 27, example: { sample_id: 'SYN01-4', depth_from: 3, depth_to: 4, other_from: 3, other_to: 4.3, other_file: 'b.csv' } }], contradicts_sources_rows: 27 }))
      .toBe('; NOT imported (27 rows): the depths read from sample numbers overlap the intervals of another file in SYN-01 (sample SYN01-4 read as 3–4, b.csv has 3–4.3); turn the rule off for this file or give a pattern')
  })

  it('says which rows were dropped', () => {
    expect(droppedText('collar', { total_rows: 1, rows_without_hole_id: 2, collars_without_coordinates: 1, holes_without_coordinates: ['SYN-05'] }))
      .toBe('collar: rows dropped: 1 total / summary row, 2 without a hole id, 1 without coordinates (SYN-05)')
    expect(droppedText('assay', { total_rows: 5 })).toBe('assay: rows dropped: 5 total / summary rows')
  })

  it('sends what a file does differently from the others', () => {
    const form = { ...emptyForm(), dip: 'positive_down' as const, lengthUnit: 'ft' as const, order: 'yxz' as const, crs: ' EPSG:32634 ' }
    expect(sourceSpec(form, false)).toEqual({ options: { dip_convention: 'positive_down', length_unit: 'ft', coordinate_order: 'yxz', crs: 'EPSG:32634' } })
    // an interval table has depths only: the unit
    expect(sourceSpec(form, true)).toEqual({ options: { length_unit: 'ft' } })
    expect(sourceText({ file: 'gis.xlsx', dip_convention: 'positive_down', length_unit: 'ft' })).toBe('gis.xlsx: dip convention positive_down, length unit ft')
  })

  it('summarises merge conflicts per column', () => {
    const report = { precedence: 'first', keys: ['hole_id'], merged: 360, filled: 0, conflicts: 4, rows: [] }
    expect(conflictsText(report)).toBe('')
    expect(conflictsText({ ...report, by_column: { au: 3, ag: 1 }, informational: { sample_id: 360 }, listing: 'conflicts_assay.csv' }))
      .toBe(' (au 3, ag 1); not counted (informational): sample_id 360; every difference is in conflicts_assay.csv')
  })

  it('says which intervals were there twice', () => {
    const report = { rule: 'keep_first', intervals: 3, rows_removed: 3, holes: 2, listing: 'duplicates_assay.csv', examples: [{ hole_id: 'SYN-01', depth_from: 4, depth_to: 5, rows: 2 }] }
    expect(duplicatesText('assay', report)).toBe('assay: 3 intervals in 2 holes more than once (keep first): 3 rows removed, e.g. SYN-01 4–5; listed in duplicates_assay.csv')
  })

  it('says which holes the sample-number rule was refused for', () => {
    const check = { rows: 10, duplicates: 0, backwards: 0, gaps: 0, overlaps: 0, unmatched: 0, unmatched_examples: [] }
    expect(beyondText(check)).toBe('')
    expect(beyondText({ ...check, beyond_hole: [{ hole_id: 'SYN-06', derived_to: 70250, max_depth: 250, rows: 250, example: 'SYN06-70250' }] }))
      .toBe('; NOT imported, deeper than the hole: SYN-06 (to 70250, hole 250)')
  })

  it('builds the hole filter', () => {
    expect(holeFilter(' ', ',\n')).toBeNull()
    expect(holeFilter('SYN-01, SYN-02\nSYN25*; SYN-01', '')).toEqual({ include: ['SYN-01', 'SYN-02', 'SYN25*'] })
    expect(holeFilter('', ' SYN-07 ')).toEqual({ exclude: ['SYN-07'] })
    expect(holeFilterText({ include: ['A*'], exclude: [], tables: { collar: { rows_kept: 2, rows_dropped: 4, holes_dropped: 4 } }, unmatched: ['X*'] }))
      .toBe('Hole filter: kept collar 2 of 6 rows; matched no hole: X*')
  })

  it('builds the hole id rule', () => {
    expect(holeIdRule(' ', 'x')).toBeNull()
    expect(holeIdRule('SYN[_-]?0*(\\d+)', '')).toEqual({ pattern: 'SYN[_-]?0*(\\d+)', template: '{0}' })
  })
})

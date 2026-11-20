import { describe, expect, it } from 'vitest'
import { droppedText, emptyForm, holeFilter, holeFilterText, holeIdRule, matchSheets, sourceSpec, sourceText } from './importSpec'

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

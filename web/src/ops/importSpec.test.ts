import { describe, expect, it } from 'vitest'
import { emptyForm, holeIdRule, matchSheets, sourceSpec } from './importSpec'

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

  it('builds the hole id rule', () => {
    expect(holeIdRule(' ', 'x')).toBeNull()
    expect(holeIdRule('SYN[_-]?0*(\\d+)', '')).toEqual({ pattern: 'SYN[_-]?0*(\\d+)', template: '{0}' })
  })
})

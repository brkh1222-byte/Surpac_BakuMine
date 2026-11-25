/** P9.T5: category colours of drillhole intervals, with and without a colour map. */
import { expect, test } from 'vitest'
import type { CategoricalAttributeInfo, CategoricalSegmentAttribute } from '../api/client'
import { legendRows } from '../colour/legend'
import { CATEGORICAL, hexToRgb, MISSING_COLOR, rgbToHex, srgbToLinear } from '../colour/ramps'
import { drillholeLegend, pieceColors, segmentColor } from './colouring'
import type { DrillholePieces } from './pieces'

const THEMES = ['dark', 'light'] as const

function units(names: string[], codes: number[]): CategoricalSegmentAttribute {
  return { name: 'unit', type: 'categorical', categories: names, codes: Int32Array.from(codes) }
}

function info(names: string[], colours?: Record<string, string>): CategoricalAttributeInfo {
  return { table: 'lithology', name: 'unit', type: 'categorical', categories: names, count: 10, missing: 0, ...(colours ? { colours } : {}) }
}

test('no colour map: intervals take the palette by category position, 20 distinct colours', () => {
  const names = Array.from({ length: 24 }, (_, i) => `U${i}`)
  const attribute = units(names, names.map((_, i) => i).concat(-1))
  for (const theme of THEMES) {
    const spec = drillholeLegend(attribute, info(names), undefined)
    if (!spec || spec.kind !== 'categorical') throw new Error('no categorical legend')
    expect(spec.colours).toBeUndefined()
    expect(spec.hasMissing).toBe(true)
    const rows = legendRows(spec, theme)
    for (let s = 0; s < 24; s++) {
      const hex = rgbToHex(segmentColor(attribute, spec, s, theme))
      expect(hex).toBe(CATEGORICAL[theme][s % 20])
      expect(hex).toBe(rows[s].color)
    }
    expect(new Set(rows.slice(0, 20).map((r) => r.color)).size).toBe(20)
    expect(rgbToHex(segmentColor(attribute, spec, 24, theme))).toBe(MISSING_COLOR[theme])
    expect(rgbToHex(segmentColor(attribute, spec, -1, theme))).toBe(MISSING_COLOR[theme])
  }
})

test('a colour map: the same unit has the same colour in tables with different category lists', () => {
  const colours = { ORE: '#e8c547', DYKE: '#203040' }
  // Two tables of one project: different units present, so different codes for ORE.
  const a = units(['DYKE', 'ORE', 'WASTE'], [1, 0, 2, -1])
  const b = units(['ORE', 'SCHIST'], [0, 1])
  for (const theme of THEMES) {
    const specA = drillholeLegend(a, info(a.categories, colours), undefined)
    const specB = drillholeLegend(b, info(b.categories, colours), undefined)
    if (!specA || !specB) throw new Error('no legend')
    expect(rgbToHex(segmentColor(a, specA, 0, theme))).toBe('#e8c547')
    expect(rgbToHex(segmentColor(b, specB, 0, theme))).toBe('#e8c547')
    expect(rgbToHex(segmentColor(a, specA, 1, theme))).toBe('#203040')
    expect(rgbToHex(segmentColor(a, specA, 2, theme))).toBe(CATEGORICAL[theme][2]) // WASTE: not in the map
    expect(rgbToHex(segmentColor(a, specA, 3, theme))).toBe(MISSING_COLOR[theme])
    expect(legendRows(specA, theme)).toEqual([
      { color: '#203040', label: 'DYKE' },
      { color: '#e8c547', label: 'ORE' },
      { color: CATEGORICAL[theme][2], label: 'WASTE' },
      { color: MISSING_COLOR[theme], label: 'No value' },
    ])

    // The line buffer (linear RGB) carries the same colours.
    const pieces = { count: 2, interval: Int32Array.of(0, 3) } as unknown as DrillholePieces
    const buffer = pieceColors(pieces, a, specA, theme)
    const want = [...hexToRgb('#e8c547').map(srgbToLinear), ...hexToRgb('#e8c547').map(srgbToLinear), ...hexToRgb(MISSING_COLOR[theme]).map(srgbToLinear), ...hexToRgb(MISSING_COLOR[theme]).map(srgbToLinear)]
    Array.from(buffer).forEach((v, k) => expect(v).toBeCloseTo(want[k], 6))
  }
})

test('the colour map of a numeric or absent manifest entry is ignored', () => {
  const a = units(['ORE'], [0])
  const spec = drillholeLegend(a, undefined, undefined)
  expect(spec).toEqual({ kind: 'categorical', title: 'unit', categories: ['ORE'], hasMissing: false })
})

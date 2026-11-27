/** P9.T5: category colours of blocks - the GPU table agrees with the legend. */
import { expect, test } from 'vitest'
import { legendRows } from '../colour/legend'
import { CATEGORICAL, hexToRgb, MISSING_COLOR, rgbToHex } from '../colour/ramps'
import { blockLegend, legendColor, MISSING, rampTexture, shaderColouring, TABLE_SIZE, tableColor } from './colour'

const THEMES = ['dark', 'light'] as const
const texel = (table: Uint8Array, i: number) => rgbToHex([table[i * 4] / 255, table[i * 4 + 1] / 255, table[i * 4 + 2] / 255])

function unitInfo(count: number, colours?: Record<string, string>) {
  const categories = Array.from({ length: count }, (_, i) => `U${String(i).padStart(2, '0')}`)
  return { table: 'blocks', name: 'unit', type: 'categorical' as const, categories, count: 1000, missing: 0, ...(colours ? { colours } : {}) }
}

test('20 units: every code has its own colour on the GPU, equal to its legend row', () => {
  for (const theme of THEMES) {
    const spec = blockLegend(unitInfo(20), undefined, false)
    const table = rampTexture(spec, theme)
    const c = shaderColouring(spec)
    const rows = legendRows(spec!, theme)
    const seen = new Set<string>()
    for (let code = 0; code < 20; code++) {
      const onScreen = rgbToHex(tableColor(table, code, c, theme))
      expect(onScreen).toBe(CATEGORICAL[theme][code])
      expect(onScreen).toBe(rows[code].color)
      expect(tableColor(table, code, c, theme)).toEqual(legendColor(spec, code, theme))
      seen.add(onScreen)
    }
    expect(seen.size).toBe(20)
  }
})

test('a colour map reaches the table, the legend and the read-out alike', () => {
  const colours = { U02: '#e8c547', U17: '#102030' }
  for (const theme of THEMES) {
    const spec = blockLegend(unitInfo(20, colours), undefined, true)
    const table = rampTexture(spec, theme)
    const c = shaderColouring(spec)
    const rows = legendRows(spec!, theme)
    expect(texel(table, 2)).toBe('#e8c547')
    expect(texel(table, 17)).toBe('#102030')
    expect(texel(table, 3)).toBe(CATEGORICAL[theme][3])
    for (let code = 0; code < 20; code++) {
      expect(rgbToHex(tableColor(table, code, c, theme))).toBe(rows[code].color)
      expect(tableColor(table, code, c, theme)).toEqual(legendColor(spec, code, theme))
    }
    expect(rows[20]).toEqual({ color: MISSING_COLOR[theme], label: 'No value' })
    expect(legendColor(spec, MISSING, theme)).toEqual(hexToRgb(MISSING_COLOR[theme]))
  }
})

test('more categories than palette entries, and than table texels: cycles, never throws', () => {
  for (const theme of THEMES) {
    const spec = blockLegend(unitInfo(300), undefined, false)
    const table = rampTexture(spec, theme)
    expect(table.length).toBe(TABLE_SIZE * 4)
    for (let i = 0; i < TABLE_SIZE; i++) expect(texel(table, i)).toBe(CATEGORICAL[theme][i % 20])
    const c = shaderColouring(spec)
    expect(rgbToHex(tableColor(table, 27, c, theme))).toBe(CATEGORICAL[theme][7])
    // Codes past the table share its last texel (as before this task).
    expect(rgbToHex(tableColor(table, 299, c, theme))).toBe(CATEGORICAL[theme][255 % 20])
  }
})

/** P9.T5: the 20-colour categorical palette and colours supplied by the data. */
import { expect, test } from 'vitest'
import { blockLegend, NEUTRAL_COLOR } from '../blocks/colour'
import { SCENE_COLORS } from '../scene/theme'
import { categoryColors, codeColor, legendRows, type CategoricalLegendSpec } from './legend'
import { CATEGORICAL, categoryColor, contrast, explicitColor, hexToRgb, MISSING_COLOR, OTHER_COLOR, rgbToHex } from './ramps'

const THEMES = ['dark', 'light'] as const

/**
 * CIELAB (D65, 2 degree observer) of an sRGB hex colour, written here from the CIE
 * definitions, independently of colour/ramps.ts.
 */
function lab(hex: string): [number, number, number] {
  const channel = (i: number) => {
    const c = parseInt(hex.slice(1 + 2 * i, 3 + 2 * i), 16) / 255
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  const [r, g, b] = [channel(0), channel(1), channel(2)]
  const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
  const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b
  const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116)
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))]
}

/** CIE76 colour difference: the Euclidean distance in CIELAB. */
function deltaE(a: string, b: string): number {
  const p = lab(a)
  const q = lab(b)
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2])
}

test('the CIELAB conversion of this test matches published values', () => {
  // White and black are fixed by the definition; the primaries are the widely tabulated
  // sRGB / D65 values (e.g. Lindbloom's calculator), to 0.05.
  const cases: [string, number[]][] = [
    ['#ffffff', [100, 0, 0]],
    ['#000000', [0, 0, 0]],
    ['#ff0000', [53.24, 80.09, 67.2]],
    ['#00ff00', [87.73, -86.18, 83.18]],
    ['#0000ff', [32.3, 79.19, -107.86]],
  ]
  for (const [hex, want] of cases) lab(hex).forEach((v, k) => expect(Math.abs(v - want[k]), `${hex}[${k}]`).toBeLessThan(0.05))
  expect(deltaE('#ffffff', '#000000')).toBeCloseTo(100, 3)
})

test('20 colours per theme; the first eight are the original palette', () => {
  expect(CATEGORICAL.light.slice(0, 8)).toEqual(['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'])
  expect(CATEGORICAL.dark.slice(0, 8)).toEqual(['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#0ca30c', '#9085e9', '#e66767'])
  for (const theme of THEMES) {
    const palette = CATEGORICAL[theme]
    expect(palette.length).toBe(20)
    expect(new Set(palette).size).toBe(20)
    for (const hex of palette) expect(hex).toMatch(/^#[0-9a-f]{6}$/)
  }
})

test('palette entries stay apart: CIE76 between neighbours, all pairs, and the greys', () => {
  for (const theme of THEMES) {
    const palette = CATEGORICAL[theme]
    const canvas = hexToRgb(SCENE_COLORS[theme].background)
    for (let i = 0; i < palette.length; i++) {
      // Neighbours in the list, including the wrap-around the cycle creates (20th -> 1st).
      const next = palette[(i + 1) % palette.length]
      expect(deltaE(palette[i], next), `${theme} neighbours ${palette[i]} / ${next}`).toBeGreaterThanOrEqual(30)
      for (let j = i + 1; j < palette.length; j++) {
        // Any pair: 20 (two pairs of the original eight sit at 20.7 and 22.0); a pair
        // with an added entry: 30.
        const floor = j >= 8 ? 30 : 20
        expect(deltaE(palette[i], palette[j]), `${theme} ${palette[i]} / ${palette[j]}`).toBeGreaterThanOrEqual(floor)
      }
    }
    // The added entries are not mistaken for "no value", "other" or uncoloured blocks,
    // and they show on the canvas.
    for (const hex of palette.slice(8)) {
      for (const grey of [MISSING_COLOR[theme], OTHER_COLOR[theme], NEUTRAL_COLOR[theme]]) {
        expect(deltaE(hex, grey), `${theme} ${hex} / grey ${grey}`).toBeGreaterThanOrEqual(20)
      }
      expect(contrast(hexToRgb(hex), canvas), `${theme} ${hex} on canvas`).toBeGreaterThanOrEqual(2.2)
    }
  }
})

test('more than 20 categories: the palette cycles, nothing throws', () => {
  for (const theme of THEMES) {
    const palette = CATEGORICAL[theme]
    for (let i = 0; i < 300; i++) expect(categoryColor(theme, i)).toBe(palette[i % 20])
    expect(categoryColor(theme, 20)).toBe(palette[0])
    expect(categoryColor(theme, 2.7)).toBe(palette[2])
    for (const bad of [-1, Number.NaN, Number.POSITIVE_INFINITY]) expect(categoryColor(theme, bad)).toBe(OTHER_COLOR[theme])

    const categories = Array.from({ length: 45 }, (_, i) => `U${i}`)
    const spec: CategoricalLegendSpec = { kind: 'categorical', title: 'unit', categories, hasMissing: true }
    const rows = legendRows(spec, theme)
    expect(rows.length).toBe(46)
    expect(rows.map((r) => r.label)).toEqual([...categories, 'No value'])
    rows.slice(0, 45).forEach((row, i) => expect(row.color).toBe(palette[i % 20]))
    expect(rows[45].color).toBe(MISSING_COLOR[theme])
    expect(rgbToHex(codeColor(44, theme, spec))).toBe(palette[4])
    // A code past the list (corrupt data) still gets a colour.
    expect(rgbToHex(codeColor(1000, theme, spec))).toBe(palette[0])
  }
})

test('without a colour map the assignment is by list position, as before', () => {
  for (const theme of THEMES) {
    const spec: CategoricalLegendSpec = { kind: 'categorical', title: 'rock', categories: ['AIR', 'WASTE', 'ORE'], hasMissing: false }
    expect(categoryColors(spec, theme)).toEqual(CATEGORICAL[theme].slice(0, 3))
    expect(legendRows(spec, theme)).toEqual([
      { color: CATEGORICAL[theme][0], label: 'AIR' },
      { color: CATEGORICAL[theme][1], label: 'WASTE' },
      { color: CATEGORICAL[theme][2], label: 'ORE' },
    ])
    for (let code = 0; code < 3; code++) {
      expect(codeColor(code, theme, spec)).toEqual(codeColor(code, theme))
      expect(rgbToHex(codeColor(code, theme))).toBe(CATEGORICAL[theme][code])
    }
    expect(rgbToHex(codeColor(-1, theme, spec))).toBe(MISSING_COLOR[theme])
  }
})

test('a colour map wins over the palette and does not depend on which categories are present', () => {
  const colours = { ORE: '#E8C547', DYKE: '#101010', GRANITE: '#aa3366' }
  for (const theme of THEMES) {
    const all: CategoricalLegendSpec = { kind: 'categorical', title: 'unit', categories: ['DYKE', 'GRANITE', 'ORE', 'WASTE'], colours, hasMissing: false }
    const some: CategoricalLegendSpec = { kind: 'categorical', title: 'unit', categories: ['ORE', 'WASTE'], colours, hasMissing: true }
    // Upper-case hex is normalised; WASTE is not in the map and takes the palette entry
    // of its position.
    expect(categoryColors(all, theme)).toEqual(['#101010', '#aa3366', '#e8c547', CATEGORICAL[theme][3]])
    expect(categoryColors(some, theme)).toEqual(['#e8c547', CATEGORICAL[theme][1]])
    expect(rgbToHex(codeColor(2, theme, all))).toBe('#e8c547')
    expect(rgbToHex(codeColor(0, theme, some))).toBe('#e8c547')
    expect(legendRows(some, theme)).toEqual([
      { color: '#e8c547', label: 'ORE' },
      { color: CATEGORICAL[theme][1], label: 'WASTE' },
      { color: MISSING_COLOR[theme], label: 'No value' },
    ])
  }
  // The same in both themes.
  const spec: CategoricalLegendSpec = { kind: 'categorical', title: 'unit', categories: ['ORE'], colours, hasMissing: false }
  expect(categoryColors(spec, 'dark')).toEqual(categoryColors(spec, 'light'))
})

test('unusable entries of a colour map fall back to the palette', () => {
  const colours = { A: 'red', B: '#12345', C: '#1234567', D: '#12345g', E: '', toString: '#000000', F: 7 } as unknown as Record<string, string>
  for (const name of ['A', 'B', 'C', 'D', 'E', 'F', 'missing']) expect(explicitColor(colours, name), name).toBeNull()
  expect(explicitColor(undefined, 'A')).toBeNull()
  // An inherited property name is not a colour of the data...
  expect(explicitColor({}, 'toString')).toBeNull()
  expect(explicitColor({}, 'constructor')).toBeNull()
  // ...but a category that really is called that works.
  expect(explicitColor(colours, 'toString')).toBe('#000000')
  const spec: CategoricalLegendSpec = { kind: 'categorical', title: 'x', categories: ['A', 'B', 'constructor'], colours, hasMissing: false }
  expect(categoryColors(spec, 'dark')).toEqual(CATEGORICAL.dark.slice(0, 3))
})

test('block legends carry the colour map of the manifest entry', () => {
  const info = { table: 'blocks', name: 'unit', type: 'categorical' as const, categories: ['A', 'B'], count: 4, missing: 0 }
  expect(blockLegend(info, undefined, false)).toEqual({ kind: 'categorical', title: 'unit', categories: ['A', 'B'], hasMissing: false })
  expect(blockLegend({ ...info, colours: { B: '#123456' } }, undefined, true)).toEqual({
    kind: 'categorical', title: 'unit', categories: ['A', 'B'], colours: { B: '#123456' }, hasMissing: true,
  })
})

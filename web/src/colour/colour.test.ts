import { expect, test } from 'vitest'
import { SCENE_COLORS } from '../scene/theme'
import { defaultSettings } from '../state/display'
import {
  codeColor,
  continuousTicks,
  decimalsFor,
  legendRows,
  legendTitle,
  numericColor,
  rampPosition,
  roundNice,
  sanitiseRange,
  steppedBreaks,
  type NumericLegendSpec,
} from './legend'
import {
  CATEGORICAL,
  categoryColor,
  contrast,
  hexToRgb,
  MISSING_COLOR,
  oklabLightness,
  OTHER_COLOR,
  RAMP_STOPS,
  rampColor,
  rgbToHex,
  srgbToLinear,
  type RampName,
} from './ramps'

const THEMES = ['dark', 'light'] as const
const RAMPS = Object.keys(RAMP_STOPS) as RampName[]

const spec = (patch: Partial<NumericLegendSpec> = {}): NumericLegendSpec => ({
  kind: 'numeric',
  title: 'au',
  unit: 'g/t',
  ramp: 'grade',
  mode: 'continuous',
  min: 0,
  max: 2,
  steps: 4,
  hasMissing: false,
  ...patch,
})

test('hex and linear conversions', () => {
  expect(hexToRgb('#ff8000')).toEqual([1, 128 / 255, 0])
  expect(rgbToHex([1, 128 / 255, 0])).toBe('#ff8000')
  expect(rgbToHex(hexToRgb('#0a0b0c'))).toBe('#0a0b0c')
  expect(srgbToLinear(0)).toBe(0)
  expect(srgbToLinear(1)).toBeCloseTo(1, 12)
  expect(srgbToLinear(0.5)).toBeCloseTo(0.21404, 4)
  expect(contrast([1, 1, 1], [0, 0, 0])).toBeCloseTo(21, 6)
})

test('ramps are perceptually ordered: lightness rises strictly from low to high', () => {
  for (const name of RAMPS) {
    for (const theme of THEMES) {
      let previous = -1
      for (let i = 0; i <= 50; i++) {
        const lightness = oklabLightness(rampColor(name, theme, i / 50))
        expect(lightness, `${name}/${theme} at ${i / 50}`).toBeGreaterThan(previous)
        previous = lightness
      }
    }
  }
})

test('grade ramp runs cool to warm and stays readable on its canvas', () => {
  for (const theme of THEMES) {
    const low = rampColor('grade', theme, 0)
    const high = rampColor('grade', theme, 1)
    expect(low[2]).toBeGreaterThan(low[0]) // low end is blue
    expect(high[0]).toBeGreaterThan(high[2]) // high end is yellow / orange
    const canvas = hexToRgb(SCENE_COLORS[theme].background)
    for (let i = 0; i <= 20; i++) {
      expect(contrast(rampColor('grade', theme, i / 20), canvas), `${theme} at ${i / 20}`).toBeGreaterThanOrEqual(2)
    }
    // "No value" is visible too, and is not mistaken for either end of the ramp.
    const missing = hexToRgb(MISSING_COLOR[theme])
    expect(contrast(missing, canvas)).toBeGreaterThanOrEqual(2)
    expect(Math.max(...missing) - Math.min(...missing)).toBeLessThan(0.08) // grey
  }
})

test('rampColor clamps and interpolates between the stops', () => {
  const stops = RAMP_STOPS.grade.dark
  expect(rgbToHex(rampColor('grade', 'dark', 0))).toBe(stops[0])
  expect(rgbToHex(rampColor('grade', 'dark', 1))).toBe(stops[stops.length - 1])
  expect(rgbToHex(rampColor('grade', 'dark', 0.5))).toBe(stops[2])
  expect(rampColor('grade', 'dark', -3)).toEqual(rampColor('grade', 'dark', 0))
  expect(rampColor('grade', 'dark', 7)).toEqual(rampColor('grade', 'dark', 1))
  expect(rampColor('grade', 'dark', Number.NaN)).toEqual(rampColor('grade', 'dark', 0))
  const a = hexToRgb(stops[0])
  const b = hexToRgb(stops[1])
  const mid = rampColor('grade', 'dark', 0.125)
  for (let k = 0; k < 3; k++) expect(mid[k]).toBeCloseTo((a[k] + b[k]) / 2, 12)
})

test('categorical colours are distinct and fixed in order; past the 20th they cycle', () => {
  for (const theme of THEMES) {
    const palette = CATEGORICAL[theme]
    expect(new Set(palette).size).toBe(palette.length)
    expect(categoryColor(theme, 0)).toBe(palette[0])
    expect(categoryColor(theme, 7)).toBe(palette[7])
    // P9.T5: 20 entries instead of 8, so the ninth is a colour of its own and the list
    // cycles past its end (it used to fall back to OTHER_COLOR from the ninth on).
    expect(palette.length).toBe(20)
    expect(categoryColor(theme, 8)).toBe(palette[8])
    expect(categoryColor(theme, 99)).toBe(palette[99 % 20])
    expect(categoryColor(theme, -1)).toBe(OTHER_COLOR[theme])
    // Every pair differs (a crude sRGB distance; the palette itself was checked for
    // colour-blind separation with the dataviz validator): clearly among the first six,
    // which is what most rock-code tables use, and still visibly for the last two.
    for (let i = 0; i < palette.length; i++) {
      for (let j = i + 1; j < palette.length; j++) {
        const a = hexToRgb(palette[i])
        const b = hexToRgb(palette[j])
        const distance = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
        expect(distance, `${palette[i]} / ${palette[j]}`).toBeGreaterThan(j < 6 ? 0.2 : 0.12)
      }
    }
    expect(rgbToHex(codeColor(-1, theme))).toBe(MISSING_COLOR[theme])
    expect(rgbToHex(codeColor(1, theme))).toBe(palette[1])
  }
})

test('continuous mapping: linear in the range, clamped outside, missing is not zero', () => {
  const s = spec()
  expect(rampPosition(s, 0)).toBe(0)
  expect(rampPosition(s, 0.5)).toBe(0.25)
  expect(rampPosition(s, 2)).toBe(1)
  expect(rampPosition(s, -1)).toBe(0)
  expect(rampPosition(s, 50)).toBe(1)
  expect(rampPosition(s, Number.NaN)).toBeNull()
  expect(rgbToHex(numericColor(s, Number.NaN, 'dark'))).toBe(MISSING_COLOR.dark)
  expect(rgbToHex(numericColor(s, 0, 'dark'))).toBe(RAMP_STOPS.grade.dark[0])
  expect(numericColor(s, Number.NaN, 'dark')).not.toEqual(numericColor(s, 0, 'dark'))
  // The edited range moves the colours.
  expect(numericColor(spec({ max: 4 }), 2, 'dark')).toEqual(rampColor('grade', 'dark', 0.5))
  // A degenerate range does not divide by zero.
  expect(rampPosition(spec({ min: 1, max: 1 }), 1)).toBe(1)
  expect(rampPosition(spec({ min: 1, max: 1 }), 0.5)).toBe(0)
})

test('stepped mapping: one colour per class, boundaries belong to the upper class', () => {
  const s = spec({ mode: 'stepped' })
  expect(steppedBreaks(s)).toEqual([0, 0.5, 1, 1.5, 2])
  expect(rampPosition(s, 0)).toBe(0.125)
  expect(rampPosition(s, 0.49)).toBe(0.125)
  expect(rampPosition(s, 0.5)).toBe(0.375)
  expect(rampPosition(s, 1.99)).toBe(0.875)
  expect(rampPosition(s, 2)).toBe(0.875)
  expect(rampPosition(s, 9)).toBe(0.875)
  expect(numericColor(s, 0.1, 'light')).toEqual(numericColor(s, 0.4, 'light'))
  expect(numericColor(s, 0.4, 'light')).not.toEqual(numericColor(s, 0.6, 'light'))
})

test('legend rows and ticks', () => {
  const stepped = spec({ mode: 'stepped', hasMissing: true, dataMin: 0, dataMax: 31.2 })
  const rows = legendRows(stepped, 'dark')
  expect(rows.map((r) => r.label)).toEqual(['≥ 1.50', '1.00 – 1.50', '0.50 – 1.00', '0.00 – 0.50', 'No value'])
  expect(rows[0].color).toBe(rgbToHex(rampColor('grade', 'dark', 0.875)))
  expect(rows[3].color).toBe(rgbToHex(numericColor(stepped, 0.2, 'dark')))
  expect(rows[4].color).toBe(MISSING_COLOR.dark)
  // Without clipping the top class is a plain range.
  expect(legendRows(spec({ mode: 'stepped', dataMax: 2 }), 'dark')[0].label).toBe('1.50 – 2.00')
  expect(legendRows(spec({ mode: 'stepped', min: 1, dataMin: 0 }), 'dark').at(-1)?.label).toBe('< 1.25')

  expect(continuousTicks(spec({ dataMin: 0, dataMax: 31.2 }))).toEqual(['0.00', '1.00', '≥ 2.00'])
  expect(continuousTicks(spec({ min: 0.5, dataMin: 0, dataMax: 1 }))).toEqual(['≤ 0.50', '1.25', '2.00'])
  expect(legendRows(spec(), 'dark')).toEqual([])

  const categorical = legendRows({ kind: 'categorical', title: 'rock', categories: ['DYKE', 'ORE'], hasMissing: false }, 'light')
  expect(categorical).toEqual([
    { color: CATEGORICAL.light[0], label: 'DYKE' },
    { color: CATEGORICAL.light[1], label: 'ORE' },
  ])
  expect(legendTitle(spec())).toBe('au (g/t)')
  expect(legendTitle(spec({ unit: null }))).toBe('au')
})

test('number helpers', () => {
  expect(decimalsFor(0.5)).toBe(2)
  expect(decimalsFor(25)).toBe(0)
  expect(decimalsFor(0)).toBe(0)
  expect(roundNice(1.9205, 'up')).toBe(2)
  expect(roundNice(0.0437, 'up')).toBe(0.044)
  expect(roundNice(1234, 'down')).toBe(1200)
  expect(roundNice(2, 'up')).toBe(2)
  expect(sanitiseRange(3, 1)).toEqual({ min: 1, max: 3 })
  expect(sanitiseRange(1, 1)).toBeNull()
  expect(sanitiseRange(Number.NaN, 1)).toBeNull()
})

test('default legend range: zero to the rounded 98th percentile', () => {
  const settings = defaultSettings({
    table: 'assay', name: 'au', type: 'numeric', unit: 'g/t',
    min: 0.001, max: 31.2, display_min: 0.008, display_max: 1.9205, count: 10, missing: 0,
  })
  expect(settings).toEqual({ mode: 'continuous', min: 0, max: 2, steps: 6 })
  const signed = defaultSettings({
    table: 't', name: 'x', type: 'numeric', unit: null,
    min: -5, max: 9, display_min: -4.31, display_max: 8.12, count: 10, missing: 0,
  })
  expect(signed.min).toBe(-4.4)
  expect(signed.max).toBe(8.2)
  const empty = defaultSettings({
    table: 't', name: 'x', type: 'numeric', unit: null,
    min: null, max: null, display_min: null, display_max: null, count: 0, missing: 4,
  })
  expect(empty.max).toBeGreaterThan(empty.min)
})

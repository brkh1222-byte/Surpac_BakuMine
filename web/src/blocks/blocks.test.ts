/** Colour mapping and pick maths of the blocks layer. */
import { expect, test } from 'vitest'
import type { BlockColumn, BlockNumericInfo } from '../api/blocks'
import { codeColor, numericColor, rampPosition, type NumericLegendSpec } from '../colour/legend'
import { categoryColor, hexToRgb, MISSING_COLOR, rgbToHex } from '../colour/ramps'
import {
  blockLegend,
  countMissing,
  legendColor,
  MISSING,
  rampTexture,
  shaderColouring,
  tableColor,
  tablePosition,
  valueBuffer,
} from './colour'
import { blockBox, buildPickIndex, pickBlock } from './pick'
import { CASES, structureOf } from './fixtures/cases'
import { parseDefinition, worldToLocal, type BlockStructure } from './structure'

const AU: BlockNumericInfo = {
  table: 'blocks', name: 'au', type: 'numeric', unit: 'g/t', min: 0.02, max: 25.9, display_min: 0.02, display_max: 4.6, percentiles: null, count: 10, missing: 2,
}

function numericSpec(mode: 'continuous' | 'stepped', min = 0, max = 4.6, steps = 6): NumericLegendSpec {
  const spec = blockLegend(AU, { mode, min, max, steps }, true)
  if (!spec || spec.kind !== 'numeric') throw new Error('expected a numeric legend')
  return spec
}

test('value buffer: raw values and codes, with one sentinel for "no value"', () => {
  const numeric: BlockColumn = { name: 'au', type: 'numeric', unit: 'g/t', values: Float32Array.of(0.5, Number.NaN, 12) }
  const values = valueBuffer(numeric, 3)
  expect(Array.from(values)).toEqual([0.5, Math.fround(MISSING), 12])
  expect(countMissing(values)).toBe(1)
  const codes: BlockColumn = { name: 'rock', type: 'categorical', categories: ['AIR', 'WASTE', 'ORE'], codes: Int16Array.of(2, -1, 0) }
  expect(Array.from(valueBuffer(codes, 3))).toEqual([2, Math.fround(MISSING), 0])
  expect(Array.from(valueBuffer(null, 2))).toEqual([0, 0])
})

test('the legend uses the grade ramp and carries the data range', () => {
  const spec = numericSpec('continuous')
  expect(spec).toMatchObject({ kind: 'numeric', title: 'au', unit: 'g/t', ramp: 'grade', min: 0, max: 4.6, dataMin: 0.02, dataMax: 25.9, hasMissing: true })
  const rock = blockLegend({ table: 'blocks', name: 'rock', type: 'categorical', categories: ['AIR', 'WASTE', 'ORE'], count: 3, missing: 0 }, undefined, false)
  expect(rock).toEqual({ kind: 'categorical', title: 'rock', categories: ['AIR', 'WASTE', 'ORE'], hasMissing: false })
  expect(blockLegend(undefined, undefined, false)).toBeNull()
  expect(blockLegend(AU, undefined, false)).toBeNull()
})

test('shader table position equals the legend position, continuous and stepped', () => {
  for (const mode of ['continuous', 'stepped'] as const) {
    const spec = numericSpec(mode, 0.5, 4.5, 5)
    const c = shaderColouring(spec)
    expect(c.steps).toBe(mode === 'stepped' ? 5 : 0)
    for (const v of [-1, 0.5, 0.51, 1.3, 2.5, 3.3, 4.49, 4.5, 30]) {
      expect(tablePosition(v, c)).toBeCloseTo(rampPosition(spec, v) ?? -1, 12)
    }
  }
})

test('colours read from the table match the legend (and so the drillholes) within a byte', () => {
  for (const theme of ['dark', 'light'] as const) {
    for (const mode of ['continuous', 'stepped'] as const) {
      const spec = numericSpec(mode)
      const table = rampTexture(spec, theme)
      const c = shaderColouring(spec)
      for (const v of [0, 0.37, 1, 1.15, 2.3, 3.45, 4.6, 9]) {
        const got = tableColor(table, v, c, theme)
        const want = numericColor(spec, v, theme)
        got.forEach((g, k) => expect(Math.abs(g - want[k])).toBeLessThan(1.5 / 255))
        expect(legendColor(spec, v, theme)).toEqual(want)
      }
      expect(rgbToHex(tableColor(table, MISSING, c, theme))).toBe(MISSING_COLOR[theme])
    }
    const categorical = blockLegend({ table: 'blocks', name: 'rock', type: 'categorical', categories: ['AIR', 'WASTE', 'ORE'], count: 3, missing: 0 }, undefined, false)
    const table = rampTexture(categorical, theme)
    const c = shaderColouring(categorical)
    expect(c.categorical).toBe(true)
    for (const code of [0, 1, 2, 7, 8, 40]) {
      expect(rgbToHex(tableColor(table, code, c, theme))).toBe(categoryColor(theme, code))
      expect(tableColor(table, code, c, theme)).toEqual(codeColor(code, theme))
    }
    expect(hexToRgb(MISSING_COLOR[theme])).toEqual(legendColor(categorical, MISSING, theme))
  }
})

function subset(structure: BlockStructure, rows: number[]): BlockStructure {
  return {
    count: rows.length,
    parent: Uint32Array.from(rows, (r) => structure.parent[r]),
    level: Uint8Array.from(rows, (r) => structure.level[r]),
    sub: Uint8Array.from(rows.flatMap((r) => Array.from(structure.sub.subarray(r * 3, r * 3 + 3)))),
  }
}

for (const c of CASES) {
  test(`${c.name}: the grid walk hits the block Python's brute force hits`, () => {
    const definition = parseDefinition(c.definition)
    const index = buildPickIndex(definition, subset(structureOf(c), c.shown_rows))
    let hits = 0
    for (const [shrink, rays] of Object.entries(c.rays)) {
      for (const ray of rays) {
        const hit = pickBlock(index, worldToLocal(definition, ray.origin), worldToLocal(definition, ray.direction, true), { shrink: Number(shrink) })
        expect(hit ? c.shown_rows[hit.index] : -1).toBe(ray.row)
        if (hit && ray.t !== null) {
          expect(hit.t).toBeCloseTo(ray.t, 6)
          hits++
        }
      }
    }
    expect(hits).toBeGreaterThan(8)
  })
}

test('pick: range limits (section slab), hidden blocks and axis-parallel rays', () => {
  const dense = CASES[0]
  const definition = parseDefinition(dense.definition)
  const index = buildPickIndex(definition, structureOf(dense))
  // Straight down the column i = 1, j = 2 (model frame): the top block (k = 1) first.
  const origin: [number, number, number] = [15, 25, 100]
  const down: [number, number, number] = [0, 0, -1]
  const top = pickBlock(index, origin, down)
  expect(top).toEqual({ index: (1 * 4 + 2) * 2 + 1, t: 90 })
  // Hidden: the ray goes through to the block below.
  expect(pickBlock(index, origin, down, { skip: (b) => b === top?.index })).toEqual({ index: (1 * 4 + 2) * 2, t: 95 })
  // A slab that starts inside the top block: the cut face is what is hit.
  expect(pickBlock(index, origin, down, { tMin: 92 })).toEqual({ index: (1 * 4 + 2) * 2 + 1, t: 92 })
  // A slab that ends before the model, and a ray beside it.
  expect(pickBlock(index, origin, down, { tMax: 50 })).toBeNull()
  expect(pickBlock(index, [-5, 25, 100], down)).toBeNull()
  // Shrunk blocks leave gaps a ray can pass through.
  expect(pickBlock(index, [10.2, 25, 100], down, { shrink: 0.9 })).toBeNull()
  expect(pickBlock(index, [10.2, 25, 100], down, { shrink: 1 })?.index).toBe((1 * 4 + 2) * 2 + 1)

  const box = new Float64Array(6)
  blockBox(index, 3, box)
  expect(Array.from(box)).toEqual([0, 10, 5, 10, 10, 5])
})

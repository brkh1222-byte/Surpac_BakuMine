/**
 * Filter panel state -> constraint JSON. fixtures/filters.json is shared with
 * tests/api/test_blocks_api.py, which sends the same JSON to the API and compares the
 * rows with `constraint.mask(model)` of the constraint built in Python.
 */
import { expect, test } from 'vitest'
import type { BlockAttributeInfo } from '../api/blocks'
import { frameOf, type SectionPlane } from '../section/plane'
import {
  conditionConstraint,
  conditionExpression,
  conditionName,
  defaultFilter,
  filterToConstraint,
  filterToWhere,
  newCondition,
  slabConstraint,
  type BlockFilter,
  type ConstraintJson,
} from './filter'
import { CASES } from './fixtures/cases'
import filtersText from './fixtures/filters.json?raw'
import { parseDefinition } from './structure'

const FILTERS = (JSON.parse(filtersText) as { cases: { name: string; filter: BlockFilter; constraint: ConstraintJson }[] }).cases

const ATTRIBUTES: BlockAttributeInfo[] = [
  { table: 'blocks', name: 'rock', type: 'categorical', categories: ['AIR', 'WASTE', 'ORE'], count: 10, missing: 0 },
  { table: 'blocks', name: 'au', type: 'numeric', unit: 'g/t', min: 0, max: 26, display_min: 0, display_max: 4.6, percentiles: null, count: 9, missing: 1 },
]

for (const c of FILTERS) {
  test(`filter "${c.name}" becomes the constraint JSON of the Python engine`, () => {
    expect(filterToConstraint(c.filter)).toEqual(c.constraint)
    expect(JSON.parse(filterToWhere(c.filter) ?? 'null')).toEqual(c.constraint)
  })
}

test('nothing to filter by is no constraint at all', () => {
  expect(filterToConstraint({ conditions: [], combine: '' })).toBeNull()
  expect(filterToWhere({ conditions: [], combine: '' })).toBeNull()
  const off: BlockFilter = { combine: '', conditions: [{ id: 1, enabled: false, kind: 'expression', text: 'au > 1' }] }
  expect(filterToWhere(off)).toBeNull()
})

test('conditions: names, incomplete ones, quoting, no category ticked', () => {
  expect([0, 1, 25, 26, 27].map(conditionName)).toEqual(['a', 'b', 'z', 'a1', 'b1'])
  expect(conditionExpression({ id: 1, enabled: true, kind: 'category', attribute: 'rock', values: [] })).toBe('false')
  expect(conditionExpression({ id: 1, enabled: true, kind: 'category', attribute: 'zone', values: ['A "rich" zone'] })).toBe('zone == "A \\"rich\\" zone"')
  expect(conditionExpression({ id: 1, enabled: true, kind: 'numeric', attribute: 'au', op: 'between', value: 0, min: null, max: null })).toBeNull()
  expect(conditionExpression({ id: 1, enabled: true, kind: 'numeric', attribute: 'au', op: 'between', value: 0, min: null, max: 3 })).toBe('au <= 3')
  expect(conditionExpression({ id: 1, enabled: true, kind: 'numeric', attribute: 'au', op: '<', value: Number.NaN, min: null, max: null })).toBeNull()
  expect(conditionExpression({ id: 1, enabled: true, kind: 'numeric', attribute: 'au', op: '!=', value: 1e-7, min: null, max: null })).toBe('au != 1e-7')
  expect(conditionConstraint({ id: 1, enabled: true, kind: 'elevation', min: null, max: null })).toBeNull()
  expect(conditionConstraint({ id: 1, enabled: true, kind: 'elevation', min: 900, max: null })).toEqual({ type: 'range', z: [900, null] })
  // A combine expression names conditions by position; a disabled one is simply not defined.
  const filter: BlockFilter = {
    combine: 'a or c',
    conditions: [
      { id: 1, enabled: true, kind: 'expression', text: 'au > 1' },
      { id: 2, enabled: false, kind: 'expression', text: 'au > 2' },
      { id: 3, enabled: true, kind: 'expression', text: 'au > 3' },
    ],
  }
  expect(filterToConstraint(filter)).toEqual({
    type: 'expression',
    expression: 'a or c',
    constraints: { a: { type: 'attribute', expression: 'au > 1' }, c: { type: 'attribute', expression: 'au > 3' } },
  })
})

test('the default filter follows the project hint; new conditions start usable', () => {
  const hinted = defaultFilter(ATTRIBUTES, { attribute: 'rock', values: ['ORE', 'GOLD'] })
  expect(hinted.conditions).toHaveLength(1)
  expect(filterToConstraint(hinted)).toEqual({ type: 'attribute', expression: 'rock == "ORE"' })
  expect(defaultFilter(ATTRIBUTES, undefined).conditions).toEqual([])
  expect(defaultFilter(ATTRIBUTES, { attribute: 'au', values: ['ORE'] }).conditions).toEqual([])

  const definition = parseDefinition(CASES[0].definition)
  const context = { attributes: ATTRIBUTES, solids: [{ id: 'ore-solid' }], surfaces: [{ id: 'topo' }], definition }
  expect(newCondition('numeric', context)).toMatchObject({ kind: 'numeric', attribute: 'au', op: '>', value: 0, enabled: true })
  expect(newCondition('category', context)).toMatchObject({ kind: 'category', attribute: 'rock', values: ['AIR', 'WASTE', 'ORE'] })
  expect(newCondition('solid', context)).toMatchObject({ solid: 'ore-solid', side: 'inside' })
  expect(newCondition('surface', context)).toMatchObject({ surface: 'topo', side: 'below' })
  expect(newCondition('index', context)).toMatchObject({ axis: 'j', min: 0, max: 3 })
  expect(newCondition('solid', { ...context, solids: [] })).toBeNull()
  const ids = [newCondition('expression', context)?.id, newCondition('elevation', context)?.id]
  expect(ids[0]).not.toBe(ids[1])
})

test('the section slab is two planes, widened by half a parent block along the normal', () => {
  const plane: SectionPlane = { centre: [500600, 5300500, 1000], azimuth: 90, dip: 90 }
  const dense = parseDefinition(CASES[0].definition) // blocks of 10 x 10 x 5
  const slab = slabConstraint(plane, { front: 12.5, back: 7.5 }, dense) as { type: string; children: { type: string; point: number[]; normal: number[]; side: string }[] }
  expect(slab.type).toBe('and')
  const n = frameOf(plane).n
  expect(n).toEqual([0, -1, 0])
  // W-E section looking north: the normal points south; half a block along y is 5 m.
  expect(slab.children[0]).toEqual({ type: 'plane', point: [500600, 5300500 - 17.5, 1000], normal: [0, -1, 0], side: 'below' })
  expect(slab.children[1]).toEqual({ type: 'plane', point: [500600, 5300500 + 12.5, 1000], normal: [0, -1, 0], side: 'above' })
  // A plan: half a block along z is 2.5 m.
  const plan = slabConstraint({ centre: [0, 0, 1000], azimuth: 90, dip: 0 }, { front: 2, back: 2 }, dense) as typeof slab
  expect(plan.children[0].point[2]).toBeCloseTo(1004.5, 9)
  expect(plan.children[1].point[2]).toBeCloseTo(995.5, 9)
  // A rotated model reaches further along the same normal than its smallest side.
  const rotated = slabConstraint(plane, { front: 0, back: 0 }, parseDefinition(CASES[2].definition)) as typeof slab
  const reach = 5300500 - rotated.children[0].point[1]
  expect(reach).toBeGreaterThan(3.75)
  expect(reach).toBeLessThan(0.5 * (12.5 + 7.5 + 4))
})

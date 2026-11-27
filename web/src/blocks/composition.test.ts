/**
 * The composition control of a block model: which attributes are layers, what a press of
 * a layer or of "Off" does to the colouring, and the legend rows of a layer.
 */
import { expect, test } from 'vitest'
import { categoryLabel, legendRows } from '../colour/legend'
import { blockLegend } from './colour'
import { activeComposition, compositionLayers, fallbackColorBy, parseComposition, switchComposition, type CompositionColouring } from './composition'

const ATTRIBUTES = [
  { name: 'rock', type: 'categorical' },
  { name: 'au', type: 'numeric' },
  { name: 'fault_block', type: 'categorical' },
  { name: 'lith', type: 'categorical' },
  { name: 'weathering', type: 'categorical' },
  { name: 'density', type: 'numeric' },
]
const RECORD = { geology: 'layered-small', layers: { lith: 'lith', weathering: 'weathering', fault_block: 'fault_block' } }
const LAYERS = compositionLayers(ATTRIBUTES, RECORD)

test('the layers come from the coding record, in the order rock unit, weathering, fault block', () => {
  expect(LAYERS).toEqual([
    { layer: 'lith', label: 'Rock unit', attribute: 'lith' },
    { layer: 'weathering', label: 'Weathering', attribute: 'weathering' },
    { layer: 'fault_block', label: 'Fault block', attribute: 'fault_block' },
  ])
  // the record names the attribute: a layer coded under another name is found, one whose
  // attribute is gone or is not a category is left out
  const renamed = compositionLayers([...ATTRIBUTES, { name: 'unit', type: 'categorical' }], { geology: null, layers: { lith: 'unit', weathering: 'gone', fault_block: 'au' } })
  expect(renamed).toEqual([{ layer: 'lith', label: 'Rock unit', attribute: 'unit' }])
  // a record without layers leaves nothing, although the model has attributes of those names
  expect(compositionLayers(ATTRIBUTES, { geology: null, layers: {} })).toEqual([])
  // no record (a model coded before records were kept): the layers' own names, categories only
  expect(compositionLayers(ATTRIBUTES, null).map((l) => l.attribute)).toEqual(['lith', 'weathering', 'fault_block'])
  expect(compositionLayers([{ name: 'lith', type: 'numeric' }, { name: 'au', type: 'numeric' }], null)).toEqual([])
})

test('switching layers remembers the colouring from before, and Off goes back to it', () => {
  let state: CompositionColouring = { colorBy: 'au' }
  expect(activeComposition(state.colorBy, LAYERS)).toBeNull()
  state = switchComposition(state, ATTRIBUTES, LAYERS, 'lith')
  expect(state).toEqual({ colorBy: 'lith', compositionReturn: 'au' })
  state = switchComposition(state, ATTRIBUTES, LAYERS, 'weathering')
  state = switchComposition(state, ATTRIBUTES, LAYERS, 'fault_block')
  expect(state).toEqual({ colorBy: 'fault_block', compositionReturn: 'au' })
  expect(activeComposition(state.colorBy, LAYERS)).toBe('fault_block')
  state = switchComposition(state, ATTRIBUTES, LAYERS, null)
  expect(state).toEqual({ colorBy: 'au' })
  // Off while it is off changes nothing, also for blocks in one neutral colour
  expect(switchComposition(state, ATTRIBUTES, LAYERS, null)).toEqual({ colorBy: 'au' })
  expect(switchComposition({ colorBy: null }, ATTRIBUTES, LAYERS, null)).toEqual({ colorBy: null })
})

test('Off returns to "no colouring" when that is what was there, and has a fallback otherwise', () => {
  const on = switchComposition({ colorBy: null }, ATTRIBUTES, LAYERS, 'lith')
  expect(on).toEqual({ colorBy: 'lith', compositionReturn: null })
  expect(switchComposition(on, ATTRIBUTES, LAYERS, null)).toEqual({ colorBy: null })
  // the layer was chosen with "Colour by": nothing remembered, Off takes the first grade
  expect(switchComposition({ colorBy: 'weathering' }, ATTRIBUTES, LAYERS, null)).toEqual({ colorBy: 'au' })
  // choosing another layer from there still remembers nothing
  expect(switchComposition({ colorBy: 'weathering' }, ATTRIBUTES, LAYERS, 'lith')).toEqual({ colorBy: 'lith' })
  // the remembered attribute left the model: the fallback
  expect(switchComposition({ colorBy: 'lith', compositionReturn: 'gone' }, ATTRIBUTES, LAYERS, null)).toEqual({ colorBy: 'au' })
  expect(fallbackColorBy([{ name: 'lith', type: 'categorical' }, { name: 'rock', type: 'categorical' }], compositionLayers([{ name: 'lith', type: 'categorical' }], null))).toBe('rock')
  expect(fallbackColorBy([{ name: 'lith', type: 'categorical' }], compositionLayers([{ name: 'lith', type: 'categorical' }], null))).toBeNull()
  // a layer the model does not have changes nothing
  const noFaults = compositionLayers(ATTRIBUTES.filter((a) => a.name !== 'fault_block'), null)
  expect(switchComposition({ colorBy: 'au' }, ATTRIBUTES, noFaults, 'fault_block')).toEqual({ colorBy: 'au' })
})

test('the manifest entry is read defensively', () => {
  expect(parseComposition({ geology: 'g', layers: { lith: 'lith', weathering: 5 } })).toEqual({ geology: 'g', layers: { lith: 'lith' } })
  expect(parseComposition({ layers: {} })).toEqual({ geology: null, layers: {} })
  for (const bad of [undefined, null, 'lith', { geology: 'g' }, { layers: 'lith' }]) expect(parseComposition(bad)).toBeNull()
})

test('the legend of a layer shows the names and the colours of the dictionary', () => {
  expect(categoryLabel('SST', { SST: 'Sandstone' })).toBe('Sandstone (SST)')
  expect(categoryLabel('AIR', { SST: 'Sandstone' })).toBe('AIR')
  expect(categoryLabel('FB1', undefined)).toBe('FB1')
  expect(categoryLabel('toString', {})).toBe('toString')
  expect(categoryLabel('Q4', { Q4: 'Q4' })).toBe('Q4')
  const info = {
    table: 'blocks', name: 'lith', type: 'categorical' as const, categories: ['SST', 'DIO', 'AIR'], count: 10, missing: 0,
    colours: { SST: '#c9b28f', DIO: '#7f8c5a' }, labels: { SST: 'Sandstone', DIO: 'Diorite' },
  }
  const spec = blockLegend(info, undefined, true)
  if (!spec) throw new Error('no legend')
  const rows = legendRows(spec, 'dark')
  expect(rows.map((r) => r.label)).toEqual(['Sandstone (SST)', 'Diorite (DIO)', 'AIR', 'No value'])
  expect(rows.slice(0, 2).map((r) => r.color)).toEqual(['#c9b28f', '#7f8c5a'])
  // without names the legend is as before
  const plain = blockLegend({ ...info, labels: undefined }, undefined, false)
  expect(plain && legendRows(plain, 'dark').map((r) => r.label)).toEqual(['SST', 'DIO', 'AIR'])
})

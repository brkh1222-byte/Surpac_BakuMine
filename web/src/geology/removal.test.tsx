/**
 * Taking the composition off: the texts of the two removals, the requests, the block
 * store's composition switch, and the controls drawn to static markup (no browser).
 */
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, expect, test } from 'vitest'
import type { BlockManifestObject } from '../api/blocks'
import { useBlocks } from '../state/blocks'
import { activeComposition, compositionLayers } from '../blocks/composition'
import { CompositionButtons } from '../ui/ObjectControls'
import { RemoveGeologyConfirm, UncodeResultView } from '../ui/ops/GeologyPanel'
import { operationLabel } from '../ops/joblog'
import { removeGeology, removeGeologyNote, removeGeologyWarning, uncodeBlockModel, uncodeBlocked, uncodeNotes, uncodePreview, type UncodeResult } from './geology'

const text = (html: string) => html.replace(/<[^>]+>/g, ' ').replace(/&quot;/g, '"').replace(/\s+/g, ' ')

const CODED = {
  geology_coding: {
    geology: 'layered-small',
    attributes: [
      { name: 'fault_block', role: 'fault_block', created: true },
      { name: 'lith', role: 'lith', created: true },
      { name: 'density', role: 'density', created: false },
    ],
  },
}

test('the confirmation names the model and the block models that keep their coding', () => {
  const lines = removeGeologyWarning('Layered small', ['model', 'owned'])
  expect(lines).toHaveLength(3)
  expect(lines[0]).toContain('Remove "Layered small" from the project?')
  expect(lines[1]).toContain('(model, owned) keep their attributes')
  expect(lines[2]).toContain('Undo in the job log')
  expect(removeGeologyWarning('G', [])).toHaveLength(2)
  expect(removeGeologyNote({ removed: 'g', name: 'G', path: 'geology/g', undoable: true, coded_models: [] })).toBe('Removed "G". Undo is in the job log.')
  expect(removeGeologyNote({ removed: 'g', name: 'G', path: 'geology/g', undoable: false, coded_models: [] })).toContain('cannot be brought back')
})

test('Remove coding says what it drops and what it keeps, or why it cannot run', () => {
  expect(uncodeBlocked(CODED)).toBeNull()
  expect(uncodePreview(CODED)).toBe('Remove coding drops fault_block, lith and keeps density (the model had it before).')
  expect(uncodeBlocked({})).toBe('This block model has no geology coding on record.')
  expect(uncodeBlocked(undefined)).toBe('Choose a block model.')
  expect(uncodePreview({})).toBeNull()
  const own = { geology_coding: { geology: null, attributes: [{ name: 'density', role: 'density', created: false }] } }
  expect(uncodeBlocked(own)).toContain('nothing to remove')
  const result: UncodeResult = { model: 'm', removed: ['lith', 'weathering'], kept: { density: 'the model had it' }, n_merged: 4321, n_blocks: 12400, geology: ['g'] }
  expect(uncodeNotes(result)).toEqual([
    'Removed lith, weathering. 12 400 blocks (4 321 sets of sub-blocks merged back). Undo is in the job log.',
    'Kept density: the model had it.',
  ])
  const plain = text(renderToStaticMarkup(<UncodeResultView result={result} />))
  expect(plain).toContain('Removed lith, weathering.')
  expect(plain).toContain('Kept density: the model had it.')
})

test('the two requests post the ids and pass the server\'s refusal on', async () => {
  const calls: { url: string; body: unknown }[] = []
  const ok = (async (url: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)) })
    return new Response(JSON.stringify({ removed: [] }), { status: 200 })
  }) as typeof fetch
  await uncodeBlockModel('model', ok)
  await removeGeology('layered-small', ok)
  expect(calls).toEqual([
    { url: '/api/v1/ops/geology/uncode', body: { model: 'model' } },
    { url: '/api/v1/ops/geology/remove', body: { geology: 'layered-small' } },
  ])
  const refuse = (async () => new Response(JSON.stringify({ detail: "no geology model 'nope' in the project" }), { status: 400 })) as typeof fetch
  await expect(removeGeology('nope', refuse)).rejects.toThrow("no geology model 'nope' in the project")
})

test('the confirmation is drawn in the page, with Remove and Cancel', () => {
  const html = renderToStaticMarkup(
    <RemoveGeologyConfirm id="g" lines={removeGeologyWarning('Layered small', ['model'])} busy={false} error={null} onRemove={() => {}} onCancel={() => {}} />,
  )
  expect(html).toContain('data-testid="geology-remove-confirm-g"')
  expect(html).toContain('data-testid="geology-remove-yes-g"')
  expect(html).toContain('data-testid="geology-remove-no-g"')
  expect(text(html)).toContain('Remove "Layered small" from the project?')
  const busy = renderToStaticMarkup(<RemoveGeologyConfirm id="g" lines={[]} busy error="no geology model 'g' in the project" onRemove={() => {}} onCancel={() => {}} />)
  expect(busy).toContain('Removing…')
  expect(busy).toContain('role="alert"')
  expect(busy.match(/disabled=""/g)).toHaveLength(2)
})

test('the job log names the two removals', () => {
  expect(operationLabel('geology_uncode')).toBe('Remove geology coding')
  expect(operationLabel('geology_remove')).toBe('Remove geology model')
})

// ---- the block store and the control ---------------------------------------------------

const DEFINITION = { origin: [0, 0, 0], blockSize: [1, 1, 1], shape: [1, 1, 1], maxDepth: [0, 0, 0] } as never

function manifest(composition: unknown, names = ['rock', 'au', 'lith', 'weathering']): BlockManifestObject {
  const attributes = names.map((name) =>
    name === 'au' ? { table: 'blocks', name, type: 'numeric', unit: 'g/t', min: 0, max: 3 } : { table: 'blocks', name, type: 'categorical', categories: ['A', 'B'], count: 2, missing: 0 },
  )
  return { id: 'model', name: 'Model', kind: 'blocks', attributes, composition } as unknown as BlockManifestObject
}

beforeEach(() => useBlocks.setState({ models: {}, status: {}, pick: null }))

test('the store switches the colouring to a layer and back, and keeps "Colour by" working', () => {
  useBlocks.getState().register(manifest({ geology: 'g', layers: { lith: 'lith', weathering: 'weathering' } }), DEFINITION)
  const model = () => useBlocks.getState().models.model
  expect(model().colorBy).toBe('au')
  expect(model().composition).toEqual({ geology: 'g', layers: { lith: 'lith', weathering: 'weathering' } })
  useBlocks.getState().setComposition('model', 'weathering')
  expect([model().colorBy, model().compositionReturn]).toEqual(['weathering', 'au'])
  // a manifest reload with the same attributes keeps the choice
  useBlocks.getState().register(manifest({ geology: 'g', layers: { lith: 'lith', weathering: 'weathering' } }), DEFINITION)
  expect(model().colorBy).toBe('weathering')
  useBlocks.getState().setComposition('model', null)
  expect([model().colorBy, model().compositionReturn]).toEqual(['au', undefined])
  // "Colour by" away from a layer: the composition is off, and Off then changes nothing
  useBlocks.getState().setComposition('model', 'lith')
  useBlocks.getState().setColorBy('model', 'rock')
  useBlocks.getState().setComposition('model', null)
  expect(model().colorBy).toBe('rock')
  // the coding was removed: the attributes changed, the display starts again without layers
  useBlocks.getState().register(manifest(undefined, ['rock', 'au']), DEFINITION)
  expect([model().colorBy, model().composition, model().compositionReturn]).toEqual(['au', null, undefined])
})

test('the control lists Off and the layers the model has, with the active one pressed', () => {
  useBlocks.getState().register(manifest({ geology: 'g', layers: { lith: 'lith', weathering: 'weathering' } }), DEFINITION)
  const draw = () => {
    const model = useBlocks.getState().models.model
    const layers = compositionLayers(model.attributes, model.composition)
    return renderToStaticMarkup(<CompositionButtons id="model" layers={layers} active={activeComposition(model.colorBy, layers)} onChoose={() => {}} />)
  }
  let html = draw()
  expect(text(html)).toContain('Composition Off Rock unit Weathering')
  expect(html).not.toContain('Fault block')
  expect(html).toContain('data-active="off"')
  expect(html).toMatch(/data-testid="composition-off-model" aria-pressed="true"/)
  useBlocks.getState().setComposition('model', 'lith')
  html = draw()
  expect(html).toContain('data-active="lith"')
  expect(html).toMatch(/data-testid="composition-lith-model" aria-pressed="true"/)
  expect(html).toMatch(/data-testid="composition-off-model" aria-pressed="false"/)
  // a model without composition layers has no such control
  useBlocks.getState().register(manifest(undefined, ['rock', 'au']), DEFINITION)
  expect(draw()).toBe('')
})

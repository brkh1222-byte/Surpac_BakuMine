/**
 * Pure logic of the geology tree and panel. The fixture holds the API's real responses
 * for the synthetic layered-small deposit (built and coded through the endpoints), so the
 * types here are checked against what the server sends. Synthetic data only.
 */
import { expect, test } from 'vitest'
import type { ManifestObject } from '../api/client'
import fixtureText from './fixtures/layered-small.json?raw'
import {
  NO_STAGE,
  PROVENANCE_SECTIONS,
  VOLUME_COLUMNS,
  codeBlockModel,
  entryCount,
  entryValue,
  fetchGeology,
  formatNumber,
  formatShare,
  geologyTree,
  gridText,
  groupState,
  parseChildId,
  pooledText,
  provenanceCounts,
  provenanceStages,
  provenanceView,
  residualRows,
  sectionHeading,
  summaryRows,
  throwRows,
  volumeCells,
  volumeRows,
  type CodeResult,
  type GeologyDetail,
  type GeologyProvenance,
  type ProvenanceRecord,
} from './geology'

const fixture = JSON.parse(fixtureText) as {
  manifest_objects: ManifestObject[]
  detail: GeologyDetail
  provenance: GeologyProvenance
  code: CodeResult
}

test('child ids are split from the right: the model id may contain dots', () => {
  expect(parseChildId('layered-small.unit.SST')).toEqual({ parent: 'layered-small', kind: 'unit', name: 'SST' })
  expect(parseChildId('pit.v2.2026.fault.F1')).toEqual({ parent: 'pit.v2.2026', kind: 'fault', name: 'F1' })
  expect(parseChildId('a.surface.b.surface.base_OXIDE')).toEqual({ parent: 'a.surface.b', kind: 'surface', name: 'base_OXIDE' })
  // a model whose own id ends like a child id is read as the child: the manifest's `parent` decides
  expect(parseChildId('north.unit.x.unit.AND')?.parent).toBe('north.unit.x')
  for (const id of ['layered-small', 'topo', 'a.b', 'a.solid.B', '.unit.SST', 'unit.SST', 'a.unit.', 'a..unit.B.']) {
    expect(parseChildId(id), id).toBeNull()
  }
})

test('every child id of the real manifest parses to its parent and group', () => {
  const children = fixture.manifest_objects.filter((o) => o.parent)
  expect(children).toHaveLength(9)
  const groups = { unit: 'Units', fault: 'Faults', surface: 'Surfaces' }
  for (const child of children) {
    const parsed = parseChildId(child.id)
    expect(parsed?.parent, child.id).toBe(child.parent)
    expect(groups[parsed!.kind], child.id).toBe(child.group)
    expect(child.kind).toBe('mesh')
    expect(child.category).toBe('geology')
    // faults carry no colour; units and surfaces carry the dictionary's
    expect(child.style?.colour === undefined, child.id).toBe(parsed!.kind === 'fault')
  }
})

test('manifest entries are grouped under their model: Units, Faults, Surfaces', () => {
  const tree = geologyTree(fixture.manifest_objects)
  expect(tree.orphans).toEqual([])
  expect(tree.models.map((m) => m.model.id)).toEqual(['layered-small'])
  const groups = tree.models[0].groups
  expect(groups.map((g) => g.name)).toEqual(['Units', 'Faults', 'Surfaces'])
  expect(groups[0].members.map((m) => parseChildId(m.id)!.name)).toEqual(['SST', 'SLT', 'AND', 'DIO', 'GRD'])
  expect(groups[1].members.map((m) => m.name)).toEqual(['Fault F1'])
  expect(groups[2].members.map((m) => m.id)).toEqual([
    'layered-small.surface.base_Q4',
    'layered-small.surface.base_OXIDE',
    'layered-small.surface.base_TRANSITION',
  ])
  // the cover unit has no mesh: only its base surface is listed
  expect(fixture.detail.units.find((u) => u.code === 'Q4')?.mesh).toBeNull()
  // every mesh the detail names is in the manifest
  const ids = new Set(fixture.manifest_objects.map((o) => o.id))
  for (const item of [...fixture.detail.units, ...fixture.detail.faults, ...fixture.detail.surfaces]) {
    if (item.mesh !== null) expect(ids.has(item.mesh), item.mesh).toBe(true)
  }
})

test('grouping does not depend on list order, falls back to the id, and keeps orphans', () => {
  const entry = (id: string, extra: object = {}) => ({ id, name: id, category: 'geology', ...extra })
  const tree = geologyTree([
    entry('g.surface.base_OX', { order: 9 }),
    entry('g.unit.B', { parent: 'g', group: 'Units', order: 5 }),
    entry('g.fault.F1', { order: 7 }),
    entry('g.unit.A', { parent: 'g', group: 'Units', order: 4 }),
    entry('g', { order: 3 }),
    entry('gone.unit.A', { parent: 'gone', group: 'Units' }),
    entry('h.v2'),
    { id: 'topo', name: 'Topography', category: 'surfaces' },
  ])
  expect(tree.models.map((m) => m.model.id)).toEqual(['g', 'h.v2'])
  expect(tree.models[0].groups.map((g) => [g.name, g.members.map((m) => m.id)])).toEqual([
    ['Units', ['g.unit.A', 'g.unit.B']],
    ['Faults', ['g.fault.F1']],
    ['Surfaces', ['g.surface.base_OX']],
  ])
  expect(tree.models[1].groups).toEqual([])
  expect(tree.orphans.map((o) => o.id)).toEqual(['gone.unit.A'])
  // an id that only looks like a child, with no such model in the list, is a model
  expect(geologyTree([entry('x.unit.A')]).models.map((m) => m.model.id)).toEqual(['x.unit.A'])
})

test('a group tick box reflects its members', () => {
  expect(groupState([{ visible: true }, { visible: true }])).toBe('all')
  expect(groupState([{ visible: true }, { visible: false }])).toBe('some')
  expect(groupState([{ visible: false }])).toBe('none')
})

const record: ProvenanceRecord = {
  entries: [
    { item: 'Rock-type log', kind: 'measured', source: 'table geology', used_for: 'rock units', stage: 'lithology', count: 3412 },
    { item: 'Topography', kind: 'measured', source: 'file topo.obj', used_for: 'air', stage: 'topography', count: 0 },
    { item: 'Fault F1', kind: 'assumed', source: 'inferred', used_for: 'fault blocks', stage: 'faults' },
    { item: 'Default density', kind: 'assumed', source: 'dictionary', used_for: 'tonnes', stage: 'units', value: 2.7 },
    { item: 'Default for max_nodes', kind: 'assumed', source: 'default', used_for: 'grid', detail: '20000000' },
    { item: 'Fault surfaces', kind: 'missing', source: 'the geologist', used_for: 'faults', stage: 'faults', count: 1 },
    { item: 'Odd', kind: 'guessed', source: '-', used_for: '-', stage: 'zzz' },
  ],
}

test('provenance is split into the three sections, in order, with counts', () => {
  expect(PROVENANCE_SECTIONS.map((s) => s.title)).toEqual(['Real data used', 'Assumed by the software', 'Inputs you still need to provide'])
  const view = provenanceView(record)
  expect(view.sections.map((s) => [s.kind, s.total, s.entries.length])).toEqual([
    ['measured', 2, 2],
    ['assumed', 3, 3],
    ['missing', 1, 1],
  ])
  expect(view.otherKinds).toBe(1)
  expect(view.sections.map((s) => sectionHeading(s, false))).toEqual([
    'Real data used (2)',
    'Assumed by the software (3)',
    'Inputs you still need to provide (1)',
  ])
  expect(provenanceCounts(record)).toEqual({ measured: 2, assumed: 3, missing: 1 })
  expect(provenanceCounts(null)).toEqual({ measured: 0, assumed: 0, missing: 0 })
  expect(provenanceView(undefined).sections.every((s) => s.entries.length === 0)).toBe(true)
})

test('the stage filter keeps the totals and narrows the entries', () => {
  expect(provenanceStages([record])).toEqual(['units', 'lithology', 'faults', 'topography', 'zzz', NO_STAGE])
  const faults = provenanceView(record, 'faults')
  expect(faults.sections.map((s) => s.entries.map((e) => e.item))).toEqual([[], ['Fault F1'], ['Fault surfaces']])
  expect(faults.sections.map((s) => sectionHeading(s, true))).toEqual([
    'Real data used (0 of 2)',
    'Assumed by the software (1 of 3)',
    'Inputs you still need to provide (1 of 1)',
  ])
  // entries without a stage have their own filter value
  expect(provenanceView(record, NO_STAGE).sections[1].entries.map((e) => e.item)).toEqual(['Default for max_nodes'])
  // a record without stages offers no filter
  expect(provenanceStages([{ entries: [{ item: 'a', kind: 'measured', source: 's', used_for: 'u' }] }])).toEqual([])
  expect(provenanceStages([null, undefined])).toEqual([])
})

test('count and value of an entry: zero is shown, absent is not', () => {
  expect(entryCount(record.entries[0])).toBe('3 412')
  expect(entryCount(record.entries[1])).toBe('0')
  expect(entryCount(record.entries[2])).toBe('')
  expect(entryCount({ ...record.entries[2], count: null })).toBe('')
  expect(entryValue(record.entries[3])).toBe('2.70')
  expect(entryValue(record.entries[0])).toBeNull()
  expect(entryValue({ ...record.entries[0], value: 'linear' })).toBe('linear')
  expect(entryValue({ ...record.entries[0], value: 5 })).toBe('5')
})

test('the real record: counts per kind, stages, and the coding record', () => {
  const { build, coding } = fixture.provenance
  expect(provenanceCounts(build)).toEqual({ measured: 34, assumed: 56, missing: 3 })
  expect(provenanceView(build).otherKinds).toBe(0)
  expect(Object.keys(coding)).toEqual(['model'])
  expect(provenanceCounts(coding.model)).toEqual({ measured: 7, assumed: 1, missing: 0 })
  // the same counts the coding call returned
  expect(fixture.code.provenance).toEqual({ measured: 7, assumed: 1, missing: 0 })
  expect(provenanceStages([build, coding.model])).toEqual(['units', 'surfaces', 'lithology', 'faults', 'coding', 'density', 'topography', 'extent', NO_STAGE])
  // the filtered sections add up to the totals
  for (const kind of [0, 1, 2]) {
    const total = provenanceView(build).sections[kind].total
    const parts = provenanceStages([build]).map((s) => provenanceView(build, s).sections[kind].entries.length)
    expect(parts.reduce((a, b) => a + b, 0)).toBe(total)
  }
  expect(provenanceView(build).sections[2].entries.map((e) => e.item)).toEqual(['Density samples', 'base:Q4 not logged', 'Fault surfaces'])
  for (const entry of [...build.entries, ...coding.model.entries]) {
    expect(typeof entry.item).toBe('string')
    expect(typeof entry.source).toBe('string')
    expect(typeof entry.used_for).toBe('string')
  }
})

test('numbers: spaces between thousands, an en dash for no value', () => {
  expect(formatNumber(11758375)).toBe('11 758 375')
  expect(formatNumber(2.4928, 2)).toBe('2.49')
  expect(formatNumber(0)).toBe('0')
  expect(formatNumber(null)).toBe('–')
  expect(formatNumber(undefined)).toBe('–')
  expect(formatNumber(NaN)).toBe('–')
  expect(formatShare(0.3152433903494318)).toBe('31.5 %')
  expect(formatShare(0)).toBe('0.0 %')
  expect(formatShare(null)).toBe('–')
})

test('volume rows keep the drilled and the extrapolated volume apart, nulls stay empty', () => {
  expect(VOLUME_COLUMNS).toHaveLength(5)
  const rows = volumeRows(
    {
      units: {
        B: { volume: 1000, inside_hull: null, extrapolated: null, extrapolated_share: null },
        A: { volume: 3000000, inside_hull: 2250000, extrapolated: 750000, extrapolated_share: 0.25 },
        Z: { volume: null, inside_hull: null, extrapolated: null, extrapolated_share: null },
      },
    },
    ['A', 'B', 'C'],
  )
  expect(rows.map((r) => r.unit)).toEqual(['A', 'B', 'Z'])
  expect(volumeCells(rows[0])).toEqual(['A', '3 000 000', '2 250 000', '750 000', '25.0 %'])
  // without a hull nothing is invented: the whole-box volume alone, the rest empty
  expect(volumeCells(rows[1])).toEqual(['B', '1 000', '–', '–', '–'])
  expect(volumeCells(rows[2])).toEqual(['Z', '–', '–', '–', '–'])
  expect(volumeRows(null)).toEqual([])
  expect(volumeRows(undefined, ['A'])).toEqual([])
})

test('the real volumes: lith order, and inside + extrapolated = whole box', () => {
  const { detail } = fixture
  const rows = volumeRows(detail.diagnostics?.volumes, detail.layers.lith)
  expect(rows.map((r) => r.unit)).toEqual(['SST', 'SLT', 'AND', 'DIO', 'GRD', 'Q4', 'AIR'])
  for (const row of rows) {
    expect(row.insideHull! + row.extrapolated!).toBeCloseTo(row.volume!, 6)
    expect(row.share!).toBeCloseTo(row.extrapolated! / row.volume!, 12)
  }
  expect(volumeCells(rows[0])).toEqual(['SST', '11 758 375', '8 051 625', '3 706 750', '31.5 %'])
})

test('throw and residual rows say what the record says', () => {
  const rows = throwRows(fixture.detail.diagnostics?.throw)
  expect(rows.map((r) => [r.fault, r.contact, r.throwText, r.basis])).toEqual([
    ['F1', 'SST / SLT', '26.2 ± 1.0 m', 'measured'],
    ['F1', 'SLT / AND', '35.7 ± 1.5 m', 'measured'],
  ])
  expect(throwRows([{ fault: 'F2', upper: 'A', lower: 'B', throw: null, kind: 'assumed', note: 'pooled' }])[0]).toEqual({
    fault: 'F2', contact: 'A / B', throwText: '–', basis: 'assumed', note: 'pooled',
  })
  expect(throwRows([{ fault: 'F3', upper: 'A', lower: 'B', throw: 4 }])[0].basis).toBe('not stated')
  expect(throwRows(null)).toEqual([])
  const residuals = residualRows(fixture.detail.diagnostics?.residuals, fixture.detail.layers.lith)
  expect(residuals.map((r) => r.unit)).toEqual(['SST', 'SLT', 'AND', 'DIO', 'GRD'])
  expect(residuals[0].points).toBe(2094)
  expect(formatShare(residuals[0].reproduced)).toBe('98.7 %')
  expect(residualRows(null)).toEqual([])
})

test('the summary of the real model', () => {
  expect(gridText(fixture.detail.grid)).toBe('81 x 81 x 63 nodes at 5 m')
  expect(gridText({ origin: [0, 0, 0], spacing: [10, 10, 5], shape: [4, 5, 6] })).toBe('4 x 5 x 6 nodes at 10 x 10 x 5 m')
  expect(gridText(null)).toBe('–')
  expect(pooledText([])).toBe('none')
  expect(pooledText(['FB3', 'FB4'])).toBe('FB3; FB4')
  const rows = Object.fromEntries(summaryRows(fixture.detail).map((r) => [r.label, r.text]))
  expect(rows).toEqual({
    Grid: '81 x 81 x 63 nodes at 5 m',
    Units: '6: SST, SLT, AND, DIO, GRD, Q4',
    'Fault blocks': '2: FB1, FB2',
    'Pooled blocks': 'none',
    Faults: 'F1',
    Surfaces: 'base:Q4, base:OXIDE, base:TRANSITION',
    Drilling: '64 holes, 717 logged intervals',
  })
})

test('requests: the id is encoded, a refusal shows the server text', async () => {
  const calls: { url: string; init?: RequestInit }[] = []
  const ok = (async (url: string, init?: RequestInit) => {
    calls.push({ url, init })
    return new Response(JSON.stringify(fixture.code), { status: 200 })
  }) as typeof fetch
  await fetchGeology('north pit.v2', ok)
  expect(calls[0].url).toBe('/api/v1/geology/north%20pit.v2')
  const result = await codeBlockModel({ model: 'model', geology: 'layered-small' }, ok)
  expect(calls[1].url).toBe('/api/v1/ops/geology/code')
  expect(calls[1].init?.method).toBe('POST')
  expect(JSON.parse(String(calls[1].init?.body))).toEqual({ model: 'model', geology: 'layered-small' })
  expect(result.tonnage?.at(-1)).toEqual({ unit: 'AIR', blocks: 5125, volume: 8778500, tonnes: 0, density: null })

  const refused = (async () => new Response(JSON.stringify({ detail: "no block model 'nope'" }), { status: 400 })) as typeof fetch
  await expect(codeBlockModel({ model: 'nope', geology: 'g' }, refused)).rejects.toThrow("no block model 'nope'")
  const broken = (async () => new Response('oops', { status: 500 })) as typeof fetch
  await expect(fetchGeology('g', broken)).rejects.toThrow('Request failed (500)')
  const down = (async () => {
    throw new TypeError('fetch failed')
  }) as typeof fetch
  await expect(fetchGeology('g', down)).rejects.toThrow('not reachable')
})

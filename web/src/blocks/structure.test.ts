/**
 * Block reconstruction against Python: tests/api/block_fixtures.py writes
 * fixtures/blocks.json from `BlockModel` (dense, sub-blocked with unequal depths, and a
 * rotated sub-blocked model at mine coordinates).
 */
import { expect, test } from 'vitest'
import { decodeBkma } from '../api/bkma'
import { blockQueryString, decodeBlocks, fetchBlocks, fetchBlockDetail } from '../api/blocks'
import { ApiError } from '../api/client'
import { CASES, structureOf, type FixtureCase } from './fixtures/cases'
import {
  blockGeometry,
  blockId,
  blockSizes,
  centroids,
  centroidsLocal,
  denseStructure,
  levelSizes,
  maxLevel,
  modelBounds,
  parentIjk,
  parseDefinition,
  sceneCentres,
  worldToLocal,
} from './structure'

function payloadOf(c: FixtureCase): ArrayBuffer {
  return Uint8Array.from(atob(c.payload_base64), (ch) => ch.charCodeAt(0)).buffer
}

test('the fixture has a dense, a sub-blocked and a rotated model', () => {
  expect(CASES.map((c) => c.name)).toEqual(['dense', 'octree', 'rotated'])
  expect(parseDefinition(CASES[0].definition).rotated).toBe(false)
  expect(parseDefinition(CASES[1].definition).maxDepth).toEqual([2, 1, 0])
  expect(parseDefinition(CASES[2].definition).rotated).toBe(true)
  expect(new Set(CASES[1].level)).toEqual(new Set([0, 1, 2]))
})

for (const c of CASES) {
  test(`${c.name}: centroids and sizes equal BlockModel's, to the last bit`, () => {
    const definition = parseDefinition(c.definition)
    const structure = structureOf(c)
    // Float64, same operations in the same order: exact equality, not a tolerance.
    expect(Array.from(centroidsLocal(definition, structure))).toEqual(c.centroids_local)
    expect(Array.from(centroids(definition, structure))).toEqual(c.centroids)
    expect(Array.from(blockSizes(definition, structure))).toEqual(c.sizes)
    const sizes = blockSizes(definition, structure)
    c.volumes.forEach((v, b) => expect(sizes[b * 3] * sizes[b * 3 + 1] * sizes[b * 3 + 2]).toBe(v))
  })

  test(`${c.name}: ijk, ids and single-block geometry`, () => {
    const definition = parseDefinition(c.definition)
    const structure = structureOf(c)
    for (let b = 0; b < c.n_blocks; b++) {
      expect(parentIjk(definition, c.parent[b])).toEqual(c.ijk.slice(b * 3, b * 3 + 3))
      expect(blockId(definition, c.parent[b], c.level[b], c.sub.slice(b * 3, b * 3 + 3))).toBe(c.block_ids[b])
    }
    const last = c.n_blocks - 1
    const g = blockGeometry(definition, structure, last)
    expect(g.centroid).toEqual(c.centroids.slice(last * 3))
    expect(g.size).toEqual(c.sizes.slice(last * 3))
    expect(g.level).toBe(c.level[last])
    // World -> model frame undoes the reconstruction.
    const back = worldToLocal(definition, g.centroid)
    back.forEach((v, a) => expect(v).toBeCloseTo(c.centroids_local[last * 3 + a], 8))
  })

  test(`${c.name}: scene centres are small float32 numbers, right to float precision`, () => {
    const definition = parseDefinition(c.definition)
    const structure = structureOf(c)
    const origin: [number, number, number] = [500500, 5300400, 900]
    const centres = sceneCentres(definition, structure, origin)
    expect(centres).toBeInstanceOf(Float32Array)
    let largest = 0
    for (let i = 0; i < centres.length; i++) {
      const exact = c.centroids[i] - origin[i % 3]
      largest = Math.max(largest, Math.abs(centres[i]))
      // Half a float32 ulp at a few hundred metres is 3e-5 m.
      expect(Math.abs(centres[i] - exact)).toBeLessThan(4e-5)
    }
    expect(largest).toBeLessThan(1e3)
  })

  test(`${c.name}: the payload decodes to the structure and attribute columns Python sent`, () => {
    const set = decodeBlocks(decodeBkma(payloadOf(c)))
    const arrays = c.payload_arrays
    expect(set.count).toBe(c.shown_rows.length)
    expect(set.total).toBe(c.n_blocks)
    expect(set.selected).toBe(c.shown_rows.length)
    expect(set.lod).toBe('full')
    expect(Array.from(set.rows ?? [])).toEqual(c.shown_rows)
    expect(Array.from(set.structure.parent)).toEqual(c.shown_rows.map((r) => c.parent[r]))
    expect(Array.from(set.structure.level)).toEqual(c.shown_rows.map((r) => c.level[r]))
    expect(Array.from(set.structure.sub)).toEqual(c.shown_rows.flatMap((r) => c.sub.slice(r * 3, r * 3 + 3)))
    if (c.name === 'dense') expect(Object.keys(arrays)).not.toContain('parent')
    else expect(Array.from(set.structure.parent)).toEqual(arrays.parent.data)
    // Geometry of the selection = the model's geometry at those rows.
    expect(Array.from(centroids(set.definition, set.structure))).toEqual(c.shown_rows.flatMap((r) => c.centroids.slice(r * 3, r * 3 + 3)))

    const [au, rock] = set.columns
    expect(au.type === 'numeric' && au.unit).toBe('g/t')
    expect(rock.type === 'categorical' && rock.categories).toEqual(['AIR', 'WASTE', 'ORE'])
    if (au.type !== 'numeric' || rock.type !== 'categorical') throw new Error('unexpected column types')
    expect(au.values).toBeInstanceOf(Float32Array)
    expect(Array.from(au.values, (v) => (Number.isNaN(v) ? null : v))).toEqual(arrays['value:au'].data)
    expect(arrays['value:au'].data).toContain(null)
    expect(Array.from(rock.codes)).toEqual(arrays['code:rock'].data)
    expect(set.bytesPerBlock).toBe(c.payload_meta.bytes_per_block)
  })
}

test('helpers: level sizes, dense structure, bounds', () => {
  const octree = parseDefinition(CASES[1].definition)
  expect(maxLevel(octree)).toBe(2)
  expect(Array.from(levelSizes(octree))).toEqual([10, 10, 5, 5, 5, 5, 2.5, 5, 5])
  const all = denseStructure(null, 4)
  expect(Array.from(all.parent)).toEqual([0, 1, 2, 3])
  expect(denseStructure(Uint32Array.of(7, 9), 2).parent).toEqual(Uint32Array.of(7, 9))
  const bounds = modelBounds(parseDefinition(CASES[0].definition))
  expect(bounds.min).toEqual([500400, 5300250, 825])
  expect(bounds.max).toEqual([500430, 5300290, 835])
  // A rotated model's box encloses every centroid.
  const rotated = modelBounds(parseDefinition(CASES[2].definition))
  CASES[2].centroids.forEach((v, i) => {
    expect(v).toBeGreaterThan(rotated.min[i % 3])
    expect(v).toBeLessThan(rotated.max[i % 3])
  })
  expect(() => parseDefinition({ origin: [0, 0], block_size: [1, 1, 1], shape: [1, 1, 1] })).toThrow(/origin/)
})

test('block requests: query string, decoding and the server error message', async () => {
  expect(blockQueryString({ attributes: [], where: null, lod: 'auto', clip: null })).toBe('')
  const text = blockQueryString({ attributes: ['au', 'rock'], where: 'rock == "ORE"', lod: 'full', clip: '{"type":"plane"}' })
  expect(Object.fromEntries(new URLSearchParams(text))).toEqual({ attributes: 'au,rock', where: 'rock == "ORE"', lod: 'full', clip: '{"type":"plane"}' })

  const seen: string[] = []
  const ok = (async (url: string) => {
    seen.push(url)
    return new Response(payloadOf(CASES[1]), { status: 200 })
  }) as unknown as typeof fetch
  const set = await fetchBlocks({ href: '/api/v1/scene/objects/m/blocks' }, { attributes: ['au'], where: 'au > 1', lod: 'auto', clip: null }, undefined, ok)
  expect(set.count).toBe(CASES[1].shown_rows.length)
  expect(seen[0]).toBe('/api/v1/scene/objects/m/blocks?attributes=au&where=au+%3E+1')

  const bad = (async () => new Response(JSON.stringify({ detail: "unknown name 'gold'; available: au, rock" }), { status: 400 })) as unknown as typeof fetch
  const error = await fetchBlocks({ href: '/x' }, { attributes: [], where: 'gold > 1', lod: 'auto', clip: null }, undefined, bad).catch((e: unknown) => e)
  expect(error).toBeInstanceOf(ApiError)
  expect((error as ApiError).status).toBe(400)
  expect((error as ApiError).message).toContain("unknown name 'gold'")

  const detail = (async (url: string) => {
    seen.push(url)
    return new Response(JSON.stringify({ row: 5, block_id: '99', attributes: { au: 1.5 } }), { status: 200 })
  }) as unknown as typeof fetch
  expect((await fetchBlockDetail({ href: '/b' }, 5, undefined, detail)).block_id).toBe('99')
  expect(seen.at(-1)).toBe('/b/detail?row=5')
})

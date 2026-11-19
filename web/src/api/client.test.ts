/**
 * Decoder round trip against payloads written by the Python side
 * (tests/api/scene_fixtures.py -> src/api/fixtures/). `expected.json` holds the same
 * arrays as plain numbers (null = NaN).
 */
import { expect, test } from 'vitest'
import { BkmaError, decodeBkma, encodeBkma } from './bkma'
import { ApiError, decodeDrillholes, decodeMesh, fetchDrillholes, fetchManifest, fetchMesh } from './client'
import drillholesUri from './fixtures/drillholes.bkma?inline'
import expectedText from './fixtures/expected.json?raw'
import meshUri from './fixtures/mesh.bkma?inline'

interface ExpectedArray {
  dtype: string
  data: unknown
}
interface ExpectedPayload {
  kind: string
  meta: Record<string, unknown>
  arrays: Record<string, ExpectedArray>
}
const expected = JSON.parse(expectedText) as { drillholes: ExpectedPayload; mesh: ExpectedPayload }

function fromDataUri(uri: string): ArrayBuffer {
  const bytes = Uint8Array.from(atob(uri.slice(uri.indexOf(',') + 1)), (c) => c.charCodeAt(0))
  return bytes.buffer
}

function flat(data: unknown): (number | null)[] {
  return (Array.isArray(data) ? data.flat(Infinity) : [data]) as (number | null)[]
}

/** Typed array -> plain numbers with NaN as null, like expected.json. */
function plain(array: ArrayLike<number>): (number | null)[] {
  return Array.from(array, (v) => (Number.isNaN(v) ? null : v))
}

test('every array of both fixtures decodes to the values Python wrote', () => {
  for (const [uri, want] of [
    [drillholesUri, expected.drillholes],
    [meshUri, expected.mesh],
  ] as const) {
    const payload = decodeBkma(fromDataUri(uri))
    expect(payload.kind).toBe(want.kind)
    expect(payload.meta).toEqual(want.meta)
    expect(payload.names()).toEqual(Object.keys(want.arrays))
    for (const [name, array] of Object.entries(want.arrays)) {
      const entry = payload.entry(name)
      expect(entry.dtype, name).toBe(array.dtype)
      // Exact: float64 survives JSON, and float32 values are exactly representable.
      expect(plain(entry.data), name).toEqual(flat(array.data))
    }
  }
})

test('drillhole payload becomes a typed DrillholeSet', () => {
  const set = decodeDrillholes(decodeBkma(fromDataUri(drillholesUri)))
  expect(set.holeIds).toEqual(['T1', 'T2', 'T3'])
  expect(set.holeOffsets).toBeInstanceOf(Uint32Array)
  expect(set.holeOffsets[0]).toBe(0)
  expect(set.holeOffsets[3]).toBe(set.pointDepth.length)
  expect(set.pointXyz.length).toBe(set.pointDepth.length * 3)
  // T1 is vertical: two points; T2 is curved: densified between its stations.
  expect(set.holeOffsets[1]).toBe(2)
  expect(set.holeOffsets[2] - set.holeOffsets[1]).toBeGreaterThan(4)
  expect(Array.from(set.collarXyz.slice(0, 3))).toEqual([500100, 5300200, 1210.5])
  expect(Array.from(set.maxDepth)).toEqual([40, 90, 30])
  // World float64: the first polyline point of each hole is its collar, exactly.
  for (let h = 0; h < 3; h++) {
    const p = set.holeOffsets[h] * 3
    expect(Array.from(set.pointXyz.slice(p, p + 3))).toEqual(Array.from(set.collarXyz.slice(h * 3, h * 3 + 3)))
  }

  const seg = set.segments
  if (!seg) throw new Error('no segments')
  expect(seg.table).toBe('assay')
  expect(seg.count).toBe(7)
  expect(Array.from(seg.hole)).toEqual([1, 0, 0, 1, 0, 1, 2])
  expect(Array.from(seg.from)).toEqual([0, 0, 10, 45, 25, 70, 5])
  expect(Array.from(seg.to)).toEqual([20, 10, 25, 70, 40, 90, 30])
  expect(seg.start.length).toBe(21)
  // T1 is vertical from Z 1210.5: its interval 10-25 m runs from Z 1200.5 to 1185.5.
  expect(seg.start[2 * 3 + 2]).toBeCloseTo(1200.5, 9)
  expect(seg.end[2 * 3 + 2]).toBeCloseTo(1185.5, 9)

  const [au, rock] = seg.attributes
  if (au.type !== 'numeric' || rock.type !== 'categorical') throw new Error('wrong attribute types')
  expect(au.name).toBe('au')
  expect(au.unit).toBe('g/t')
  // Missing stays missing: NaN, not 0.
  expect(plain(au.values)).toEqual([0.25, 0.031, null, 3.5, 1.75, 0.4, 12.125])
  expect(rock.categories).toEqual(['DYKE', 'ORE', 'WASTE'])
  expect(Array.from(rock.codes)).toEqual([2, 2, 1, 1, -1, 0, 1])
})

test('mesh payload becomes typed MeshData', () => {
  const mesh = decodeMesh(decodeBkma(fromDataUri(meshUri)))
  expect(mesh.closed).toBe(true)
  expect(mesh.vertices).toBeInstanceOf(Float64Array)
  expect(mesh.vertices.length).toBe(12)
  expect(Array.from(mesh.faces)).toEqual([0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2])
  expect(Array.from(mesh.vertices.slice(9, 12))).toEqual([500100, 5300200, 1007.25])
  if (!mesh.normals) throw new Error('no normals')
  for (let v = 0; v < 4; v++) {
    const n = mesh.normals.subarray(v * 3, v * 3 + 3)
    expect(Math.hypot(n[0], n[1], n[2])).toBeCloseTo(1, 5)
  }
})

test('encodeBkma / decodeBkma round trip; arrays are views on the buffer', () => {
  const buffer = encodeBkma('test', { names: ['a'] }, [
    { name: 'x', dtype: 'f8', shape: [2, 3], data: new Float64Array([1, 2, 3, 4, 5, 6.5]) },
    { name: 'odd', dtype: 'u1', shape: [5], data: new Uint8Array([1, 2, 3, 4, 5]) },
    { name: 'i', dtype: 'i4', shape: [3], data: new Int32Array([-1, 0, 7]) },
    { name: 'empty', dtype: 'f4', shape: [0, 3], data: new Float32Array(0) },
  ])
  const payload = decodeBkma(buffer)
  expect(payload.meta).toEqual({ names: ['a'] })
  const x = payload.get('x', 'f8', 3)
  expect(Array.from(x)).toEqual([1, 2, 3, 4, 5, 6.5])
  expect(x.buffer).toBe(buffer)
  expect(x.byteOffset % 8).toBe(0)
  expect(payload.get('i', 'i4').byteOffset % 8).toBe(0)
  expect(Array.from(payload.get('i', 'i4'))).toEqual([-1, 0, 7])
  expect(payload.get('empty', 'f4', 3).length).toBe(0)
  expect(() => payload.get('x', 'f4', 3)).toThrow(BkmaError)
  expect(() => payload.get('x', 'f8')).toThrow(BkmaError)
  expect(() => payload.get('nope', 'f8')).toThrow(BkmaError)
})

test('malformed payloads are rejected', () => {
  const good = fromDataUri(meshUri)
  expect(() => decodeBkma(new ArrayBuffer(4))).toThrow(BkmaError)
  expect(() => decodeBkma(new TextEncoder().encode('{"not":"bkma"}  ').buffer as ArrayBuffer)).toThrow(BkmaError)
  expect(() => decodeBkma(good.slice(0, good.byteLength - 8))).toThrow(/does not fit/)
  const wrongVersion = good.slice(0)
  new DataView(wrongVersion).setUint32(4, 99, true)
  expect(() => decodeBkma(wrongVersion)).toThrow(/version/)
})

test('client: URLs, payload kinds and errors', async () => {
  const calls: string[] = []
  const serve = (body: ArrayBuffer | string, status = 200) =>
    (async (url: string) => {
      calls.push(url)
      return new Response(body, { status })
    }) as unknown as typeof fetch

  const object = { href: '/api/v1/scene/objects/holes/drillholes' }
  const set = await fetchDrillholes(object, 'assay', undefined, serve(fromDataUri(drillholesUri)))
  expect(set.segments?.count).toBe(7)
  await fetchDrillholes(object, null, undefined, serve(fromDataUri(drillholesUri)))
  const mesh = await fetchMesh({ href: '/api/v1/scene/objects/tetra/mesh' }, undefined, serve(fromDataUri(meshUri)))
  expect(mesh.faces.length).toBe(12)
  expect(calls).toEqual([
    '/api/v1/scene/objects/holes/drillholes?table=assay',
    '/api/v1/scene/objects/holes/drillholes',
    '/api/v1/scene/objects/tetra/mesh?normals=1',
  ])

  // A mesh payload where drillholes were expected.
  await expect(fetchDrillholes(object, null, undefined, serve(fromDataUri(meshUri)))).rejects.toThrow(ApiError)
  // The server's `detail` is passed on.
  await expect(fetchMesh(object, undefined, serve('{"detail":"no mesh object \'holes\'"}', 404))).rejects.toThrow(
    /404: no mesh object/,
  )
  const down = (async () => {
    throw new TypeError('fetch failed')
  }) as unknown as typeof fetch
  await expect(fetchManifest(undefined, down)).rejects.toThrow(/not reachable/)
  await expect(fetchManifest(undefined, serve('{"objects":"x"}'))).rejects.toThrow(/Bad response/)
  const manifest = await fetchManifest(undefined, serve('{"api_version":1,"project":{"id":"p","name":"P"},"version":"1","objects":[]}'))
  expect(manifest.project.name).toBe('P')
})

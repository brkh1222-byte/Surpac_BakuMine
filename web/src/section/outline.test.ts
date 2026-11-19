/// <reference types="node" />
/**
 * Mesh-plane intersection: analytic cases (cube, sphere) and the polylines the Python
 * `section()` cuts from the synthetic deposit (fixtures/sections.json, written by
 * tests/api/section_fixtures.py; the meshes are read from data/synthetic/default).
 */
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { expect, test } from 'vitest'
import fixtureText from './fixtures/sections.json?raw'
import { cutMesh, outlineCount, outlinePoints, outlineSegments, type Outline } from './outline'
import { sectionFrame } from './plane'
import type { Vec3 } from '../scene/origin'

interface FixtureCase {
  name: string
  mesh: string
  azimuth: number
  dip: number
  origin: [number, number, number]
  normal: [number, number, number]
  closed: boolean[]
  area: number
  length: number
  polylines: number[][][]
}
const fixture = JSON.parse(fixtureText) as { cases: FixtureCase[] }

/** Minimal .npy reader: little-endian f8 / i8 / i4, C order. */
function readNpy(path: string): Float64Array {
  const bytes = readFileSync(path)
  const major = bytes[6]
  const headerLength = major >= 2 ? bytes.readUInt32LE(8) : bytes.readUInt16LE(8)
  const start = (major >= 2 ? 12 : 10) + headerLength
  const header = bytes.subarray(major >= 2 ? 12 : 10, start).toString('latin1')
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1]
  if (/'fortran_order':\s*True/.test(header)) throw new Error('fortran order')
  const view = new DataView(bytes.buffer, bytes.byteOffset + start, bytes.length - start)
  if (descr === '<f8') return Float64Array.from({ length: view.byteLength / 8 }, (_, i) => view.getFloat64(i * 8, true))
  if (descr === '<i8') return Float64Array.from({ length: view.byteLength / 8 }, (_, i) => Number(view.getBigInt64(i * 8, true)))
  if (descr === '<i4') return Float64Array.from({ length: view.byteLength / 4 }, (_, i) => view.getInt32(i * 4, true))
  throw new Error(`unsupported dtype ${descr}`)
}

const DEMO = fileURLToPath(new URL('../../../data/synthetic/default/', import.meta.url))
const haveDemo = existsSync(DEMO + 'ore_solid_vertices.npy')

function loadMesh(name: string): { vertices: Float64Array; faces: Uint32Array } {
  return { vertices: readNpy(`${DEMO}${name}_vertices.npy`), faces: Uint32Array.from(readNpy(`${DEMO}${name}_faces.npy`)) }
}

/** Axis-aligned box, outward winding, 8 shared vertices. */
function box(min: Vec3, max: Vec3) {
  const vertices: number[] = []
  for (let i = 0; i < 8; i++) vertices.push(i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2])
  // prettier-ignore
  const faces = [
    0, 2, 1, 1, 2, 3, // bottom (z min)
    4, 5, 6, 5, 7, 6, // top
    0, 1, 4, 1, 5, 4, // y min
    2, 6, 3, 3, 6, 7, // y max
    0, 4, 2, 2, 4, 6, // x min
    1, 3, 5, 3, 7, 5, // x max
  ]
  return { vertices: Float64Array.from(vertices), faces: Uint32Array.from(faces) }
}

/** Latitude-longitude sphere with shared vertices, outward winding. */
function sphere(centre: Vec3, radius: number, rings: number, sectors: number) {
  const vertices: number[] = [centre[0], centre[1], centre[2] + radius]
  for (let r = 1; r < rings; r++) {
    const polar = (Math.PI * r) / rings
    for (let s = 0; s < sectors; s++) {
      const a = (2 * Math.PI * s) / sectors
      vertices.push(
        centre[0] + radius * Math.sin(polar) * Math.cos(a),
        centre[1] + radius * Math.sin(polar) * Math.sin(a),
        centre[2] + radius * Math.cos(polar),
      )
    }
  }
  vertices.push(centre[0], centre[1], centre[2] - radius)
  const south = vertices.length / 3 - 1
  const at = (r: number, s: number) => 1 + (r - 1) * sectors + (s % sectors)
  const faces: number[] = []
  for (let s = 0; s < sectors; s++) {
    faces.push(0, at(1, s), at(1, s + 1))
    faces.push(south, at(rings - 1, s + 1), at(rings - 1, s))
    for (let r = 1; r < rings - 1; r++) {
      faces.push(at(r, s), at(r + 1, s), at(r + 1, s + 1), at(r, s), at(r + 1, s + 1), at(r, s + 1))
    }
  }
  return { vertices: Float64Array.from(vertices), faces: Uint32Array.from(faces) }
}

/** Signed volume of a closed mesh: positive for outward winding. */
function volume(mesh: { vertices: Float64Array; faces: Uint32Array }): number {
  const v = mesh.vertices
  let sum = 0
  for (let i = 0; i < mesh.faces.length; i += 3) {
    const a = mesh.faces[i] * 3
    const b = mesh.faces[i + 1] * 3
    const c = mesh.faces[i + 2] * 3
    sum +=
      v[a] * (v[b + 1] * v[c + 2] - v[b + 2] * v[c + 1]) -
      v[a + 1] * (v[b] * v[c + 2] - v[b + 2] * v[c]) +
      v[a + 2] * (v[b] * v[c + 1] - v[b + 1] * v[c])
  }
  return sum / 6
}

/** Plane coordinates (u, v) of polyline k, in the frame of the section (azimuth, dip). */
function inPlane(outline: Outline, k: number, origin: Vec3, azimuth: number, dip: number): [number, number][] {
  const { u, v } = sectionFrame(azimuth, dip)
  const p = outlinePoints(outline, k)
  const out: [number, number][] = []
  for (let i = 0; i < p.length; i += 3) {
    const d = [p[i] - origin[0], p[i + 1] - origin[1], p[i + 2] - origin[2]]
    out.push([d[0] * u[0] + d[1] * u[1] + d[2] * u[2], d[0] * v[0] + d[1] * v[1] + d[2] * v[2]])
  }
  return out
}

function signedArea(points: [number, number][]): number {
  let sum = 0
  for (let i = 0; i < points.length; i++) {
    const a = points[i]
    const b = points[(i + 1) % points.length]
    sum += a[0] * b[1] - b[0] * a[1]
  }
  return sum / 2
}

function length(outline: Outline, k: number): number {
  const p = outlinePoints(outline, k)
  const n = p.length / 3
  let sum = 0
  for (let i = 0; i < (outline.closed[k] ? n : n - 1); i++) {
    const a = i * 3
    const b = ((i + 1) % n) * 3
    sum += Math.hypot(p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2])
  }
  return sum
}

const CUBE_MIN: Vec3 = [500100, 5300200, 1000]
const CUBE_MAX: Vec3 = [500110, 5300220, 1030]

test('the test meshes are closed and wound outwards', () => {
  expect(volume(box(CUBE_MIN, CUBE_MAX))).toBeCloseTo(10 * 20 * 30, 6)
  const ball = sphere([0, 0, 0], 1, 48, 96)
  expect(volume(ball)).toBeGreaterThan(4.17)
  expect(volume(ball)).toBeLessThan((4 / 3) * Math.PI)
})

test('cube: axis-aligned cuts are the exact rectangles, counter-clockwise for the viewer', () => {
  const mesh = box(CUBE_MIN, CUBE_MAX)
  // W-E section through the middle: 10 m wide, 30 m high.
  const origin: Vec3 = [500105, 5300210, 1015]
  const we = cutMesh(mesh.vertices, mesh.faces, origin, sectionFrame(90, 90).n)
  expect(outlineCount(we)).toBe(1)
  expect(we.closed[0]).toBe(1)
  const uv = inPlane(we, 0, origin, 90, 90)
  expect(signedArea(uv)).toBeCloseTo(300, 9)
  expect(length(we, 0)).toBeCloseTo(80, 9)
  for (const [s, t] of uv) {
    expect(Math.abs(Math.abs(s) - 5) < 1e-9 || Math.abs(Math.abs(t) - 15) < 1e-9).toBe(true)
  }
  const p = outlinePoints(we, 0)
  for (let i = 0; i < p.length; i += 3) expect(p[i + 1]).toBe(5300210)

  // Plan: 10 x 20; S-N: 20 x 30.
  const plan = cutMesh(mesh.vertices, mesh.faces, origin, sectionFrame(90, 0).n)
  expect(signedArea(inPlane(plan, 0, origin, 90, 0))).toBeCloseTo(200, 9)
  const sn = cutMesh(mesh.vertices, mesh.faces, origin, sectionFrame(0, 90).n)
  expect(signedArea(inPlane(sn, 0, origin, 0, 90))).toBeCloseTo(600, 9)
  expect(length(sn, 0)).toBeCloseTo(100, 9)
})

test('cube: a diagonal vertical cut and the half-open convention at its faces', () => {
  const mesh = box([0, 0, 0], [10, 10, 30])
  // Vertical plane through two opposite vertical edges: 10 * sqrt(2) wide, 30 high.
  const diagonal = cutMesh(mesh.vertices, mesh.faces, [5, 5, 15], sectionFrame(45, 90).n)
  expect(outlineCount(diagonal)).toBe(1)
  expect(signedArea(inPlane(diagonal, 0, [5, 5, 15], 45, 90))).toBeCloseTo(300 * Math.SQRT2, 9)

  // Plane in the bottom face: the solid continues on the positive side -> full rectangle.
  const bottom = cutMesh(mesh.vertices, mesh.faces, [0, 0, 0], [0, 0, 1])
  expect(outlineCount(bottom)).toBe(1)
  expect(signedArea(inPlane(bottom, 0, [0, 0, 0], 90, 0))).toBeCloseTo(100, 9)
  // Plane in the top face: the solid ends there -> nothing. And nothing outside the box.
  expect(outlineCount(cutMesh(mesh.vertices, mesh.faces, [0, 0, 30], [0, 0, 1]))).toBe(0)
  expect(outlineCount(cutMesh(mesh.vertices, mesh.faces, [0, 0, 31], [0, 0, 1]))).toBe(0)
  expect(outlineCount(cutMesh(mesh.vertices, mesh.faces, [0, 0, -1], [0, 0, 1]))).toBe(0)
})

test('sphere: every cut is a circle of radius sqrt(R^2 - d^2)', () => {
  const centre: Vec3 = [500600, 5300500, 1000]
  const radius = 50
  const mesh = sphere(centre, radius, 90, 180)
  for (const [azimuth, dip, offset] of [
    [90, 90, 0],
    [90, 90, 30],
    [33, 90, -20],
    [140, 55, 12],
    [90, 0, 0.5],
    [10, 20, 44],
  ]) {
    const { n } = sectionFrame(azimuth, dip)
    const origin: Vec3 = [centre[0] + n[0] * offset, centre[1] + n[1] * offset, centre[2] + n[2] * offset]
    const cut = cutMesh(mesh.vertices, mesh.faces, origin, n)
    expect(outlineCount(cut)).toBe(1)
    expect(cut.closed[0]).toBe(1)
    const expected = Math.sqrt(radius * radius - offset * offset)
    const p = outlinePoints(cut, 0)
    for (let i = 0; i < p.length; i += 3) {
      // On the plane, and on the faceted sphere: within the chord sag of the true circle.
      const d = [p[i] - origin[0], p[i + 1] - origin[1], p[i + 2] - origin[2]]
      expect(Math.abs(d[0] * n[0] + d[1] * n[1] + d[2] * n[2])).toBeLessThan(1e-7)
      const r = Math.hypot(...d)
      expect(r).toBeLessThanOrEqual(expected + 1e-7)
      expect(r).toBeGreaterThan(expected - 0.05)
    }
    const area = signedArea(inPlane(cut, 0, origin, azimuth, dip))
    expect(area).toBeGreaterThan(0) // counter-clockwise for the viewer
    expect(area / (Math.PI * expected * expected)).toBeGreaterThan(0.998)
    expect(area / (Math.PI * expected * expected)).toBeLessThanOrEqual(1)
    expect(length(cut, 0) / (2 * Math.PI * expected)).toBeGreaterThan(0.999)
  }
  // Beyond the sphere: nothing.
  expect(outlineCount(cutMesh(mesh.vertices, mesh.faces, [centre[0], centre[1] + 51, centre[2]], [0, -1, 0]))).toBe(0)
})

test('two spheres give two loops; an open surface gives an open profile', () => {
  const a = sphere([0, 0, 0], 10, 24, 48)
  const b = sphere([40, 0, 0], 6, 24, 48)
  const shift = a.vertices.length / 3
  const vertices = Float64Array.from([...a.vertices, ...b.vertices])
  const faces = Uint32Array.from([...a.faces, ...Array.from(b.faces, (f) => f + shift)])
  const cut = cutMesh(vertices, faces, [0, 1, 0], [0, -1, 0])
  expect(outlineCount(cut)).toBe(2)
  expect(Array.from(cut.closed)).toEqual([1, 1])

  // A 20 x 20 m sheet of 4 x 4 cells, z = x / 2.
  const sheetVertices: number[] = []
  const sheetFaces: number[] = []
  for (let j = 0; j <= 4; j++) for (let i = 0; i <= 4; i++) sheetVertices.push(i * 5, j * 5, i * 2.5)
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < 4; i++) {
      const k = j * 5 + i
      sheetFaces.push(k, k + 1, k + 6, k, k + 6, k + 5)
    }
  }
  const profile = cutMesh(Float64Array.from(sheetVertices), Uint32Array.from(sheetFaces), [0, 7, 0], [0, -1, 0])
  expect(outlineCount(profile)).toBe(1)
  expect(profile.closed[0]).toBe(0)
  expect(length(profile, 0)).toBeCloseTo(Math.hypot(20, 10), 9)
  const p = outlinePoints(profile, 0)
  for (let i = 0; i < p.length; i += 3) {
    expect(p[i + 1]).toBeCloseTo(7, 12)
    expect(p[i + 2]).toBeCloseTo(p[i] / 2, 12)
  }
})

test('scene-space segments: closed loops are closed, coordinates are origin-relative', () => {
  const mesh = box(CUBE_MIN, CUBE_MAX)
  const cut = cutMesh(mesh.vertices, mesh.faces, [500105, 5300210, 1015], [0, -1, 0])
  const points = outlinePoints(cut, 0).length / 3
  const segments = outlineSegments(cut, [500100, 5300200, 1000])
  expect(segments.length).toBe(points * 6)
  for (const value of segments) expect(Math.abs(value)).toBeLessThanOrEqual(30)
  // The last segment ends where the first one starts.
  expect(Array.from(segments.subarray(segments.length - 3))).toEqual(Array.from(segments.subarray(0, 3)))
})

/** Index shift that maps loop `a` onto loop `b` (same direction), or -1. */
function rotation(a: number[][], b: number[][], tolerance: number): number {
  if (a.length !== b.length) return -1
  const near = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) <= tolerance
  for (let shift = 0; shift < b.length; shift++) {
    if (!near(a[0], b[shift])) continue
    if (a.every((p, i) => near(p, b[(i + shift) % b.length]))) return shift
  }
  return -1
}

test.skipIf(!haveDemo)('matches the Python section() on the synthetic ore solid and topography', () => {
  expect(fixture.cases.length).toBeGreaterThanOrEqual(6)
  const meshes = new Map<string, ReturnType<typeof loadMesh>>()
  for (const item of fixture.cases) {
    if (!meshes.has(item.mesh)) meshes.set(item.mesh, loadMesh(item.mesh))
    const mesh = meshes.get(item.mesh)!
    // The viewer derives the normal from azimuth and dip: it must be the fixture's.
    const { n } = sectionFrame(item.azimuth, item.dip)
    for (let k = 0; k < 3; k++) expect(n[k]).toBeCloseTo(item.normal[k], 14)

    const cut = cutMesh(mesh.vertices, mesh.faces, item.origin, n)
    expect(outlineCount(cut), item.name).toBe(item.polylines.length)
    const mine = Array.from({ length: outlineCount(cut) }, (_, k) => {
      const p = outlinePoints(cut, k)
      return { closed: cut.closed[k] === 1, points: Array.from({ length: p.length / 3 }, (_, i) => [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]]) }
    })
    const used = new Set<number>()
    item.polylines.forEach((expected, e) => {
      // Same points in the same direction; a closed loop may start anywhere.
      const match = mine.findIndex((m, k) => {
        if (used.has(k) || m.closed !== item.closed[e]) return false
        return m.closed ? rotation(expected, m.points, 2e-6) >= 0 : rotation(expected, m.points, 2e-6) === 0
      })
      expect(match, `${item.name}: polyline ${e}`).toBeGreaterThanOrEqual(0)
      used.add(match)
    })
    // Area and length as the Python side reports them.
    let area = 0
    let total = 0
    for (let k = 0; k < outlineCount(cut); k++) {
      if (cut.closed[k]) area += signedArea(inPlane(cut, k, item.origin, item.azimuth, item.dip))
      total += length(cut, k)
    }
    expect(area).toBeCloseTo(item.area, 4)
    expect(total).toBeCloseTo(item.length, 4)
  }
})

test.skipIf(!haveDemo)('a million triangles are cut in well under a frame budget of work per 100k', () => {
  // The ore solid repeated to ~1M triangles (separate copies, shifted east).
  const ore = loadMesh('ore_solid')
  const copies = Math.ceil(1_000_000 / (ore.faces.length / 3))
  const nv = ore.vertices.length / 3
  const vertices = new Float64Array(ore.vertices.length * copies)
  const faces = new Uint32Array(ore.faces.length * copies)
  for (let c = 0; c < copies; c++) {
    for (let i = 0; i < ore.vertices.length; i += 3) {
      vertices[c * ore.vertices.length + i] = ore.vertices[i] + c * 400
      vertices[c * ore.vertices.length + i + 1] = ore.vertices[i + 1]
      vertices[c * ore.vertices.length + i + 2] = ore.vertices[i + 2]
    }
    for (let i = 0; i < ore.faces.length; i++) faces[c * ore.faces.length + i] = ore.faces[i] + c * nv
  }
  const origin = fixture.cases[0].origin
  cutMesh(vertices, faces, origin, [0, -1, 0]) // warm up
  const start = performance.now()
  const cut = cutMesh(vertices, faces, origin, [0, -1, 0])
  const ms = performance.now() - start
  expect(outlineCount(cut)).toBe(copies)
  console.log(`[perf] cutMesh: ${faces.length / 3} triangles, ${copies} loops, ${cut.points.length / 3} points in ${ms.toFixed(1)} ms (node)`)
  expect(ms).toBeLessThan(1000)
})

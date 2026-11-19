import { beforeEach, expect, test } from 'vitest'
import { niceRange } from '../colour/legend'
import { rampColor, srgbToLinear } from '../colour/ramps'
import { buildPerfMesh } from '../scene/perfData'
import { useDisplay } from '../state/display'
import {
  clampOpacity,
  defaultElevationSettings,
  defaultMeshDisplay,
  elevationColors,
  elevationLegend,
  zRangeOf,
} from './display'

beforeEach(() => useDisplay.setState({ meshes: {}, legends: {} }))

test('niceRange widens to round numbers', () => {
  expect(niceRange(1172.39, 1232.45)).toEqual({ min: 1170, max: 1235 })
  expect(niceRange(868.2, 1129.4)).toEqual({ min: 850, max: 1150 })
  expect(niceRange(0.012, 0.047)).toEqual({ min: 0.01, max: 0.05 })
  expect(niceRange(-3.2, 4.1)).toEqual({ min: -3.5, max: 4.5 })
  expect(niceRange(100, 200)).toEqual({ min: 100, max: 200 })
  expect(niceRange(5, 5)).toBeNull()
  expect(niceRange(Number.NaN, 5)).toBeNull()
})

test('defaults: solids opaque, surfaces see-through, distinct colours per kind', () => {
  const solid = defaultMeshDisplay(true, [868, 1129])
  const surface = defaultMeshDisplay(false, [1172.39, 1232.45])
  expect(solid.mode).toBe('shaded')
  expect(surface.mode).toBe('transparent')
  expect(solid.color).not.toBe(surface.color)
  expect(defaultMeshDisplay(true, [0, 1], 1).color).not.toBe(solid.color)
  expect(surface.elevation).toEqual({ mode: 'continuous', steps: 6, min: 1170, max: 1235 })
  expect(surface.colorByElevation).toBe(false)
  // A flat surface still gets a usable range.
  const flat = defaultElevationSettings([100, 100])
  expect(flat.max).toBeGreaterThan(flat.min)
  expect(clampOpacity(7)).toBe(1)
  expect(clampOpacity(0)).toBe(0.05)
  expect(clampOpacity(Number.NaN)).toBe(1)
})

test('zRangeOf reads the third coordinate', () => {
  expect(zRangeOf([1, 2, 30, 4, 5, -6, 7, 8, 9])).toEqual([-6, 30])
  expect(zRangeOf([])).toEqual([0, 0])
})

test('elevation colours follow the legend, by true elevation', () => {
  const display = { ...defaultMeshDisplay(false, [1000, 1100]), colorByElevation: true }
  const spec = elevationLegend(display)
  expect(spec.title).toBe('Elevation')
  expect(spec.unit).toBe('m')
  expect([spec.min, spec.max]).toEqual([1000, 1100])
  // x and y at mine scale must not matter; z below / above the range clamps.
  const vertices = [500000, 5300000, 1000, 500010, 5300000, 1050, 500020, 5300010, 1100, 0, 0, 900, 0, 0, 1500]
  const colors = elevationColors(vertices, spec, 'dark')
  const at = (v: number) => Array.from(colors.slice(v * 3, v * 3 + 3))
  const want = (t: number) => rampColor('elevation', 'dark', t).map(srgbToLinear)
  at(0).forEach((c, k) => expect(c).toBeCloseTo(want(0)[k], 6))
  at(2).forEach((c, k) => expect(c).toBeCloseTo(want(1)[k], 6))
  // Mid value: within one of the 256 quantisation steps of the exact colour.
  at(1).forEach((c, k) => expect(Math.abs(c - want(0.5)[k])).toBeLessThan(0.01))
  expect(at(3)).toEqual(at(0))
  expect(at(4)).toEqual(at(2))
  // Classes: both in the lowest of four classes -> same colour; next class differs.
  const stepped = elevationColors([0, 0, 1001, 0, 0, 1020, 0, 0, 1030], { ...spec, mode: 'stepped', steps: 4 }, 'light')
  expect(Array.from(stepped.slice(0, 3))).toEqual(Array.from(stepped.slice(3, 6)))
  expect(Array.from(stepped.slice(0, 3))).not.toEqual(Array.from(stepped.slice(6, 9)))
  // Reuses the buffer it is given.
  const out = new Float32Array(15)
  expect(elevationColors(vertices, spec, 'dark', out)).toBe(out)
})

test('store: register, edit and reset a mesh display', () => {
  const { registerMesh, setMesh } = useDisplay.getState()
  registerMesh('topo', false, [1172.39, 1232.45])
  registerMesh('ore', true, [868, 1129])
  registerMesh('ore2', true, [0, 10])
  const mesh = (id: string) => useDisplay.getState().meshes[id]
  expect(mesh('ore').color).not.toBe(mesh('ore2').color)

  setMesh('ore', { mode: 'transparent', opacity: 0.2, color: '#00ff00', shading: 'flat' })
  expect(mesh('ore')).toMatchObject({ mode: 'transparent', opacity: 0.2, color: '#00ff00', shading: 'flat' })
  setMesh('ore', { opacity: 9, color: 'red' })
  expect(mesh('ore').opacity).toBe(1)
  expect(mesh('ore').color).toBe('#00ff00') // not a hex colour: refused

  setMesh('topo', { colorByElevation: true, elevation: { ...mesh('topo').elevation, min: 1200, max: 1180, mode: 'stepped' } })
  expect(mesh('topo').elevation).toMatchObject({ min: 1180, max: 1200, mode: 'stepped' })
  setMesh('topo', { elevation: { ...mesh('topo').elevation, min: 5, max: 5 } })
  expect(mesh('topo').elevation).toMatchObject({ min: 1180, max: 1200 }) // empty range: refused
  setMesh('topo', { elevation: undefined })
  expect(mesh('topo').elevation).toMatchObject({ min: 1170, max: 1235, mode: 'stepped' })

  // Registering again (data reloaded) keeps the user's choices.
  registerMesh('ore', true, [868, 1129])
  expect(mesh('ore').mode).toBe('transparent')
  setMesh('nope', { mode: 'shaded' })
  expect(mesh('nope')).toBeUndefined()
})

test('generated performance surface has the requested size and valid indices', () => {
  const small = buildPerfMesh(20_000)
  expect(small.faces.length / 3).toBe(20_000)
  expect(small.vertices.length / 3).toBe(101 * 101)
  expect(Math.max(...small.faces)).toBe(101 * 101 - 1)
  const big = buildPerfMesh(1_000_000)
  expect(big.faces.length / 3).toBeGreaterThan(995_000)
  expect(big.faces.length / 3).toBeLessThan(1_005_000)
  // World coordinates at mine scale, float64.
  expect(big.vertices).toBeInstanceOf(Float64Array)
  expect(big.vertices[0]).toBeGreaterThan(499_000)
  expect(big.vertices[1]).toBeGreaterThan(5_299_000)
})

import { describe, expect, it } from 'vitest'
import { buildSurfaceRequest, type SurfaceForm } from './geometryBuilders'

describe('surface from points request', () => {
  const form: SurfaceForm = {
    name: ' Topo ', databases: ['drillholes'], points: [], strings: [], breaklines: [],
    extendMode: 'none', padding: '', rectangle: '', model: '', extendZ: '', duplicates: 'mean',
  }
  it('collars alone', () => {
    expect(buildSurfaceRequest(form)).toEqual({ name: 'Topo', databases: ['drillholes'], duplicates: 'mean' })
  })
  it('combined sources, breaklines and an extension', () => {
    const body = buildSurfaceRequest({
      ...form,
      points: [{ upload: 'abc', file: 'pts.csv', columns: 'E, N, RL' }, { upload: 'def', file: 'more.xlsx', columns: '' }],
      strings: ['contours'],
      breaklines: [{ upload: 'ghi', file: 'crest.str' }],
      extendMode: 'model', model: 'model', padding: '25', extendZ: '1200', duplicates: 'max',
    })
    expect(body).toEqual({
      name: 'Topo', databases: ['drillholes'], duplicates: 'max',
      points: [{ upload: 'abc', file: 'pts.csv', columns: ['E', 'N', 'RL'] }, { upload: 'def', file: 'more.xlsx' }],
      strings: ['contours'],
      breaklines: [{ upload: 'ghi', file: 'crest.str' }],
      extend: { mode: 'model', model: 'model', padding: 25, z: 1200 },
    })
    expect(buildSurfaceRequest({ ...form, extendMode: 'padding', padding: '100' })).toMatchObject({ extend: { mode: 'padding', padding: 100, z: 'nearest' } })
    expect(buildSurfaceRequest({ ...form, extendMode: 'rectangle', rectangle: '0, 0, 100, 50' })).toMatchObject({ extend: { mode: 'rectangle', rectangle: [0, 0, 100, 50], z: 'nearest' } })
  })
  it('reports what is missing', () => {
    expect(buildSurfaceRequest({ ...form, databases: [] })).toMatch(/source/)
    expect(buildSurfaceRequest({ ...form, name: '' })).toMatch(/name/)
    expect(buildSurfaceRequest({ ...form, extendMode: 'padding', padding: '' })).toMatch(/positive/)
    expect(buildSurfaceRequest({ ...form, extendMode: 'model' })).toMatch(/model/)
    expect(buildSurfaceRequest({ ...form, extendMode: 'rectangle', rectangle: '0, 0, 0, 5' })).toMatch(/Rectangle/)
    expect(buildSurfaceRequest({ ...form, extendMode: 'padding', padding: '10', extendZ: 'high' })).toMatch(/elevation/)
    expect(buildSurfaceRequest({ ...form, points: [{ upload: 'a', file: 'p.csv', columns: 'x, y' }] })).toMatch(/three columns/)
  })
})

import { describe, expect, it } from 'vitest'
import { buildRasterRequest, buildSurfaceRequest, type RasterForm, type SurfaceForm } from './geometryBuilders'

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

describe('elevation files', () => {
  const dem = { rasters: [{ upload: 'tok', file: 'dem.tif' }], rasterCrs: '', crs: '', spacing: '', zShift: '' }
  const points: SurfaceForm = {
    name: 'Topo', databases: ['drillholes'], points: [], strings: [], breaklines: [],
    extendMode: 'padding', padding: '200', rectangle: '', model: '', extendZ: '', duplicates: 'mean',
  }
  it('fills the area around the data', () => {
    expect(buildSurfaceRequest({ ...points, dem: { ...dem, dropWithin: '30', spacing: '25', crs: ' EPSG:32633 ' } })).toEqual({
      name: 'Topo', databases: ['drillholes'], duplicates: 'mean',
      extend: { mode: 'padding', padding: 200, z: 'nearest' },
      raster: { rasters: [{ upload: 'tok', file: 'dem.tif' }], crs: 'EPSG:32633', spacing: 25, z_shift: 'median', drop_within: 30 },
    })
    expect(buildSurfaceRequest({ ...points, dem: { ...dem, zShift: '-1.5' } })).toMatchObject({ raster: { z_shift: -1.5 } })
    expect(buildSurfaceRequest({ ...points, extendZ: '1000', dem })).toMatch(/does not apply/)
    expect(buildSurfaceRequest({ ...points, dem: { ...dem, rasters: [] } })).toMatch(/Upload/)
    expect(buildSurfaceRequest({ ...points, dem: { ...dem, zShift: 'up' } })).toMatch(/shift/)
  })
  const raster: RasterForm = {
    ...dem, name: ' DEM topo ', database: 'drillholes', areaMode: 'collars', model: '', rectangle: '', padding: '100', tolerance: '',
  }
  it('builds the from-raster request', () => {
    expect(buildRasterRequest(raster)).toEqual({
      name: 'DEM topo', rasters: [{ upload: 'tok', file: 'dem.tif' }], z_shift: 'median',
      area: { mode: 'collars', padding: 100 }, database: 'drillholes',
    })
    expect(buildRasterRequest({ ...raster, areaMode: 'model', model: 'bm', tolerance: '3', rasterCrs: 'EPSG:4326' })).toMatchObject({
      area: { mode: 'model', model: 'bm', padding: 100 }, collar_tolerance: 3, raster_crs: 'EPSG:4326',
    })
    expect(buildRasterRequest({ ...raster, database: '', areaMode: 'rectangle', rectangle: '0, 0, 500, 400', zShift: 'none' })).toEqual({
      name: 'DEM topo', rasters: [{ upload: 'tok', file: 'dem.tif' }], z_shift: 'none',
      area: { mode: 'rectangle', rectangle: [0, 0, 500, 400], padding: 100 },
    })
  })
  it('reports what is missing', () => {
    expect(buildRasterRequest({ ...raster, database: '' })).toMatch(/database/)
    expect(buildRasterRequest({ ...raster, database: '', areaMode: 'rectangle', rectangle: '0, 0, 5, 5' })).toMatch(/median/)
    expect(buildRasterRequest({ ...raster, areaMode: 'model' })).toMatch(/block model/)
    expect(buildRasterRequest({ ...raster, spacing: '-2' })).toMatch(/spacing/)
    expect(buildRasterRequest({ ...raster, name: '' })).toMatch(/name/)
  })
})

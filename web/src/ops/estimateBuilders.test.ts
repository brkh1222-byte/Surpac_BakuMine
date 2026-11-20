import { describe, expect, it } from 'vitest'
import { buildEstimateSpec, type EstimateForm } from './builders'

describe('estimation request with nested structures', () => {
  const form: EstimateForm = {
    attribute: 'au', method: 'ok', power: '2', radii: '150, 100, 40', angles: '0, 0, 0', nugget: '0.1',
    orientation: 'plane', dip: '69', dipAzimuth: '216', pitch: '',
    structures: [
      { type: 'spherical', sill: '0.4', ranges: '30, 20, 8' },
      { type: 'exponential', sill: '0.3', ranges: '90 60 15', orientation: 'angles', angles: '20, -5, 30' },
      { type: 'gaussian', sill: '0.2', ranges: '200, 120, 40', orientation: 'plane', dip: '60', dipAzimuth: '200', pitch: '35' },
    ],
    domainName: '', blockWhere: '', sampleColumn: '', sampleValues: '',
    passes: [
      { name: 'near', radiusFactor: '0.5', minSamples: '4', maxSamples: '12', maxPerHole: '3' },
      { name: 'far', radiusFactor: '1', minSamples: '1', maxSamples: '12', maxPerHole: '' },
    ],
  }
  it('each structure has its own ranges and orientation (the variogram of tests/api/test_estimate_structures.py)', () => {
    expect(buildEstimateSpec(form)).toEqual({
      attribute: 'au', method: 'ok',
      ellipsoid: { radii: [150, 100, 40], plane: { dip: 69, dip_azimuth: 216, pitch: 0 } },
      variogram: {
        nugget: 0.1,
        structures: [
          { type: 'spherical', sill: 0.4, ranges: [30, 20, 8] },
          { type: 'exponential', sill: 0.3, ranges: [90, 60, 15], angles: [20, -5, 30] },
          { type: 'gaussian', sill: 0.2, ranges: [200, 120, 40], plane: { dip: 60, dip_azimuth: 200, pitch: 35 } },
        ],
      },
      domain: { name: 'all' },
      passes: [
        { name: 'near', radius_factor: 0.5, min_samples: 4, max_samples: 12, max_per_hole: 3 },
        { name: 'far', radius_factor: 1, min_samples: 1, max_samples: 12 },
      ],
    })
  })
  it('a structure without ranges or orientation follows the search ellipsoid (the old request)', () => {
    const spec = buildEstimateSpec({ ...form, orientation: 'angles', angles: '35, -10, 50', structures: [{ type: 'spherical', sill: '0.85' }, { type: 'spherical', sill: '' }] })
    expect(spec).toMatchObject({ ellipsoid: { radii: [150, 100, 40], angles: [35, -10, 50] }, variogram: { nugget: 0.1 } })
    expect((spec as { variogram: { structures: object[] } }).variogram.structures).toEqual([{ type: 'spherical', sill: 0.85 }])
    // the same with the fields the dialog now keeps for every structure, left at their defaults
    const blank = { type: 'spherical', sill: '0.85', ranges: '', orientation: 'search' as const, angles: '', dip: '', dipAzimuth: '', pitch: '' }
    expect((buildEstimateSpec({ ...form, structures: [blank] }) as { variogram: { structures: object[] } }).variogram.structures).toEqual([{ type: 'spherical', sill: 0.85 }])
  })
  it('reports invalid nested fields', () => {
    const one = form.structures[0]
    expect(buildEstimateSpec({ ...form, structures: [{ ...one, ranges: '30, 20' }] })).toMatch(/Structure 1: ranges/)
    expect(buildEstimateSpec({ ...form, structures: [{ ...one, ranges: '30, 20, 0' }] })).toMatch(/Structure 1: ranges/)
    expect(buildEstimateSpec({ ...form, structures: [one, { ...one, orientation: 'angles', angles: '1' }] })).toMatch(/Structure 2: three numbers/)
    expect(buildEstimateSpec({ ...form, structures: [{ ...one, orientation: 'plane', dip: '', dipAzimuth: '10' }] })).toMatch(/Structure 1: Give the dip/)
    expect(buildEstimateSpec({ ...form, structures: [one, one, one, one] })).toMatch(/At most 3/)
    expect(buildEstimateSpec({ ...form, dip: '' })).toMatch(/Search ellipsoid: Give the dip/)
    expect(buildEstimateSpec({ ...form, orientation: 'angles', angles: '1, 2' })).toMatch(/Angles: three numbers/)
  })
})

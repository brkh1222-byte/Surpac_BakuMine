import { describe, expect, it } from 'vitest'
import { buildImplicitRequest, describeAnisotropy, parseRatios, type ImplicitForm } from './implicitSolid'

describe('implicit solid request', () => {
  const form: ImplicitForm = {
    database: 'drillholes', table: 'geology', column: 'rock', codes: 'ORE', mode: '',
    dip: '', dipAzimuth: '', pitch: '', angles: '', ratios: '', maxExtrapolation: '', topography: '',
    resolution: '', minVolume: '', name: '',
  }
  it('asks for the orientation by default', () => {
    expect(buildImplicitRequest(form)).toMatch(/orientation/)
    expect(buildImplicitRequest({ ...form, mode: 'plane' })).toMatch(/Ratios/)
    expect(buildImplicitRequest({ ...form, mode: 'plane', ratios: '4:2:1' })).toMatch(/dip/)
    expect(buildImplicitRequest({ ...form, mode: 'angles', ratios: '4:2:1', angles: '1, 2' })).toMatch(/Angles/)
  })
  it('a plane with ratios, clip and limits', () => {
    expect(
      buildImplicitRequest({
        ...form, codes: 'ORE, OXIDE', mode: 'plane', dip: '69', dipAzimuth: '216', ratios: '4:2:1',
        maxExtrapolation: '60', topography: 'topo', resolution: '2.5', minVolume: '0', name: ' Lode ',
      }),
    ).toEqual({
      database: 'drillholes', table: 'geology', column: 'rock', codes: ['ORE', 'OXIDE'],
      anisotropy: { mode: 'plane', dip: 69, dip_azimuth: 216, pitch: 0, ratios: [4, 2, 1] },
      max_extrapolation: 60, resolution: 2.5, min_component_volume: 0, topography: 'topo', name: 'Lode',
    })
  })
  it('angles, automatic and none', () => {
    expect(buildImplicitRequest({ ...form, mode: 'angles', angles: '126, 0, -69', ratios: '3 3 1' })).toMatchObject({ anisotropy: { mode: 'angles', angles: [126, 0, -69], ratios: [3, 3, 1] } })
    expect(buildImplicitRequest({ ...form, mode: 'auto' })).toEqual({ database: 'drillholes', table: 'geology', column: 'rock', codes: ['ORE'], anisotropy: { mode: 'auto' } })
    expect(buildImplicitRequest({ ...form, mode: 'none' })).toMatchObject({ anisotropy: { mode: 'none' } })
    expect(buildImplicitRequest({ ...form, mode: 'auto', codes: ' ' })).toMatch(/code/)
    expect(buildImplicitRequest({ ...form, mode: 'auto', maxExtrapolation: '-5' })).toMatch(/extrapolation/)
    expect(buildImplicitRequest({ ...form, mode: 'auto', resolution: '0' })).toMatch(/Resolution/)
    expect(buildImplicitRequest({ ...form, mode: 'auto', database: '' })).toMatch(/database/)
  })
  it('parses ratios', () => {
    expect(parseRatios('4:2:1')).toEqual([4, 2, 1])
    expect(parseRatios('4, 2, 1')).toEqual([4, 2, 1])
    expect(parseRatios('4:2')).toBeNull()
    expect(parseRatios('4:0:1')).toBeNull()
  })
  it('states the anisotropy that was used', () => {
    expect(
      describeAnisotropy({
        source: 'auto', method: 'intercept midpoints', ratios: [1, 0.5531, 0.0695],
        major: { azimuth: 28, plunge: 17.5 }, semi_major: { azimuth: 275.7, plunge: 50.3 }, minor: { azimuth: 130.4, plunge: 34.3 },
        plane: { dip: 55.7, dip_azimuth: 310.4 },
      }),
    ).toBe('Anisotropy (auto): plane dips 56 towards 310; major 028/18, semi-major 276/50, minor 130/34; extent 1 : 0.55 : 0.07.')
    expect(describeAnisotropy({ source: 'none', method: 'given' })).toBe('Anisotropy: none (isotropic).')
    expect(describeAnisotropy({ source: 'none (auto failed)', method: 'none', confidence: 0.021, candidate: { dip: 28, dip_azimuth: 212 } }))
      .toBe('Anisotropy: none (auto failed) (isotropic). Confidence 0.02.')
  })
})

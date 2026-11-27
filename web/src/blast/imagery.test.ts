import { describe, expect, it } from 'vitest'
import { mosaicSize, uvArray, uvAt, type ImageryPlan } from './imagery'
import { steepShade } from './scene/Ground'

/** A 3 x 3 grid over x 1000..1200, y 5000..5400 where u = (x-1000)/400 and v = (5400-y)/800 exactly. */
function plan(): ImageryPlan {
  const u: number[] = []
  const v: number[] = []
  for (let ix = 0; ix < 3; ix++) {
    for (let iy = 0; iy < 3; iy++) {
      u.push((ix * 100) / 400)
      v.push((400 - iy * 200) / 800)
    }
  }
  return {
    source: 'test', attribution: 'test', crs: 'EPSG:32633', rectangle: [1000, 5000, 1200, 5400], zoom: 15, tile_size: 256,
    tiles: { x0: 10, y0: 20, nx: 2, ny: 3 }, metres_per_pixel: 3,
    grid: { origin: [1000, 5000], step: [100, 200], shape: [3, 3], u, v },
  }
}

describe('satellite texture coordinates', () => {
  it('interpolates the grid and flips v for three.js', () => {
    const p = plan()
    // South-west corner: u = 0, v(south) = 0.5 -> flipped 0.5.
    expect(uvAt(p, 1000, 5000)).toEqual([0, 0.5])
    // North-east corner: u = 0.5, v(south) = 0 -> flipped 1.
    expect(uvAt(p, 1200, 5400)).toEqual([0.5, 1])
    // Inside a cell: linear in both directions.
    const [u, v] = uvAt(p, 1050, 5300)
    expect(u).toBeCloseTo(50 / 400, 12)
    expect(v).toBeCloseTo(1 - 100 / 800, 12)
  })

  it('extrapolates linearly outside the rectangle', () => {
    const [u, v] = uvAt(plan(), 1300, 4900)
    expect(u).toBeCloseTo(300 / 400, 12)
    expect(v).toBeCloseTo(1 - 500 / 800, 12)
  })

  it('fills a uv attribute from scene positions and the scene origin', () => {
    const uv = uvArray(plan(), [0, 0, 7, 100, 200, 9], [1000, 5000, 300])
    expect(Array.from(uv)).toEqual([0, 0.5, 0.25, 0.75])
  })

  it('scales a large tile block down to the canvas limit', () => {
    const p = plan()
    expect(mosaicSize(p)).toEqual({ width: 512, height: 768, scale: 1 })
    p.tiles = { x0: 0, y0: 0, nx: 20, ny: 10 }
    const m = mosaicSize(p)
    expect(m.width).toBe(4096)
    expect(m.height).toBe(2048)
  })
})

describe('steep faces under the satellite picture', () => {
  it('keeps flat ground bright and darkens cut faces', () => {
    expect(steepShade(0)).toBe(1)
    expect(steepShade(Math.tan((35 * Math.PI) / 180))).toBe(1)
    expect(steepShade(Math.tan((75 * Math.PI) / 180))).toBeCloseTo(0.45, 12)
    const mid = steepShade(Math.tan((48 * Math.PI) / 180))
    expect(mid).toBeLessThan(1)
    expect(mid).toBeGreaterThan(0.45)
  })
})

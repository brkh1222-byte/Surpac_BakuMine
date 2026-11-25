/**
 * The project's topography for the blast scene: one fetch shared by the layers that need
 * it (the terrain itself, the equipment and the ground scatter standing on it), and a
 * height lookup on its vertices.
 */
import { useEffect, useState } from 'react'
import { fetchManifest, fetchMesh, type MeshData } from '../../api/client'
import type { BlastBundle } from '../types'
import { blastView } from './common'
import { groundRelief, reliefPatches, reliefWeight, type ReliefHole, type ReliefPatch } from './realism'

/** Height of a surface at a plan point, in the units of its vertices; NaN outside it. */
export type HeightAt = (x: number, y: number) => number

/**
 * Height lookup on the vertices `position` (x, y, z triples) of a surface.
 *
 * When the vertices form a complete regular grid in plan (a surface made from a raster),
 * the height is the bilinear interpolation in the grid cell, which reproduces a plane
 * exactly. Otherwise it is the inverse-distance-squared mean of the nearest vertices found
 * in the 3 x 3 neighbourhood of a hash cell (exact at a vertex), NaN when none is there.
 */
export function heightSampler(position: ArrayLike<number>): HeightAt {
  const n = Math.floor(position.length / 3)
  if (n === 0) return () => Number.NaN
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (let i = 0; i < n; i++) {
    const x = position[i * 3]
    const y = position[i * 3 + 1]
    if (x < x0) x0 = x
    if (x > x1) x1 = x
    if (y < y0) y0 = y
    if (y > y1) y1 = y
  }
  const regular = regularGrid(position, n, x0, y0, x1, y1)
  if (regular) return regular

  const cell = Math.max(1e-9, Math.sqrt(((x1 - x0) * (y1 - y0)) / n) || Math.max(x1 - x0, y1 - y0, 1))
  const nx = Math.max(1, Math.floor((x1 - x0) / cell) + 1)
  const ny = Math.max(1, Math.floor((y1 - y0) / cell) + 1)
  const start = new Int32Array(nx * ny + 1)
  const key = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const k = Math.min(nx - 1, Math.floor((position[i * 3] - x0) / cell)) * ny + Math.min(ny - 1, Math.floor((position[i * 3 + 1] - y0) / cell))
    key[i] = k
    start[k + 1]++
  }
  for (let k = 0; k < nx * ny; k++) start[k + 1] += start[k]
  const fill = start.slice(0, nx * ny)
  const order = new Int32Array(n)
  for (let i = 0; i < n; i++) order[fill[key[i]]++] = i
  return (x, y) => {
    if (!(x >= x0 - cell && x <= x1 + cell && y >= y0 - cell && y <= y1 + cell)) return Number.NaN
    const cx = Math.min(nx - 1, Math.max(0, Math.floor((x - x0) / cell)))
    const cy = Math.min(ny - 1, Math.max(0, Math.floor((y - y0) / cell)))
    let sum = 0
    let weight = 0
    for (let ix = Math.max(0, cx - 1); ix <= Math.min(nx - 1, cx + 1); ix++) {
      for (let iy = Math.max(0, cy - 1); iy <= Math.min(ny - 1, cy + 1); iy++) {
        for (let s = start[ix * ny + iy]; s < start[ix * ny + iy + 1]; s++) {
          const i = order[s]
          const d2 = (position[i * 3] - x) ** 2 + (position[i * 3 + 1] - y) ** 2
          if (d2 < 1e-12) return position[i * 3 + 2]
          sum += position[i * 3 + 2] / d2
          weight += 1 / d2
        }
      }
    }
    return weight > 0 ? sum / weight : Number.NaN
  }
}

/** Bilinear lookup when the vertices are a complete grid `origin + (i dx, j dy)`; else null. */
function regularGrid(position: ArrayLike<number>, n: number, x0: number, y0: number, x1: number, y1: number): HeightAt | null {
  let dx = Infinity
  let dy = Infinity
  for (let i = 0; i < n; i++) {
    const ax = position[i * 3] - x0
    const ay = position[i * 3 + 1] - y0
    if (ax > 1e-6 && ax < dx) dx = ax
    if (ay > 1e-6 && ay < dy) dy = ay
  }
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) return null
  const nx = Math.round((x1 - x0) / dx) + 1
  const ny = Math.round((y1 - y0) / dy) + 1
  if (nx < 2 || ny < 2 || nx * ny !== n) return null
  const z = new Float64Array(n).fill(Number.NaN)
  const tolerance = 1e-3 * Math.min(dx, dy)
  for (let i = 0; i < n; i++) {
    const fx = (position[i * 3] - x0) / dx
    const fy = (position[i * 3 + 1] - y0) / dy
    const ix = Math.round(fx)
    const iy = Math.round(fy)
    if (Math.abs(fx - ix) * dx > tolerance || Math.abs(fy - iy) * dy > tolerance) return null
    if (!Number.isNaN(z[ix * ny + iy])) return null
    z[ix * ny + iy] = position[i * 3 + 2]
  }
  return (x, y) => {
    const fx = (x - x0) / dx
    const fy = (y - y0) / dy
    if (!(fx >= 0 && fx <= nx - 1 && fy >= 0 && fy <= ny - 1)) return Number.NaN
    const ix = Math.min(nx - 2, Math.floor(fx))
    const iy = Math.min(ny - 2, Math.floor(fy))
    const u = fx - ix
    const v = fy - iy
    return (
      z[ix * ny + iy] * (1 - u) * (1 - v) + z[(ix + 1) * ny + iy] * u * (1 - v) + z[ix * ny + iy + 1] * (1 - u) * v + z[(ix + 1) * ny + iy + 1] * u * v
    )
  }
}

export interface Topography {
  mesh: MeshData
  /** World height of the surveyed surface at a world plan point; NaN outside it. */
  surveyed: HeightAt
  /** The same with the fine relief of the site view added (realism.groundRelief): what things stand on. */
  heightAt: HeightAt
  patches: ReliefPatch[]
  hole: ReliefHole
}

/**
 * The refined ground is lifted by this much, m, where the relief is fully on: the surveyed
 * mesh is two flat triangles per cell and can rise above the bilinear height by a quarter of
 * the cell's twist, and must stay hidden under the refined ground.
 */
export const RELIEF_LIFT_M = 0.15

let cached: { key: string; promise: Promise<Topography | null> } | null = null

async function load(x: number, y: number, bundle: BlastBundle): Promise<Topography | null> {
  try {
    const manifest = await fetchManifest()
    const surfaces = manifest.objects.filter((o) => o.kind === 'mesh' && o.category === 'surfaces' && o.bounds)
    const inside = surfaces.filter(
      (o) => o.bounds !== null && o.bounds !== undefined && x >= o.bounds.min[0] && x <= o.bounds.max[0] && y >= o.bounds.min[1] && y <= o.bounds.max[1],
    )
    const pick = inside.find((o) => /topo|dtm|dem|surface/i.test(o.name)) ?? inside[0]
    if (!pick) return null
    const mesh = await fetchMesh(pick)
    const surveyed = heightSampler(mesh.vertices)
    const { patches, hole } = reliefPatches(bundle)
    const heightAt: HeightAt = (px, py) => surveyed(px, py) + (RELIEF_LIFT_M + groundRelief(px, py)) * reliefWeight(px, py, patches, hole)
    return { mesh, surveyed, heightAt, patches, hole }
  } catch {
    return null // no project or no surface: the blast draws without it
  }
}

/**
 * The first project surface whose bounds hold the blast (null while loading or when there
 * is none). Every caller with the same bundle shares one fetch. `on` false gives null.
 */
export function useTopography(bundle: BlastBundle, origin: readonly number[], on = true): Topography | null {
  const [topography, setTopography] = useState<Topography | null>(null)
  useEffect(() => {
    if (!on) {
      setTopography(null)
      return
    }
    let alive = true
    const key = `${origin[0]},${origin[1]},${bundle.timeline.t_end_ms},${bundle.timeline.holes.length}`
    if (!cached || cached.key !== key) cached = { key, promise: load(origin[0], origin[1], bundle) }
    void cached.promise.then((t) => {
      if (!alive) return
      setTopography(t)
      blastView.groundAt = t ? t.heightAt : null
    })
    return () => {
      alive = false
    }
  }, [bundle, origin, on])
  return topography
}

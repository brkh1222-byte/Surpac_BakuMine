/**
 * Satellite imagery draped on the blast terrain.
 *
 *   GET  /api/v1/blast/imagery                   { available, plan, crs, project }
 *   POST /api/v1/blast/imagery                   { crs } -> downloads the tiles (explicit action)
 *   GET  /api/v1/blast/imagery/tile/{z}/{y}/{x}  one downloaded tile
 *
 * The plan carries, on a regular grid of model points, the texture coordinates (u east,
 * v south, 0..1) in the block of tiles; `uvAt` interpolates them for any vertex.
 */
import { useEffect, useState } from 'react'
import { CanvasTexture, ClampToEdgeWrapping, LinearFilter, LinearMipmapLinearFilter, SRGBColorSpace, type Texture } from 'three'
import { API_BASE } from '../api/client'
import { detailText } from './api'

export interface ImageryPlan {
  source: string
  attribution: string
  crs: string
  rectangle: [number, number, number, number]
  zoom: number
  tile_size: number
  tiles: { x0: number; y0: number; nx: number; ny: number }
  metres_per_pixel: number
  grid: { origin: [number, number]; step: [number, number]; shape: [number, number]; u: number[]; v: number[] }
  fetched?: { tiles: number; downloaded: number; cached: number; missing: number }
}

export interface ImageryState {
  available: boolean
  plan: ImageryPlan | null
  /** The project's coordinate system when it states one (prefills the form). */
  crs: string | null
  /** False on the demo bench: no real coordinates, no imagery. */
  project: boolean
}

type FetchFn = typeof fetch

async function answer(response: Response): Promise<ImageryState> {
  if (!response.ok) {
    let detail: string | null = null
    try {
      detail = detailText(((await response.json()) as { detail?: unknown } | null)?.detail)
    } catch {
      // not JSON
    }
    throw new Error(detail ?? `${response.status} ${response.statusText}`)
  }
  return (await response.json()) as ImageryState
}

export async function loadImagery(signal?: AbortSignal, fetchFn: FetchFn = fetch): Promise<ImageryState> {
  return answer(await fetchFn(`${API_BASE}/blast/imagery`, { signal }))
}

/** Downloads the tiles over the project topography. Sends the site's tile numbers to the imagery provider. */
export async function fetchImagery(crs: string, fetchFn: FetchFn = fetch): Promise<ImageryState> {
  return answer(
    await fetchFn(`${API_BASE}/blast/imagery`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(crs.trim() ? { crs: crs.trim() } : {}),
    }),
  )
}

/**
 * Texture coordinates of the model point (x, y): [u, v] with v already flipped for a
 * three.js texture (v = 1 at the north edge of the tile block). Points outside the
 * plan's rectangle are extrapolated from the nearest grid cell.
 */
export function uvAt(plan: ImageryPlan, x: number, y: number): [number, number] {
  const { origin, step, shape, u, v } = plan.grid
  const [nx, ny] = shape
  const fx = (x - origin[0]) / step[0]
  const fy = (y - origin[1]) / step[1]
  const ix = Math.min(nx - 2, Math.max(0, Math.floor(fx)))
  const iy = Math.min(ny - 2, Math.max(0, Math.floor(fy)))
  const tx = fx - ix
  const ty = fy - iy
  const a = ix * ny + iy
  const b = (ix + 1) * ny + iy
  const c = ix * ny + iy + 1
  const d = (ix + 1) * ny + iy + 1
  const lerp = (g: number[]) => g[a] * (1 - tx) * (1 - ty) + g[b] * tx * (1 - ty) + g[c] * (1 - tx) * ty + g[d] * tx * ty
  return [lerp(u), 1 - lerp(v)]
}

/** `uv` attribute data for scene positions (world minus `origin`). */
export function uvArray(plan: ImageryPlan, position: ArrayLike<number>, origin: readonly number[]): Float32Array {
  const n = Math.floor(position.length / 3)
  const out = new Float32Array(n * 2)
  for (let i = 0; i < n; i++) {
    const [u, v] = uvAt(plan, position[i * 3] + origin[0], position[i * 3 + 1] + origin[1])
    out[i * 2] = u
    out[i * 2 + 1] = v
  }
  return out
}

/** The largest canvas side used for the mosaic, px. */
export const MAX_MOSAIC = 4096

/** Canvas size of the tile block, scaled down to `MAX_MOSAIC` when larger. */
export function mosaicSize(plan: ImageryPlan): { width: number; height: number; scale: number } {
  const w = plan.tiles.nx * plan.tile_size
  const h = plan.tiles.ny * plan.tile_size
  const scale = Math.min(1, MAX_MOSAIC / Math.max(w, h))
  return { width: Math.round(w * scale), height: Math.round(h * scale), scale }
}

async function buildTexture(plan: ImageryPlan, signal: AbortSignal): Promise<Texture | null> {
  const { width, height, scale } = mosaicSize(plan)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) return null
  context.fillStyle = '#3a3f3c'
  context.fillRect(0, 0, width, height)
  const size = plan.tile_size * scale
  const { x0, y0, nx, ny } = plan.tiles
  const jobs: Promise<void>[] = []
  let drawn = 0
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      jobs.push(
        (async () => {
          try {
            const response = await fetch(`${API_BASE}/blast/imagery/tile/${plan.zoom}/${y0 + j}/${x0 + i}`, { signal })
            if (!response.ok) return
            const bitmap = await createImageBitmap(await response.blob())
            // One pixel of overlap hides seams when the mosaic is scaled.
            context.drawImage(bitmap, i * size, j * size, size + (scale < 1 ? 1 : 0), size + (scale < 1 ? 1 : 0))
            bitmap.close()
            drawn++
          } catch {
            // A tile that cannot be read stays grey.
          }
        })(),
      )
    }
  }
  await Promise.all(jobs)
  if (signal.aborted || drawn === 0) return null
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  texture.wrapS = ClampToEdgeWrapping
  texture.wrapT = ClampToEdgeWrapping
  texture.minFilter = LinearMipmapLinearFilter
  texture.magFilter = LinearFilter
  texture.anisotropy = 8
  texture.needsUpdate = true
  return texture
}

/** The mosaic texture of a plan (null while loading, when off, or when no tile could be read). */
export function useSatelliteTexture(plan: ImageryPlan | null, on: boolean): Texture | null {
  const [texture, setTexture] = useState<Texture | null>(null)
  useEffect(() => {
    if (!plan || !on) {
      setTexture(null)
      return
    }
    const controller = new AbortController()
    let made: Texture | null = null
    void buildTexture(plan, controller.signal).then((t) => {
      if (controller.signal.aborted) {
        t?.dispose()
        return
      }
      made = t
      setTexture(t)
    })
    return () => {
      controller.abort()
      made?.dispose()
    }
  }, [plan, on])
  return texture
}

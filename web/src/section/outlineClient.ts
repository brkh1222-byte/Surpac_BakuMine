/**
 * Mesh outlines off the main thread.
 *
 * Why client-side (and not an endpoint around the Python `section()`): the mesh buffers
 * are already in the browser, so a cut is one pass over typed arrays (~10 ms per million
 * triangles) with no network round trip and no JSON; stepping stays live, and it works
 * for the client-generated scenes (`?mock=perf`) that have no API behind them. The
 * algorithm is a port of the Python one and is tested against its output.
 *
 * Protocol: the mesh is copied to the worker once, at the first cut (never if no section
 * is used). Each cut is one small message there and one transferred reply back. Per
 * mesh, one cut is in flight at a time and only the newest waiting request is kept
 * (stepping fast skips the planes in between). Results are cached per plane, so
 * stepping back is free.
 */
import type { Vec3 } from '../scene/origin'
import { cutMesh, type Outline } from './outline'
import type { OutlineReply, OutlineRequest } from './outline.worker'

const CACHE_SIZE = 96

interface Waiting {
  key: string
  origin: Vec3
  normal: Vec3
  resolve: (outline: Outline | null) => void
}

interface MeshEntry {
  vertices: Float64Array
  faces: Uint32Array
  sent: boolean
  cache: Map<string, Outline>
  inFlight: Waiting | null
  queued: Waiting | null
  started: number
}

export interface OutlineStats {
  /** Cuts computed (cache misses). */
  cuts: number
  cacheHits: number
  /** Worker time of the last cut and the largest so far, ms. */
  lastCutMs: number
  maxCutMs: number
  /** Request-to-reply time of the last cut (includes the worker queue), ms. */
  lastRoundTripMs: number
  maxRoundTripMs: number
  /** Longest main-thread time spent handing a mesh to the worker (once per mesh), ms. */
  maxSendMs: number
  worker: boolean
}

function planeKey(origin: Vec3, normal: Vec3): string {
  const length = Math.hypot(normal[0], normal[1], normal[2]) || 1
  const n = normal.map((c) => c / length)
  const offset = origin[0] * n[0] + origin[1] * n[1] + origin[2] * n[2]
  // 1e-6 m and 1e-9 of direction: far below anything the UI can set apart.
  return `${n.map((c) => c.toFixed(9)).join(',')}|${offset.toFixed(6)}`
}

class OutlineClient {
  private worker: Worker | null = null
  private failed = false
  private meshes = new Map<string, MeshEntry>()
  private requests = new Map<number, string>()
  private nextRequest = 1
  readonly stats: OutlineStats = {
    cuts: 0,
    cacheHits: 0,
    lastCutMs: 0,
    maxCutMs: 0,
    lastRoundTripMs: 0,
    maxRoundTripMs: 0,
    maxSendMs: 0,
    worker: false,
  }

  private ensureWorker(): Worker | null {
    if (this.worker || this.failed) return this.worker
    if (typeof Worker === 'undefined') {
      this.failed = true
      return null
    }
    try {
      this.worker = new Worker(new URL('./outline.worker.ts', import.meta.url), { type: 'module' })
      this.worker.onmessage = (event: MessageEvent<OutlineReply>) => this.receive(event.data)
      this.worker.onerror = () => this.fallBack()
      this.stats.worker = true
    } catch {
      this.failed = true
    }
    return this.worker
  }

  /** The worker died: finish what was waiting on the main thread and stay there. */
  private fallBack(): void {
    this.worker?.terminate()
    this.worker = null
    this.failed = true
    this.stats.worker = false
    this.requests.clear()
    for (const [id, entry] of this.meshes) {
      const waiting = [entry.inFlight, entry.queued]
      entry.inFlight = entry.queued = null
      for (const w of waiting) if (w) w.resolve(this.store(id, w.key, cutMesh(entry.vertices, entry.faces, w.origin, w.normal)))
    }
  }

  /** Declares (or replaces) a mesh. Nothing is copied until the first cut. */
  register(id: string, vertices: Float64Array, faces: Uint32Array): void {
    const previous = this.meshes.get(id)
    if (previous && previous.vertices === vertices && previous.faces === faces) return
    this.unregister(id)
    this.meshes.set(id, { vertices, faces, sent: false, cache: new Map(), inFlight: null, queued: null, started: 0 })
  }

  unregister(id: string): void {
    const entry = this.meshes.get(id)
    if (!entry) return
    entry.inFlight?.resolve(null)
    entry.queued?.resolve(null)
    if (entry.sent) this.worker?.postMessage({ type: 'drop', id } satisfies OutlineRequest)
    this.meshes.delete(id)
  }

  private store(id: string, key: string, outline: Outline): Outline {
    const cache = this.meshes.get(id)?.cache
    if (cache) {
      cache.set(key, outline)
      if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value as string)
    }
    return outline
  }

  private send(id: string, entry: MeshEntry, waiting: Waiting, worker: Worker): void {
    if (!entry.sent) {
      const start = performance.now()
      worker.postMessage({ type: 'mesh', id, vertices: entry.vertices, faces: entry.faces } satisfies OutlineRequest)
      this.stats.maxSendMs = Math.max(this.stats.maxSendMs, performance.now() - start)
      entry.sent = true
    }
    const request = this.nextRequest++
    this.requests.set(request, id)
    entry.inFlight = waiting
    entry.started = performance.now()
    worker.postMessage({
      type: 'cut',
      id,
      request,
      origin: [waiting.origin[0], waiting.origin[1], waiting.origin[2]],
      normal: [waiting.normal[0], waiting.normal[1], waiting.normal[2]],
    } satisfies OutlineRequest)
  }

  private receive(reply: OutlineReply): void {
    const id = this.requests.get(reply.request)
    this.requests.delete(reply.request)
    const entry = id === undefined ? undefined : this.meshes.get(id)
    if (id === undefined || !entry || !entry.inFlight) return
    const done = entry.inFlight
    entry.inFlight = null
    const roundTrip = performance.now() - entry.started
    this.stats.cuts++
    this.stats.lastCutMs = reply.ms
    this.stats.maxCutMs = Math.max(this.stats.maxCutMs, reply.ms)
    this.stats.lastRoundTripMs = roundTrip
    this.stats.maxRoundTripMs = Math.max(this.stats.maxRoundTripMs, roundTrip)
    done.resolve(this.store(id, done.key, { points: reply.points, offsets: reply.offsets, closed: reply.closed }))
    const next = entry.queued
    entry.queued = null
    if (next && this.worker) this.send(id, entry, next, this.worker)
  }

  /**
   * The outline of mesh `id` on the plane through `origin` with normal `normal` (world).
   * Resolves with null when a newer request for the same mesh replaced this one or the
   * mesh went away.
   */
  cut(id: string, origin: Vec3, normal: Vec3): Promise<Outline | null> {
    const entry = this.meshes.get(id)
    if (!entry) return Promise.resolve(null)
    const key = planeKey(origin, normal)
    const cached = entry.cache.get(key)
    if (cached) {
      this.stats.cacheHits++
      return Promise.resolve(cached)
    }
    const worker = this.ensureWorker()
    if (!worker) {
      const start = performance.now()
      const outline = this.store(id, key, cutMesh(entry.vertices, entry.faces, origin, normal))
      this.stats.cuts++
      this.stats.lastCutMs = performance.now() - start
      this.stats.maxCutMs = Math.max(this.stats.maxCutMs, this.stats.lastCutMs)
      return Promise.resolve(outline)
    }
    return new Promise((resolve) => {
      const waiting: Waiting = { key, origin, normal, resolve }
      if (entry.inFlight) {
        entry.queued?.resolve(null)
        entry.queued = waiting
      } else {
        this.send(id, entry, waiting, worker)
      }
    })
  }

  /** True while any cut is being computed or waiting. */
  busy(): boolean {
    for (const entry of this.meshes.values()) if (entry.inFlight || entry.queued) return true
    return false
  }
}

export const outlineClient = new OutlineClient()

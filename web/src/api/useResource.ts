/**
 * The data-loading pattern of the viewer.
 *
 *   const mesh = useResource(`mesh:${object.id}`, object.name, (signal) => fetchMesh(object, signal))
 *   if (mesh.status !== 'ready') return null     // the sidebar and status bar show why
 *   ... draw mesh.data ...
 *
 * `useResource` runs the loader once per key, keeps the result in a module cache (so a
 * remount, or switching back to an attribute table, costs nothing) and mirrors the
 * progress into the store (`loads[key]`), where the sidebar and the status bar read
 * it. Failed loads are retried by `retryFailed()`.
 */
import { useEffect, useState, useSyncExternalStore } from 'react'
import { useViewer } from '../state/store'

export type Resource<T> =
  | { status: 'idle'; data: null; error: null }
  | { status: 'loading'; data: null; error: null }
  | { status: 'ready'; data: T; error: null }
  | { status: 'error'; data: null; error: string }

interface Entry {
  promise: Promise<unknown>
  settled: Resource<unknown> | null
}

const cache = new Map<string, Entry>()
let generation = 0
const listeners = new Set<() => void>()

const IDLE = { status: 'idle', data: null, error: null } as const
const LOADING = { status: 'loading', data: null, error: null } as const

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

function bump(): void {
  generation++
  for (const listener of listeners) listener()
}

/** Forgets every failed load; mounted hooks then load again. */
export function retryFailed(): void {
  for (const [key, entry] of cache) {
    if (entry.settled?.status === 'error') cache.delete(key)
  }
  bump()
}

/** Forgets everything (project closed / tests). */
export function clearResources(): void {
  cache.clear()
  bump()
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Loads the resource `key` (null = nothing to load).
 * `label` names it in messages; `size` reports the payload size for the status bar.
 */
export function useResource<T>(
  key: string | null,
  label: string,
  load: (signal?: AbortSignal) => Promise<T>,
  size?: (data: T) => number,
): Resource<T> {
  const gen = useSyncExternalStore(subscribe, () => generation, () => generation)
  const [, setTick] = useState(0)

  useEffect(() => {
    if (key === null) return
    const { setLoad } = useViewer.getState()
    let entry = cache.get(key)
    if (!entry) {
      const created: Entry = { promise: Promise.resolve(), settled: null }
      created.promise = load().then(
        (data) => {
          created.settled = { status: 'ready', data, error: null }
        },
        (error: unknown) => {
          created.settled = { status: 'error', data: null, error: errorText(error) }
        },
      )
      cache.set(key, created)
      entry = created
    }
    let mounted = true
    const report = () => {
      const settled = entry.settled
      if (!mounted || !settled) return
      if (settled.status === 'ready') {
        setLoad(key, { label, status: 'ready', bytes: size ? size(settled.data as T) : undefined })
      } else if (settled.status === 'error') {
        setLoad(key, { label, status: 'error', error: settled.error })
      }
      setTick((t) => t + 1)
    }
    if (entry.settled) report()
    else {
      setLoad(key, { label, status: 'loading' })
      void entry.promise.then(report)
    }
    return () => {
      mounted = false
      setLoad(key, null)
    }
    // `load` and `size` are identified by `key`; they are deliberately not dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, label, gen])

  if (key === null) return IDLE
  const settled = cache.get(key)?.settled
  return (settled as Resource<T> | null | undefined) ?? LOADING
}

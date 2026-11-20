/**
 * Loads the blocks of one model for the current query (filter, colour attribute, level
 * of detail, section slab) and keeps the previous blocks on screen until the new ones
 * are there.
 *
 * A change of query is debounced (a slider being dragged sends one request, not fifty),
 * the request in flight is aborted when a newer one starts, and the outcome is mirrored
 * to the block store (`status[id]`: loading / ready with counts / error with the
 * server's message) where the filter panel and the status bar read it.
 */
import { useEffect, useRef, useState } from 'react'
import { useBlocks } from '../state/blocks'
import { useViewer } from '../state/store'
import { blockQueryString, fetchBlocks, type BlockQuery, type BlockSet } from './blocks'
import { useRetryGeneration } from './useResource'

export const BLOCK_DEBOUNCE_MS = 220

export interface LoadedBlocks {
  set: BlockSet
  /** Query string the set answers. */
  key: string
  /** `performance.now()` when the request was sent, and how long it took, ms. */
  requestedAt: number
  requestMs: number
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export function useBlockSet(object: { id: string; name: string; href: string }, query: BlockQuery | null, debounceMs = BLOCK_DEBOUNCE_MS): LoadedBlocks | null {
  const [loaded, setLoaded] = useState<LoadedBlocks | null>(null)
  const first = useRef(true)
  // The sidebar's "Retry" (after a failed load) sends the request again.
  const retry = useRetryGeneration()
  const key = query ? blockQueryString(query) : null
  const { id, name, href } = object

  useEffect(() => {
    if (key === null || !query) return
    const { setStatus } = useBlocks.getState()
    const { setLoad } = useViewer.getState()
    const loadKey = `blocks:${id}`
    const controller = new AbortController()
    setStatus(id, { state: 'loading' })
    if (first.current) setLoad(loadKey, { label: name, status: 'loading' })
    const run = () => {
      const requestedAt = performance.now()
      fetchBlocks({ href }, query, controller.signal).then(
        (set) => {
          if (controller.signal.aborted) return
          first.current = false
          setLoaded({ set, key, requestedAt, requestMs: performance.now() - requestedAt })
          setStatus(id, {
            state: 'ready',
            error: null,
            shown: set.count,
            selected: set.selected,
            valued: set.valued,
            present: set.present,
            total: set.total,
            volume: set.selectedVolume,
            lod: set.lod,
            clipped: set.clipped,
          })
          setLoad(loadKey, { label: name, status: 'ready', bytes: set.byteLength })
        },
        (error: unknown) => {
          if (isAbort(error) || controller.signal.aborted) return
          const message = error instanceof Error ? error.message : String(error)
          // "400: <the engine's message>" -> the message alone.
          setStatus(id, { state: 'error', error: message.replace(/^400: /, '') })
          // A bad filter is not a failed load; nothing on screen at all is.
          if (first.current) setLoad(loadKey, { label: name, status: 'error', error: message })
        },
      )
    }
    // The first load has nothing to wait for.
    const timer = first.current || debounceMs <= 0 ? (run(), 0) : window.setTimeout(run, debounceMs)
    return () => {
      controller.abort()
      if (timer) window.clearTimeout(timer)
    }
    // `query` is identified by `key`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, name, href, key, debounceMs, retry])

  useEffect(() => () => useViewer.getState().setLoad(`blocks:${id}`, null), [id])
  return loaded
}

/**
 * Blast BakuMine state (zustand): the loaded bundle, the design form, the player clock
 * and the layer switches. The scene reads `t` inside its frame loop (`getState()`), so a
 * running clock re-renders only the components that select `t` (the timeline).
 */
import { create } from 'zustand'
import { loadDefaults, loadResult, simulate as apiSimulate, type Loaded } from './api'
import { DEFAULT_DESIGN, makeDemoBundle } from './fixture'
import { lastFire, timelineEnd } from './math'
import type { BlastBundle, DesignParams } from './types'

export const LAYERS = [
  { id: 'ground', label: 'Ground / muckpile' },
  { id: 'holes', label: 'Blast holes' },
  { id: 'labels', label: 'Delay labels' },
  { id: 'allLabels', label: 'All delay labels' },
  { id: 'waves', label: 'Stress waves' },
  { id: 'ppv', label: 'PPV map' },
  { id: 'damage', label: 'Damage blocks' },
  { id: 'throw', label: 'Throw (fragments)' },
  { id: 'flyrock', label: 'Flyrock + clearance' },
  { id: 'airblast', label: 'Airblast rings' },
  { id: 'monitors', label: 'Monitors' },
  { id: 'topography', label: 'Project topography' },
] as const

export type LayerId = (typeof LAYERS)[number]['id']

export const SPEEDS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1] as const

export interface BlastState {
  bundle: BlastBundle | null
  source: 'api' | 'demo'
  /** Why demo data is shown. */
  reason: string | null
  status: 'idle' | 'loading' | 'ready'
  /** Why the last Simulate did not give a result (the API's words). */
  error: string | null
  design: DesignParams

  /** Player clock, ms after the first initiation. */
  t: number
  tEnd: number
  lastFire: number
  playing: boolean
  speed: number
  /** Slow motion while holes fire, real time for the rock flight. */
  autoSpeed: boolean
  loop: boolean

  layers: Record<LayerId, boolean>
  /** Holes drawn much wider than they are, so that they can be seen. */
  exaggerateHoles: boolean
  groundOpacity: number
  /** Fragments drawn this many times their size. */
  fragmentScale: number

  setLoaded: (loaded: Loaded) => void
  load: (demoOnly: boolean) => Promise<void>
  simulate: () => Promise<void>
  patchDesign: (patch: Partial<DesignParams>) => void
  setTime: (t: number) => void
  setPlaying: (playing: boolean) => void
  togglePlay: () => void
  setSpeed: (speed: number) => void
  setAutoSpeed: (on: boolean) => void
  setLoop: (on: boolean) => void
  setLayer: (id: LayerId, on: boolean) => void
  setExaggerateHoles: (on: boolean) => void
  setGroundOpacity: (v: number) => void
  setFragmentScale: (v: number) => void
}

export const useBlast = create<BlastState>((set, get) => ({
  bundle: null,
  source: 'demo',
  reason: null,
  status: 'idle',
  error: null,
  design: DEFAULT_DESIGN,

  t: 0,
  tEnd: 1,
  lastFire: 0,
  playing: false,
  speed: 0.05,
  autoSpeed: true,
  loop: false,

  layers: {
    ground: true, holes: true, labels: true, allLabels: false, waves: true, ppv: false, damage: true, throw: true,
    flyrock: true, airblast: false, monitors: true, topography: true,
  },
  exaggerateHoles: true,
  groundOpacity: 1,
  fragmentScale: 1,

  setLoaded: ({ bundle, source, reason }) =>
    set({ bundle, source, reason, status: 'ready', error: null, t: 0, playing: false, tEnd: timelineEnd(bundle), lastFire: lastFire(bundle) }),

  load: async (demoOnly) => {
    set({ status: 'loading' })
    if (demoOnly) {
      get().setLoaded({ bundle: makeDemoBundle(get().design), source: 'demo', reason: 'Demo requested (?blast=demo)' })
      return
    }
    const defaults = await loadDefaults()
    if (defaults) set({ design: defaults })
    get().setLoaded(await loadResult(get().design))
  },

  simulate: async () => {
    set({ status: 'loading', playing: false, error: null })
    const demoOnly = get().reason?.startsWith('Demo requested') ?? false
    if (demoOnly) {
      get().setLoaded({ bundle: makeDemoBundle(get().design), source: 'demo', reason: 'Demo requested (?blast=demo)' })
      return
    }
    const hadResult = get().source === 'api' && get().bundle !== null
    try {
      const loaded = await apiSimulate(get().design)
      // A real result on screen is not replaced by demo numbers because the API went away.
      if (loaded.source === 'demo' && hadResult) set({ status: 'ready', error: loaded.reason })
      else get().setLoaded(loaded)
    } catch (error) {
      set({ status: get().bundle ? 'ready' : 'idle', error: error instanceof Error ? error.message : String(error) })
    }
  },

  patchDesign: (patch) => set((s) => ({ design: { ...s.design, ...patch } })),
  setTime: (t) => set((s) => ({ t: Math.min(s.tEnd, Math.max(0, t)) })),
  setPlaying: (playing) => set((s) => ({ playing, t: playing && s.t >= s.tEnd ? 0 : s.t })),
  togglePlay: () => get().setPlaying(!get().playing),
  setSpeed: (speed) => set({ speed, autoSpeed: false }),
  setAutoSpeed: (autoSpeed) => set({ autoSpeed }),
  setLoop: (loop) => set({ loop }),
  setLayer: (id, on) => set((s) => ({ layers: { ...s.layers, [id]: on } })),
  setExaggerateHoles: (exaggerateHoles) => set({ exaggerateHoles }),
  setGroundOpacity: (groundOpacity) => set({ groundOpacity }),
  setFragmentScale: (fragmentScale) => set({ fragmentScale }),
}))

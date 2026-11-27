/**
 * Blast BakuMine: design a bench blast and replay its simulation in 3D. A separate
 * product next to the block modeller (`?app=blast`); it only reads the modeller's objects.
 * Layout: left panel (design, layers, results), the 3D view, the timeline player below.
 */
import { useEffect, useState } from 'react'
import { demoRequested } from './product'
import { BlastViewport } from './scene/BlastViewport'
import { useBlast } from './state'
import { DesignPanel } from './ui/DesignPanel'
import { LayersPanel } from './ui/LayersPanel'
import { ResultsPanel } from './ui/ResultsPanel'
import { Timeline } from './ui/Timeline'

type Tab = 'design' | 'results'

export function BlastApp() {
  const [tab, setTab] = useState<Tab>('results')
  useEffect(() => {
    if (useBlast.getState().status === 'idle') void useBlast.getState().load(demoRequested(window.location.search))
  }, [])

  return (
    <>
      <aside className="blast-panel" data-testid="blast-panel">
        <div className="group blast-tabs" role="tablist" aria-label="Blast panel">
          {(['design', 'results'] as const).map((id) => (
            <button key={id} type="button" role="tab" aria-selected={tab === id} aria-pressed={tab === id} data-testid={`blast-tab-${id}`} onClick={() => setTab(id)}>
              {id === 'design' ? 'Design' : 'Results'}
            </button>
          ))}
        </div>
        {tab === 'design' ? <DesignPanel /> : <ResultsPanel />}
        <LayersPanel />
      </aside>
      <BlastViewport />
      <Timeline />
    </>
  )
}

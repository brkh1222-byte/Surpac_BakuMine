/**
 * The performance probe scene (`?mock=perf`): large client-generated objects drawn by
 * the real layers, without the API. Sizes come from the query string:
 * `holes` (500), `per` intervals per hole (200).
 */
import { useLayoutEffect, useMemo } from 'react'
import { useDisplay } from '../state/display'
import { boundsOf } from '../state/store'
import { DrillholesLayer } from './DrillholesLayer'
import type { LayerObject } from './layers'
import { buildPerfDrillholes } from './perfData'

function queryNumber(name: string, fallback: number): number {
  const value = Number(new URLSearchParams(window.location.search).get(name))
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback
}

function PerfDrillholes() {
  const data = useMemo(() => buildPerfDrillholes(queryNumber('holes', 500), queryNumber('per', 200)), [])
  const object = useMemo<LayerObject>(
    () => ({ id: 'perf-holes', name: 'Generated holes', category: 'drillholes', bounds: boundsOf(data.set.pointXyz) }),
    [data],
  )
  useLayoutEffect(() => {
    useDisplay.getState().registerDrillholes(object.id, data.attributes)
    return () => useDisplay.getState().unregisterDrillholes(object.id)
  }, [object.id, data])
  const colorBy = useDisplay((s) => s.drillholes[object.id]?.colorBy ?? null)
  return <DrillholesLayer object={object} set={data.set} colorBy={colorBy} />
}

export function PerfScene() {
  return <PerfDrillholes />
}

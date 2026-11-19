/**
 * The performance probe scene (`?mock=perf`): large client-generated objects drawn by
 * the real layers, without the API. Sizes come from the query string:
 * `holes` (500), `per` intervals per hole (200), `tris` triangles of the surface
 * (1 000 000; 0 leaves the surface out).
 */
import { useLayoutEffect, useMemo } from 'react'
import { useDisplay } from '../state/display'
import { boundsOf } from '../state/store'
import { DrillholesLayer } from './DrillholesLayer'
import type { LayerObject } from './layers'
import { zRangeOf } from '../meshes/display'
import { MeshLayer } from './MeshLayer'
import { buildPerfDrillholes, buildPerfMesh } from './perfData'

function queryNumber(name: string, fallback: number): number {
  const text = new URLSearchParams(window.location.search).get(name)
  const value = Number(text)
  return text !== null && Number.isFinite(value) && value >= 0 ? Math.floor(value) : fallback
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

function PerfMesh({ triangles }: { triangles: number }) {
  const mesh = useMemo(() => buildPerfMesh(triangles), [triangles])
  const object = useMemo<LayerObject>(
    () => ({ id: 'perf-mesh', name: 'Generated surface', category: 'surfaces', bounds: boundsOf(mesh.vertices) }),
    [mesh],
  )
  useLayoutEffect(() => {
    useDisplay.getState().registerMesh(object.id, false, zRangeOf(mesh.vertices))
    return () => useDisplay.getState().unregisterMesh(object.id)
  }, [object.id, mesh])
  return <MeshLayer object={object} mesh={mesh} />
}

export function PerfScene() {
  const holes = queryNumber('holes', 500)
  const triangles = queryNumber('tris', 1_000_000)
  return (
    <>
      {triangles > 0 && <PerfMesh triangles={triangles} />}
      {holes > 0 && <PerfDrillholes />}
    </>
  )
}

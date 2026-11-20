/**
 * The performance probe scene (`?mock=perf`): large client-generated objects drawn by
 * the real layers, without the API. Sizes come from the query string:
 * `holes` (500), `per` intervals per hole (200), `tris` triangles of the surface
 * (1 000 000; 0 leaves the surface out), `blocks` blocks of a generated block model
 * (none by default; `blocks=2000000` is the 2M-block probe).
 */
import { useLayoutEffect, useMemo } from 'react'
import { fetchBlockInfo } from '../api/blocks'
import { useResource } from '../api/useResource'
import { useDisplay } from '../state/display'
import { boundsOf } from '../state/store'
import { BlocksObject } from './BlocksObject'
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

/**
 * A generated dense block model of about `count` blocks. Unlike the other perf objects
 * it comes from the API (`perf-<count>`, bakumine/api/blocks.py), because transport is
 * part of what is measured: the same endpoint, payload and layer as a real model.
 */
function PerfBlocks({ count }: { count: number }) {
  const info = useResource(`blockinfo:perf-${count}`, 'Generated blocks', (signal) => fetchBlockInfo(`perf-${count}`, signal))
  const object = useMemo(() => (info.status === 'ready' ? { ...info.data, name: 'Generated blocks' } : null), [info])
  return object ? <BlocksObject object={object} /> : null
}

export function PerfScene() {
  const holes = queryNumber('holes', 500)
  const triangles = queryNumber('tris', 1_000_000)
  const blocks = queryNumber('blocks', 0)
  return (
    <>
      {triangles > 0 && <PerfMesh triangles={triangles} />}
      {holes > 0 && <PerfDrillholes />}
      {blocks > 0 && <PerfBlocks count={blocks} />}
    </>
  )
}

import { useEffect, useMemo } from 'react'
import { BufferAttribute, BufferGeometry } from 'three'
import { useLayerObject } from './layers'
import { buildMockData, type MockData } from './mockData'
import { worldToScene } from './origin'

function MockSolids({ data }: { data: MockData }) {
  const { ready, visible, origin } = useLayerObject({
    id: 'mock-solids',
    name: 'Mock boxes',
    category: 'solids',
    bounds: data.boxBounds,
  })
  const geometry = useMemo(() => {
    if (!ready) return null
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(worldToScene(data.boxPositions, origin), 3))
    g.setAttribute('normal', new BufferAttribute(data.boxNormals, 3))
    g.setAttribute('color', new BufferAttribute(data.boxColors, 3))
    g.computeBoundingSphere()
    return g
  }, [ready, origin, data])
  useEffect(() => () => geometry?.dispose(), [geometry])

  if (!geometry) return null
  return (
    <mesh geometry={geometry} visible={visible} name="mock-solids">
      <meshLambertMaterial vertexColors />
    </mesh>
  )
}

function MockTraces({ data }: { data: MockData }) {
  const { ready, visible, origin } = useLayerObject({
    id: 'mock-traces',
    name: 'Mock traces',
    category: 'drillholes',
    bounds: data.lineBounds,
  })
  const geometry = useMemo(() => {
    if (!ready) return null
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(worldToScene(data.linePositions, origin), 3))
    g.setAttribute('color', new BufferAttribute(data.lineColors, 3))
    g.computeBoundingSphere()
    return g
  }, [ready, origin, data])
  useEffect(() => () => geometry?.dispose(), [geometry])

  if (!geometry) return null
  return (
    <lineSegments geometry={geometry} visible={visible} name="mock-traces">
      <lineBasicMaterial vertexColors />
    </lineSegments>
  )
}

/** Boxes and polylines at E 500 000 / N 5 300 000 / Z 1 200, shown with `?mock=1`. */
export function MockLayer() {
  const data = useMemo(() => buildMockData(), [])
  return (
    <>
      <MockSolids data={data} />
      <MockTraces data={data} />
    </>
  )
}

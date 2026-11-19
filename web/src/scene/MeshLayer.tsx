/**
 * Triangle meshes: DTM surfaces and closed solids.
 *
 * One indexed BufferGeometry per object, built once; a display change only swaps
 * material properties, so there is no CPU work per frame and one draw call per object
 * (two in "shaded + edges": the fill and its triangle edges).
 *
 *   shaded       lit, opaque, two-sided (open surfaces are seen from below as well)
 *   edges        the same with the triangle edges drawn over it; the fill is pushed
 *                back with a polygon offset so the edges do not z-fight
 *   wireframe    triangle edges only, in the object colour
 *   transparent  alpha-blended and NOT written to the depth buffer: whatever is inside
 *                or behind (drillholes, other solids) stays visible, and because the
 *                object has one colour, the blend does not depend on triangle order
 *
 * Smooth shading uses vertex normals (from the API, else computed); flat shading is the
 * material's derivative-based option, so both share the same buffers.
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BufferAttribute, BufferGeometry, Color, DoubleSide } from 'three'
import type { MeshData } from '../api/client'
import { elevationColors, elevationLegend } from '../meshes/display'
import { useDisplay } from '../state/display'
import { useViewer, type Theme } from '../state/store'
import { useLayerObject, type LayerObject } from './layers'
import { worldToScene } from './origin'
import { SectionOutline } from './SectionScene'

/** Above this many triangles the mesh is skipped by the cursor raycast (too slow per move). */
const RAYCAST_LIMIT = 300_000

/** Edge colour over a shaded fill: the object colour pulled towards the canvas ink. */
function edgeColor(color: string, theme: Theme): Color {
  return new Color(color).lerp(new Color(theme === 'dark' ? '#0b0d10' : '#1e2327'), 0.62)
}

export interface MeshLayerProps {
  object: LayerObject
  mesh: MeshData
}

export function MeshLayer({ object, mesh }: MeshLayerProps) {
  const { ready, visible, origin } = useLayerObject(object)
  const invalidate = useThree((s) => s.invalidate)
  const theme = useViewer((s) => s.theme)
  const display = useDisplay((s) => s.meshes[object.id])

  const geometry = useMemo(() => {
    if (!ready) return null
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(worldToScene(mesh.vertices, origin), 3))
    g.setIndex(new BufferAttribute(mesh.faces, 1))
    if (mesh.normals) g.setAttribute('normal', new BufferAttribute(mesh.normals, 3))
    else g.computeVertexNormals()
    g.computeBoundingSphere()
    g.computeBoundingBox()
    return g
  }, [ready, origin, mesh])
  useEffect(() => () => geometry?.dispose(), [geometry])

  // Elevation colours: a vertex attribute filled once per change of range, type or theme.
  const byElevation = display?.colorByElevation ?? false
  const spec = useMemo(
    () => (display && byElevation ? elevationLegend(display) : null),
    // Only the elevation settings matter here, not the whole display object.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [byElevation, display?.elevation, display?.zRange],
  )
  // Filled while rendering (not in an effect) so that the material below is created
  // with `vertexColors` in the same pass that makes the attribute exist.
  const tinted = useMemo(() => {
    if (!geometry || !spec) return false
    const existing = geometry.getAttribute('color') as BufferAttribute | undefined
    if (existing) {
      elevationColors(mesh.vertices, spec, theme, existing.array as Float32Array)
      existing.needsUpdate = true
    } else {
      geometry.setAttribute('color', new BufferAttribute(elevationColors(mesh.vertices, spec, theme), 3))
    }
    return true
  }, [geometry, spec, theme, mesh])

  useEffect(() => {
    const { setLegend, setMesh } = useDisplay.getState()
    const key = `${object.id}:elevation`
    if (!spec || !visible) {
      setLegend(key, null)
      return
    }
    setLegend(key, {
      spec,
      subject: object.name,
      onChange: (patch) => {
        const current = useDisplay.getState().meshes[object.id]
        if (current) setMesh(object.id, { elevation: { ...current.elevation, ...patch } })
      },
      onReset: () => setMesh(object.id, { elevation: undefined }),
    })
  }, [spec, visible, object.id, object.name])
  useEffect(() => () => useDisplay.getState().setLegend(`${object.id}:elevation`, null), [object.id])

  useEffect(() => invalidate(), [display, tinted, theme, invalidate])

  if (!geometry || !display) return null
  const { mode, color, opacity, shading } = display
  const fill = mode !== 'wireframe'
  const lines = mode === 'wireframe' || mode === 'edges'
  const seeThrough = mode === 'transparent' || mode === 'wireframe'
  const heavy = mesh.faces.length / 3 > RAYCAST_LIMIT
  const userData = { noRaycast: heavy, seeThrough }

  return (
    <group visible={visible} name={`mesh:${object.id}`}>
      {visible && <SectionOutline id={object.id} vertices={mesh.vertices} faces={mesh.faces} color={color} origin={origin} />}
      {fill && (
        <mesh geometry={geometry} name={`mesh-fill:${object.id}`} userData={userData}>
          <meshLambertMaterial
            // Recompile when a shader-affecting option changes.
            key={`${shading}-${tinted}-${mode === 'transparent'}`}
            color={tinted ? '#ffffff' : color}
            vertexColors={tinted}
            side={DoubleSide}
            flatShading={shading === 'flat'}
            transparent={mode === 'transparent'}
            opacity={mode === 'transparent' ? opacity : 1}
            depthWrite={mode !== 'transparent'}
            // three would draw a two-sided transparent mesh twice (back faces, then
            // front faces); with one object colour the blend is order-independent.
            forceSinglePass
            polygonOffset={mode === 'edges'}
            polygonOffsetFactor={1}
            polygonOffsetUnits={1}
          />
        </mesh>
      )}
      {lines && (
        <mesh geometry={geometry} name={`mesh-edges:${object.id}`} userData={{ noRaycast: fill || heavy, seeThrough: true }}>
          <meshBasicMaterial
            key={`wire-${tinted && mode === 'wireframe'}`}
            wireframe
            color={mode === 'edges' ? edgeColor(color, theme) : tinted ? '#ffffff' : color}
            vertexColors={tinted && mode === 'wireframe'}
          />
        </mesh>
      )}
    </group>
  )
}

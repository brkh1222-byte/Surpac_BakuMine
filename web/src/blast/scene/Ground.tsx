/**
 * The ground: the muckpile heightmap morphing from z_before to z_after as fragments land,
 * a dim context surface from the PPV grid around it (demo / no topography), and, when the
 * API serves one, the project's topography mesh underneath.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useState } from 'react'
import { BufferAttribute, BufferGeometry, DoubleSide } from 'three'
import { fetchManifest, fetchMesh, type MeshData } from '../../api/client'
import { worldToScene } from '../../scene/origin'
import { useViewer } from '../../state/store'
import { landedFraction, sortedFinite } from '../math'
import { useBlast } from '../state'
import type { GridSpec } from '../types'
import { BLAST_COLORS, hexLinear, type LayerProps } from './common'

/** Triangles of a node grid; cells with a missing node (or rejected by `skip`) are left out. */
function gridIndex(grid: GridSpec, ok: (i: number) => boolean, skip?: (ix: number, iy: number) => boolean): Uint32Array {
  const [nx, ny] = grid.shape
  const out: number[] = []
  for (let ix = 0; ix + 1 < nx; ix++) {
    for (let iy = 0; iy + 1 < ny; iy++) {
      const a = ix * ny + iy
      const b = (ix + 1) * ny + iy
      const c = (ix + 1) * ny + iy + 1
      const d = ix * ny + iy + 1
      if (!ok(a) || !ok(b) || !ok(c) || !ok(d) || skip?.(ix, iy)) continue
      out.push(a, b, c, a, c, d)
    }
  }
  return Uint32Array.from(out)
}

function gridPositions(grid: GridSpec, z: readonly (number | null)[], origin: readonly number[], lift = 0): Float32Array {
  const [nx, ny] = grid.shape
  const out = new Float32Array(nx * ny * 3)
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      const i = ix * ny + iy
      out[i * 3] = grid.origin[0] + ix * grid.spacing - origin[0]
      out[i * 3 + 1] = grid.origin[1] + iy * grid.spacing - origin[1]
      out[i * 3 + 2] = (z[i] ?? 0) - origin[2] + lift
    }
  }
  return out
}

const finite = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)

export function Muckpile({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const opacity = useBlast((s) => s.groundOpacity)
  const grid = bundle.muckpile

  const built = useMemo(() => {
    const { z_before: zb, z_after: za } = grid
    const n = grid.shape[0] * grid.shape[1]
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(gridPositions(grid, zb, origin), 3))
    g.setAttribute('color', new BufferAttribute(new Float32Array(n * 3), 3))
    g.setIndex(new BufferAttribute(gridIndex(grid, (i) => finite(zb[i]) && finite(za[i])), 1))
    return { g, landed: sortedFinite(bundle.throw.particles.t_land_ms), state: { f: -1, theme: '' } }
  }, [grid, origin, bundle])
  useEffect(() => () => built.g.dispose(), [built])

  useFrame(() => {
    const f = landedFraction(built.landed, useBlast.getState().t)
    if (f === built.state.f && theme === built.state.theme) return
    built.state.f = f
    built.state.theme = theme
    const position = built.g.getAttribute('position') as BufferAttribute
    const color = built.g.getAttribute('color') as BufferAttribute
    const ground = hexLinear(BLAST_COLORS[theme].ground)
    const muck = hexLinear(BLAST_COLORS[theme].muck)
    const { z_before: zb, z_after: za } = grid
    for (let i = 0; i < position.count; i++) {
      const a = zb[i] ?? 0
      const b = za[i] ?? a
      position.setZ(i, a + (b - a) * f - origin[2])
      // Where the surface moved, it is broken rock.
      const m = Math.min(1, Math.abs(b - a) / 0.5) * f
      color.setXYZ(i, ground[0] + (muck[0] - ground[0]) * m, ground[1] + (muck[1] - ground[1]) * m, ground[2] + (muck[2] - ground[2]) * m)
    }
    position.needsUpdate = true
    color.needsUpdate = true
    built.g.computeVertexNormals()
    built.g.computeBoundingSphere()
  })

  return (
    <mesh geometry={built.g} renderOrder={2}>
      <meshStandardMaterial vertexColors side={DoubleSide} roughness={1} metalness={0} flatShading transparent={opacity < 1} opacity={opacity} />
    </mesh>
  )
}

/** The PPV grid's surface as plain ground around the muckpile patch. */
export function ContextGround({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const geometry = useMemo(() => {
    const grid = bundle.ppv_surface
    const m = bundle.muckpile
    const x0 = m.origin[0]
    const y0 = m.origin[1]
    const x1 = x0 + (m.shape[0] - 1) * m.spacing
    const y1 = y0 + (m.shape[1] - 1) * m.spacing
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(gridPositions(grid, grid.z, origin, -0.05), 3))
    g.setIndex(
      new BufferAttribute(
        gridIndex(grid, (i) => finite(grid.z[i]), (ix, iy) => {
          // The muckpile patch draws this part itself.
          const ax = grid.origin[0] + ix * grid.spacing
          const ay = grid.origin[1] + iy * grid.spacing
          return ax >= x0 && ax + grid.spacing <= x1 && ay >= y0 && ay + grid.spacing <= y1
        }),
        1,
      ),
    )
    g.computeVertexNormals()
    return g
  }, [bundle, origin])
  useEffect(() => () => geometry.dispose(), [geometry])
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={BLAST_COLORS[theme].context} side={DoubleSide} roughness={1} metalness={0} />
    </mesh>
  )
}

/** The project's topography (first surface whose bounds hold the blast), dimmed. */
export function Topography({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const invalidate = useThree((s) => s.invalidate)
  const [mesh, setMesh] = useState<MeshData | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setMesh(null)
    ;(async () => {
      try {
        const manifest = await fetchManifest(controller.signal)
        const surfaces = manifest.objects.filter((o) => o.kind === 'mesh' && o.category === 'surfaces' && o.bounds)
        const holds = (o: (typeof surfaces)[number]) =>
          o.bounds !== null && o.bounds !== undefined &&
          origin[0] >= o.bounds.min[0] && origin[0] <= o.bounds.max[0] && origin[1] >= o.bounds.min[1] && origin[1] <= o.bounds.max[1]
        const inside = surfaces.filter(holds)
        const pick = inside.find((o) => /topo|dtm|dem|surface/i.test(o.name)) ?? inside[0]
        if (!pick) return
        const data = await fetchMesh(pick, controller.signal)
        if (!controller.signal.aborted) setMesh(data)
      } catch {
        // No project or no surface: the blast draws without it.
      }
    })()
    return () => controller.abort()
  }, [bundle, origin])

  const geometry = useMemo(() => {
    if (!mesh) return null
    const g = new BufferGeometry()
    const position = worldToScene(mesh.vertices, origin)
    // A hair below the muckpile patch, which shows the same ground in more detail.
    for (let i = 2; i < position.length; i += 3) position[i] -= 0.1
    g.setAttribute('position', new BufferAttribute(position, 3))
    g.setIndex(new BufferAttribute(mesh.faces, 1))
    if (mesh.normals) g.setAttribute('normal', new BufferAttribute(mesh.normals, 3))
    else g.computeVertexNormals()
    return g
  }, [mesh, origin])
  useEffect(() => {
    invalidate()
    return () => geometry?.dispose()
  }, [geometry, invalidate])

  if (!geometry) return null
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={BLAST_COLORS[theme].context} side={DoubleSide} roughness={1} metalness={0} />
    </mesh>
  )
}

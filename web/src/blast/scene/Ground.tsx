/**
 * The ground: the muckpile heightmap morphing from z_before to z_after as fragments land,
 * a dim context surface from the PPV grid around it (demo / no topography), and, when the
 * API serves one, the project's topography mesh underneath.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BufferAttribute, BufferGeometry, DoubleSide, Plane, Vector3, type Texture } from 'three'
import { worldToScene } from '../../scene/origin'
import { useViewer } from '../../state/store'
import { uvArray, useSatelliteTexture, type ImageryPlan } from '../imagery'
import { landedFraction, sortedFinite } from '../math'
import { useBlast } from '../state'
import type { GridSpec } from '../types'
import { BLAST_COLORS, hexLinear, type LayerProps } from './common'
import { ShadowCatcher } from './Effects'
import { cutRock, MUCK_COLOR, ROCK_COLOR, TerrainMaterial } from './terrain'
import { reliefWeight } from './realism'
import { useTopography } from './topography'

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

/** The project topography is drawn this far below its true elevation, m. */
const TOPOGRAPHY_DROP = 0.5

const finite = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)

/** 1 on a slope of `gradient` (rise / run) up to 35 degrees, falling to 0.45 from 60 degrees: how much of the satellite picture a slope keeps. */
export function steepShade(gradient: number): number {
  const lo = Math.tan((35 * Math.PI) / 180)
  const hi = Math.tan((60 * Math.PI) / 180)
  const t = Math.min(1, Math.max(0, (gradient - lo) / (hi - lo)))
  return 1 - 0.55 * t * t * (3 - 2 * t)
}

/** The satellite mosaic and its plan while the layer is on and the tiles are loaded. */
function useSatellite(): { plan: ImageryPlan; texture: Texture } | null {
  const plan = useBlast((s) => s.imagery)
  const on = useBlast((s) => s.layers.satellite && s.source === 'api')
  const invalidate = useThree((s) => s.invalidate)
  const texture = useSatelliteTexture(plan, on)
  useEffect(() => invalidate(), [texture, invalidate])
  return useMemo(() => (plan && texture ? { plan, texture } : null), [plan, texture])
}

export function Muckpile({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const opacity = useBlast((s) => s.groundOpacity)
  const grid = bundle.muckpile
  const satellite = useSatellite()

  const built = useMemo(() => {
    const { z_before: zb, z_after: za } = grid
    const n = grid.shape[0] * grid.shape[1]
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(gridPositions(grid, zb, origin), 3))
    g.setAttribute('color', new BufferAttribute(new Float32Array(n * 3), 3))
    // How much of the node is drawn as rock, not as the satellite picture (terrain.tsx).
    g.setAttribute('rock', new BufferAttribute(new Float32Array(n), 1))
    // A cell with a missing node is not drawn (never at z = 0); a node missing only in one state keeps the other.
    g.setIndex(new BufferAttribute(gridIndex(grid, (i) => finite(zb[i]) || finite(za[i])), 1))
    return { g, landed: sortedFinite(bundle.throw.particles.t_land_ms), state: { f: -1, theme: '' } }
  }, [grid, origin, bundle])
  useEffect(() => () => built.g.dispose(), [built])
  useEffect(() => {
    if (satellite) built.g.setAttribute('uv', new BufferAttribute(uvArray(satellite.plan, built.g.getAttribute('position').array, origin), 2))
    built.state.f = -1 // recolour: the ground is white under the picture
  }, [built, satellite, origin])

  useFrame(() => {
    const f = landedFraction(built.landed, useBlast.getState().t)
    const key = satellite ? `${theme}+satellite` : theme
    if (f === built.state.f && key === built.state.theme) return
    built.state.f = f
    built.state.theme = key
    const position = built.g.getAttribute('position') as BufferAttribute
    const color = built.g.getAttribute('color') as BufferAttribute
    const rock = built.g.getAttribute('rock') as BufferAttribute
    const ground = satellite ? hexLinear(ROCK_COLOR) : hexLinear(BLAST_COLORS[theme].ground)
    const muck = hexLinear(satellite ? MUCK_COLOR : BLAST_COLORS[theme].muck)
    const cut = grid.cut_m
    const { z_before: zb, z_after: za } = grid
    for (let i = 0; i < position.count; i++) {
      const before = zb[i]
      const after = za[i]
      const a = finite(before) ? before : finite(after) ? after : origin[2]
      const b = finite(after) ? after : a
      position.setZ(i, a + (b - a) * f - origin[2])
      // Where the surface moved, it is broken rock.
      const m = Math.min(1, Math.abs(b - a) / 0.5) * f
      color.setXYZ(i, ground[0] + (muck[0] - ground[0]) * m, ground[1] + (muck[1] - ground[1]) * m, ground[2] + (muck[2] - ground[2]) * m)
      // Broken rock and cut ground are rock; the rest keeps the picture.
      rock.setX(i, Math.max(m, cutRock(cut?.[i])))
    }
    if (satellite) {
      // A picture taken from above smears down a steep face: steep ground is rock too.
      const [nx, ny] = grid.shape
      for (let ix = 0; ix < nx; ix++) {
        for (let iy = 0; iy < ny; iy++) {
          const i = ix * ny + iy
          const zx = position.getZ(Math.min(nx - 1, ix + 1) * ny + iy) - position.getZ(Math.max(0, ix - 1) * ny + iy)
          const zy = position.getZ(ix * ny + Math.min(ny - 1, iy + 1)) - position.getZ(ix * ny + Math.max(0, iy - 1))
          const steep = (1 - steepShade(Math.hypot(zx, zy) / (2 * grid.spacing))) / 0.55
          if (steep > rock.getX(i)) rock.setX(i, steep)
        }
      }
    }
    position.needsUpdate = true
    color.needsUpdate = true
    rock.needsUpdate = true
    built.g.computeVertexNormals()
    built.g.computeBoundingSphere()
  })

  return (
    <>
      {satellite && opacity >= 1 && <ShadowCatcher geometry={built.g} />}
      <mesh geometry={built.g} renderOrder={2} frustumCulled={false} castShadow={satellite !== null}>
        {satellite ? (
          <TerrainMaterial key="satellite" texture={satellite.texture} opacity={opacity} />
        ) : (
          <meshStandardMaterial key="plain" vertexColors side={DoubleSide} roughness={1} metalness={0} flatShading transparent={opacity < 1} opacity={opacity} />
        )}
      </mesh>
    </>
  )
}

/** The PPV grid's surface as plain ground around the muckpile patch. */
export function ContextGround({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const satellite = useSatellite()
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
    if (satellite) g.setAttribute('uv', new BufferAttribute(uvArray(satellite.plan, g.getAttribute('position').array, origin), 2))
    return g
  }, [bundle, origin, satellite])
  useEffect(() => () => geometry.dispose(), [geometry])
  return (
    <mesh geometry={geometry}>
      {satellite ? (
        <TerrainMaterial key="satellite" texture={satellite.texture} />
      ) : (
        <meshStandardMaterial key="plain" color={BLAST_COLORS[theme].context} side={DoubleSide} roughness={1} metalness={0} />
      )}
    </mesh>
  )
}

/** The project's topography (first surface whose bounds hold the blast), dimmed. */
export function Topography({ bundle, origin }: LayerProps) {
  const theme = useViewer((s) => s.theme)
  const invalidate = useThree((s) => s.invalidate)
  const satellite = useSatellite()
  const mesh = useTopography(bundle, origin)?.mesh ?? null

  const geometry = useMemo(() => {
    if (!mesh) return null
    const g = new BufferGeometry()
    const position = worldToScene(mesh.vertices, origin)
    // Half a metre down: where it overlaps the bundle's ground, that one wins without z-fighting.
    for (let i = 2; i < position.length; i += 3) position[i] -= TOPOGRAPHY_DROP
    g.setAttribute('position', new BufferAttribute(position, 3))
    g.setIndex(new BufferAttribute(mesh.faces, 1))
    if (mesh.normals) g.setAttribute('normal', new BufferAttribute(mesh.normals, 3))
    else g.computeVertexNormals()
    if (satellite) g.setAttribute('uv', new BufferAttribute(uvArray(satellite.plan, position, origin), 2))
    return g
  }, [mesh, origin, satellite])
  useEffect(() => {
    invalidate()
    return () => geometry?.dispose()
  }, [geometry, invalidate])

  // Not drawn inside the heightmap's rectangle: there the bundle's ground (with the cut
  // face and the muckpile) replaces it. With `clipIntersection` a fragment is dropped only
  // when it is on the clipped side of all four planes, i.e. inside the rectangle.
  const planes = useMemo(() => {
    const m = bundle.muckpile
    const inset = m.spacing
    const x0 = m.origin[0] - origin[0] + inset
    const y0 = m.origin[1] - origin[1] + inset
    const x1 = m.origin[0] - origin[0] + (m.shape[0] - 1) * m.spacing - inset
    const y1 = m.origin[1] - origin[1] + (m.shape[1] - 1) * m.spacing - inset
    if (!(x1 > x0) || !(y1 > y0)) return []
    return [
      new Plane(new Vector3(1, 0, 0), -x1),
      new Plane(new Vector3(-1, 0, 0), x0),
      new Plane(new Vector3(0, 1, 0), -y1),
      new Plane(new Vector3(0, -1, 0), y0),
    ]
  }, [bundle, origin])

  if (!geometry) return null
  return (
    <>
      {satellite && <ShadowCatcher geometry={geometry} planes={planes} />}
      <mesh geometry={geometry}>
        {satellite ? (
          <TerrainMaterial key="satellite" texture={satellite.texture} planes={planes} offset />
        ) : (
          <meshStandardMaterial
            key="plain"
            color={BLAST_COLORS[theme].context}
            side={DoubleSide}
            roughness={1}
            metalness={0}
            clippingPlanes={planes}
            clipIntersection
            polygonOffset
            polygonOffsetFactor={2}
            polygonOffsetUnits={2}
          />
        )}
      </mesh>
    </>
  )
}

/** Grid spacing of the refined ground: around the blast and (finer) around the machines, m. */
const DETAIL_CELL_M = [2.5, 1] as const

/**
 * The ground refined near the blast and the machines (site view): the surveyed surface
 * resampled on a fine grid with the invented fine relief added (realism.groundRelief). It
 * lies on or above the surveyed surface, which stays drawn underneath and beyond.
 */
export function TerrainDetail({ bundle, origin }: LayerProps) {
  const invalidate = useThree((s) => s.invalidate)
  const satellite = useSatellite()
  const topography = useTopography(bundle, origin)
  const geometries = useMemo(() => {
    if (!topography || !satellite) return []
    const { hole } = topography
    return topography.patches.map((patch, k) => {
      const cell = DETAIL_CELL_M[Math.min(k, DETAIL_CELL_M.length - 1)]
      const n = Math.round((2 * patch.half) / cell) + 1
      const position = new Float32Array(n * n * 3)
      const ok = new Uint8Array(n * n)
      for (let ix = 0; ix < n; ix++) {
        for (let iy = 0; iy < n; iy++) {
          const x = patch.x - patch.half + ix * cell
          const y = patch.y - patch.half + iy * cell
          const base = topography.surveyed(x, y)
          const i = ix * n + iy
          // A hair above the surveyed surface where the relief is zero, so the two do not flicker.
          const lift = reliefWeight(x, y, topography.patches, hole) > 0 ? 0.03 : -1
          ok[i] = Number.isFinite(base) && lift > 0 ? 1 : 0
          position[i * 3] = x - origin[0]
          position[i * 3 + 1] = y - origin[1]
          position[i * 3 + 2] = (Number.isFinite(base) ? topography.heightAt(x, y) : 0) - origin[2] + lift
        }
      }
      const index: number[] = []
      for (let ix = 0; ix + 1 < n; ix++) {
        for (let iy = 0; iy + 1 < n; iy++) {
          const a = ix * n + iy
          const b = (ix + 1) * n + iy
          const c = (ix + 1) * n + iy + 1
          const d = ix * n + iy + 1
          if (!ok[a] || !ok[b] || !ok[c] || !ok[d]) continue
          // Not over the bench and muckpile: that ground is the simulation's.
          const x = patch.x - patch.half + ix * cell
          const y = patch.y - patch.half + iy * cell
          if (x + cell > hole[0] && x < hole[2] && y + cell > hole[1] && y < hole[3]) continue
          index.push(a, b, c, a, c, d)
        }
      }
      const g = new BufferGeometry()
      g.setAttribute('position', new BufferAttribute(position, 3))
      g.setIndex(new BufferAttribute(Uint32Array.from(index), 1))
      g.computeVertexNormals()
      g.setAttribute('uv', new BufferAttribute(uvArray(satellite.plan, position, origin), 2))
      return g
    })
  }, [topography, satellite, origin])
  useEffect(() => {
    invalidate()
    return () => {
      for (const g of geometries) g.dispose()
    }
  }, [geometries, invalidate])
  if (!satellite) return null
  return (
    <>
      {geometries.map((g, i) => (
        <group key={i}>
          <ShadowCatcher geometry={g} />
          <mesh geometry={g} renderOrder={1}>
            <TerrainMaterial texture={satellite.texture} />
          </mesh>
        </group>
      ))}
    </>
  )
}

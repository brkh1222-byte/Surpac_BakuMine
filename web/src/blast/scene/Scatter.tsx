/**
 * Loose stones, low shrubs and dry grass tufts on the ground near the blast and around the equipment. They
 * are invented scenery (no survey of them exists): they give the flat satellite picture
 * depth and a sense of scale when the camera is close. Their places come from a fixed
 * pseudo-random sequence; none is put on the cut bench or the muckpile area.
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { Color, ConeGeometry, IcosahedronGeometry, InstancedMesh, Matrix4, MeshStandardMaterial, Quaternion, Vector3, type BufferGeometry } from 'three'
import type { LayerProps } from './common'
import { blastCentre, equipmentSites, hash01 } from './realism'
import { rockGeometry } from './Throw'
import { useTopography } from './topography'

/** Stones and shrubs are scattered within these radii, m. */
const NEAR_BLAST_M = 280
const NEAR_EQUIPMENT_M = 110
const STONES = 4200
const SHRUBS = 3000
const STONE_TONES = ['#8a8378', '#75695b', '#9c9488', '#5f5a53'] as const
const TUFTS = 7000
const TUFT_TONES = ['#b3a36c', '#c2b27a', '#9c9160', '#a89a58'] as const
const SHRUB_TONES = ['#6d7150', '#7c7956', '#5d6446', '#8a8362'] as const

export function Scatter({ bundle, origin }: LayerProps) {
  const invalidate = useThree((s) => s.invalidate)
  const topography = useTopography(bundle, origin)

  const meshes = useMemo(() => {
    if (!topography) return null
    const [cx, cy] = blastCentre(bundle)
    const pile = bundle.muckpile
    const px1 = pile.origin[0] + (pile.shape[0] - 1) * pile.spacing
    const py1 = pile.origin[1] + (pile.shape[1] - 1) * pile.spacing
    const discs = [{ x: cx, y: cy, r: NEAR_BLAST_M }, ...equipmentSites(bundle).map((s) => ({ x: s.x, y: s.y, r: NEAR_EQUIPMENT_M }))]
    const wide = [0.9, 0.05, 0.05]
    const close = [0.45, 0.275, 0.275]

    const make = (count: number, salt: number, geometry: BufferGeometry, tones: readonly string[], kind: 'stone' | 'shrub' | 'tuft') => {
      const shrub = kind !== 'stone'
      const share = kind === 'tuft' ? close : wide
      const mesh = new InstancedMesh(geometry, new MeshStandardMaterial({ roughness: 1, metalness: 0, flatShading: true }), count)
      const m = new Matrix4()
      const q = new Quaternion()
      const up = new Vector3(0, 0, 1)
      const scale = new Vector3()
      const position = new Vector3()
      const colour = new Color()
      let n = 0
      for (let i = 0; i < count * 3 && n < count; i++) {
        const pick = hash01(i, salt)
        const disc = discs[pick < share[0] || discs.length < 3 ? 0 : pick < share[0] + share[1] ? 1 : 2]
        // Uniform over the disc: radius by the square root.
        const r = disc.r * Math.sqrt(hash01(i, salt + 1))
        const a = hash01(i, salt + 2) * Math.PI * 2
        const x = disc.x + r * Math.cos(a)
        const y = disc.y + r * Math.sin(a)
        if (x > pile.origin[0] - 2 && x < px1 + 2 && y > pile.origin[1] - 2 && y < py1 + 2) continue
        // Patchy: fewer where a slow noise is low.
        const patch = Math.sin(x * 0.045 + 1.3) * Math.sin(y * 0.038 + 0.4) + 0.6 * Math.sin((x + y) * 0.013)
        if (hash01(i, salt + 3) > 0.55 + 0.4 * patch) continue
        const z = topography.heightAt(x, y)
        if (!Number.isFinite(z)) continue
        const size = kind === 'tuft' ? 0.04 + 0.06 * hash01(i, salt + 4) : shrub ? 0.14 + 0.3 * hash01(i, salt + 4) ** 2 : 0.05 + 0.4 * hash01(i, salt + 4) ** 4
        scale.set(size * (0.8 + 0.5 * hash01(i, salt + 5)), size * (0.8 + 0.5 * hash01(i, salt + 6)), size * (kind === 'tuft' ? 2.0 + 1.6 * hash01(i, salt + 10) : shrub ? 0.7 : 0.6))
        q.setFromAxisAngle(up, hash01(i, salt + 7) * Math.PI * 2)
        position.set(x - origin[0], y - origin[1], z - origin[2] + scale.z * (kind === 'tuft' ? 0.45 : shrub ? 0.55 : 0.3))
        m.compose(position, q, scale)
        mesh.setMatrixAt(n, m)
        colour.set(tones[Math.floor(hash01(i, salt + 8) * tones.length) % tones.length]).multiplyScalar(0.8 + 0.4 * hash01(i, salt + 9))
        mesh.setColorAt(n, colour)
        n++
      }
      mesh.count = n
      mesh.frustumCulled = false
      mesh.castShadow = true
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
      return mesh
    }
    // A grass tuft: a narrow five-sided cone standing on its base (z up).
    const tuft = new ConeGeometry(0.5, 1, 5).rotateX(Math.PI / 2)
    return [
      make(STONES, 100, rockGeometry(), STONE_TONES, 'stone'),
      make(SHRUBS, 200, new IcosahedronGeometry(1, 1), SHRUB_TONES, 'shrub'),
      make(TUFTS, 300, tuft, TUFT_TONES, 'tuft'),
    ]
  }, [bundle, origin, topography])

  useEffect(() => {
    invalidate()
    return () => {
      for (const mesh of meshes ?? []) {
        mesh.geometry.dispose()
        ;(mesh.material as MeshStandardMaterial).dispose()
      }
    }
  }, [meshes, invalidate])
  if (!meshes) return null
  return (
    <>
      {meshes.map((mesh, i) => (
        <primitive key={i} object={mesh} />
      ))}
    </>
  )
}

/**
 * Two hydraulic excavators waiting in the safe zone (realism.equipmentSites: behind the
 * blast, outside the clearance radius), standing on the project topography. They are
 * scenery that gives the eye a scale: a generic 30-tonne-class machine built from boxes
 * and cylinders, about 10 m long with the boom, 3 m wide, 3.2 m to the cab roof. Not a
 * model of a particular product.
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import { DirectionalLight, PlaneGeometry, type Group } from 'three'
import type { LayerProps } from './common'
import { equipmentSites } from './realism'
import { SUN } from './terrain'
import { useTopography } from './topography'

const YELLOW = '#d99a16'
const YELLOW_DARK = '#b57f12'
const STEEL = '#2a2c2f'
const RUBBER = '#1c1d1f'
const GLASS = '#3a5261'
const ROD = '#b9bcc0'

function Track({ y }: { y: number }) {
  return (
    <group position={[0, y, 0.46]}>
      <mesh>
        <boxGeometry args={[3.7, 0.74, 0.92]} />
        <meshStandardMaterial color={RUBBER} roughness={0.95} />
      </mesh>
      {[-1.85, 1.85].map((x) => (
        <mesh key={x} position={[x, 0, 0]}>
          <cylinderGeometry args={[0.46, 0.46, 0.74, 20]} />
          <meshStandardMaterial color={RUBBER} roughness={0.95} />
        </mesh>
      ))}
      {/* Rollers and sprocket hubs, seen from the side. */}
      {[-1.85, -0.95, 0, 0.95, 1.85].map((x) => (
        <mesh key={x} position={[x, 0, -0.08]}>
          <cylinderGeometry args={[x === -1.85 || x === 1.85 ? 0.3 : 0.17, x === -1.85 || x === 1.85 ? 0.3 : 0.17, 0.78, 14]} />
          <meshStandardMaterial color={STEEL} roughness={0.7} metalness={0.3} />
        </mesh>
      ))}
    </group>
  )
}

/** One excavator: x forward, y left, z up, standing on z = 0. `swing` turns the upper structure. */
export function Excavator({ swing = 0 }: { swing?: number }) {
  const boom = (40 * Math.PI) / 180
  const stick = (112 * Math.PI) / 180
  return (
    <group>
      <Track y={1.22} />
      <Track y={-1.22} />
      <mesh position={[0, 0, 0.62]}>
        <boxGeometry args={[2.5, 2.0, 0.42]} />
        <meshStandardMaterial color={STEEL} roughness={0.8} />
      </mesh>
      <mesh position={[0, 0, 0.92]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.85, 0.85, 0.3, 24]} />
        <meshStandardMaterial color={STEEL} roughness={0.7} metalness={0.3} />
      </mesh>
      <group position={[0, 0, 1.05]} rotation={[0, 0, swing]}>
        {/* House, engine cover, counterweight. */}
        <mesh position={[-0.75, 0, 0.55]}>
          <boxGeometry args={[4.1, 2.75, 1.1]} />
          <meshStandardMaterial color={YELLOW} roughness={0.55} metalness={0.1} />
        </mesh>
        <mesh position={[-1.5, -0.1, 1.28]}>
          <boxGeometry args={[2.3, 2.3, 0.36]} />
          <meshStandardMaterial color={YELLOW_DARK} roughness={0.6} />
        </mesh>
        <mesh position={[-3.0, 0, 0.6]}>
          <boxGeometry args={[0.7, 2.6, 1.15]} />
          <meshStandardMaterial color={YELLOW_DARK} roughness={0.7} />
        </mesh>
        <mesh position={[-1.9, -0.9, 1.75]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.7, 10]} />
          <meshStandardMaterial color={STEEL} roughness={0.6} metalness={0.5} />
        </mesh>
        {/* Cab on the left, glass around its upper half. */}
        <mesh position={[0.75, 0.82, 1.45]}>
          <boxGeometry args={[1.55, 1.05, 1.75]} />
          <meshStandardMaterial color={YELLOW} roughness={0.55} metalness={0.1} />
        </mesh>
        <mesh position={[0.78, 0.82, 1.72]}>
          <boxGeometry args={[1.52, 1.09, 0.95]} />
          <meshStandardMaterial color={GLASS} roughness={0.15} metalness={0.7} />
        </mesh>
        <mesh position={[0.75, 0.82, 2.35]}>
          <boxGeometry args={[1.62, 1.12, 0.07]} />
          <meshStandardMaterial color={STEEL} roughness={0.7} />
        </mesh>
        {/* Boom, stick, bucket: a chain of joints turning about the y axis. */}
        <group position={[1.15, -0.25, 0.75]} rotation={[0, -boom, 0]}>
          <mesh position={[2.9, 0, 0]}>
            <boxGeometry args={[5.8, 0.46, 0.62]} />
            <meshStandardMaterial color={YELLOW} roughness={0.55} metalness={0.1} />
          </mesh>
          <mesh position={[1.9, 0, -0.52]} rotation={[0, 0.17, Math.PI / 2]}>
            <cylinderGeometry args={[0.09, 0.09, 2.6, 10]} />
            <meshStandardMaterial color={ROD} roughness={0.25} metalness={0.9} />
          </mesh>
          <mesh position={[4.0, 0, 0.5]} rotation={[0, -0.1, Math.PI / 2]}>
            <cylinderGeometry args={[0.08, 0.08, 2.6, 10]} />
            <meshStandardMaterial color={ROD} roughness={0.25} metalness={0.9} />
          </mesh>
          <group position={[5.8, 0, 0]} rotation={[0, stick, 0]}>
            <mesh position={[1.45, 0, 0]}>
              <boxGeometry args={[3.3, 0.36, 0.46]} />
              <meshStandardMaterial color={YELLOW} roughness={0.55} metalness={0.1} />
            </mesh>
            <group position={[3.0, 0, 0]} rotation={[0, 1.0, 0]}>
              <mesh position={[0.45, 0, 0.1]}>
                <boxGeometry args={[1.1, 1.15, 0.85]} />
                <meshStandardMaterial color={STEEL} roughness={0.6} metalness={0.4} />
              </mesh>
              {[-0.42, -0.14, 0.14, 0.42].map((y) => (
                <mesh key={y} position={[1.08, y, -0.22]}>
                  <boxGeometry args={[0.28, 0.1, 0.1]} />
                  <meshStandardMaterial color={ROD} roughness={0.4} metalness={0.8} />
                </mesh>
              ))}
            </group>
          </group>
        </group>
      </group>
    </group>
  )
}

/** Side of the square patch of ground that catches the machines' shadows, m, and its cell size. */
const PATCH_M = 80
const PATCH_CELL_M = 2

export function Equipment({ bundle, origin }: LayerProps) {
  const invalidate = useThree((s) => s.invalidate)
  const topography = useTopography(bundle, origin)
  const machines = useRef<Group>(null)
  const sites = useMemo(() => {
    if (!topography) return []
    return equipmentSites(bundle)
      .map((s) => ({ ...s, z: topography.heightAt(s.x, s.y) }))
      .filter((s) => Number.isFinite(s.z))
  }, [bundle, topography])

  // The shadows: a weak sun-direction light whose shadow map covers only this place, and a
  // patch of ground (the topography resampled, a few centimetres up) that shows nothing
  // but the shadow falling on it.
  const shadow = useMemo(() => {
    if (!topography || sites.length === 0) return null
    const mx = sites.reduce((a, s) => a + s.x, 0) / sites.length
    const my = sites.reduce((a, s) => a + s.y, 0) / sites.length
    const mz = sites.reduce((a, s) => a + s.z, 0) / sites.length
    const light = new DirectionalLight('#ffffff', 0.01)
    light.castShadow = true
    light.position.set(mx - origin[0] + SUN[0] * 120, my - origin[1] + SUN[1] * 120, mz - origin[2] + SUN[2] * 120)
    light.target.position.set(mx - origin[0], my - origin[1], mz - origin[2])
    light.shadow.mapSize.set(2048, 2048)
    const camera = light.shadow.camera
    camera.left = camera.bottom = -PATCH_M / 2
    camera.right = camera.top = PATCH_M / 2
    camera.near = 1
    camera.far = 260
    camera.updateProjectionMatrix()
    light.shadow.bias = -0.0005
    const n = Math.round(PATCH_M / PATCH_CELL_M)
    const geometry = new PlaneGeometry(PATCH_M, PATCH_M, n, n)
    const position = geometry.getAttribute('position')
    for (let i = 0; i < position.count; i++) {
      const z = topography.heightAt(mx + position.getX(i), my + position.getY(i))
      position.setZ(i, (Number.isFinite(z) ? z : mz) - mz + 0.08)
    }
    geometry.computeVertexNormals()
    return { light, geometry, centre: [mx - origin[0], my - origin[1], mz - origin[2]] as const }
  }, [topography, sites, origin])
  useEffect(() => {
    machines.current?.traverse((o) => {
      o.castShadow = true
    })
    invalidate()
    return () => {
      shadow?.geometry.dispose()
      shadow?.light.dispose()
    }
  }, [shadow, sites, invalidate])

  return (
    <>
      <group ref={machines}>
        {sites.map((s, i) => (
          <group key={i} position={[s.x - origin[0], s.y - origin[1], s.z - origin[2]]} rotation={[0, 0, s.heading]}>
            <Excavator swing={s.swing} />
          </group>
        ))}
      </group>
      {shadow && (
        <>
          <primitive object={shadow.light} />
          <primitive object={shadow.light.target} />
          <mesh geometry={shadow.geometry} position={shadow.centre} receiveShadow renderOrder={3}>
            <shadowMaterial transparent opacity={0.5} depthWrite={false} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
          </mesh>
        </>
      )}
    </>
  )
}

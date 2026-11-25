/**
 * Two hydraulic excavators waiting in the safe zone (realism.equipmentSites: behind the
 * blast, outside the clearance radius), standing on the project topography. They are
 * scenery that gives the eye a scale: a generic 30-tonne-class machine, about 10 m long
 * with the boom, 3.2 m wide, 3.1 m to the cab roof. Not a model of a particular product.
 *
 * The parts are side profiles (track loop, house, cab, gooseneck boom, stick, bucket)
 * extruded across the machine with bevelled edges, plus hydraulic rams between their
 * joints. Local axes: x forward, y left, z up, standing on z = 0.
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import { DirectionalLight, ExtrudeGeometry, Shape, type BufferGeometry, type Group } from 'three'
import type { LayerProps } from './common'
import { equipmentSites } from './realism'
import { SUN } from './terrain'
import { useTopography } from './topography'

const YELLOW = '#d6981a'
const YELLOW_DARK = '#a8760f'
const STEEL = '#33363a'
const RUBBER = '#1d1e20'
const GLASS = '#2f4654'
const ROD = '#c4c7cb'

type Xz = readonly [number, number]

/** A side profile (x forward, z up) extruded `width` across the machine, centred on y = 0. */
function profile(points: readonly Xz[], width: number, bevel = 0.04): BufferGeometry {
  const shape = new Shape()
  points.forEach(([x, z], i) => (i === 0 ? shape.moveTo(x, z) : shape.lineTo(x, z)))
  shape.closePath()
  const geometry = new ExtrudeGeometry(shape, {
    depth: Math.max(0.01, width - 2 * bevel),
    bevelEnabled: bevel > 0,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
    steps: 1,
  })
  // The shape lies in the geometry's xy plane and is extruded along +z: turn it upright.
  geometry.rotateX(Math.PI / 2)
  geometry.translate(0, (width - 2 * bevel) / 2, 0)
  return geometry
}

/** The loop of a track seen from the side: two half circles joined by straight runs. */
function trackLoop(length: number, height: number, inset = 0): Xz[] {
  const r = height / 2 - inset
  const half = length / 2 - height / 2
  const out: Xz[] = []
  for (let i = 0; i <= 12; i++) {
    const a = -Math.PI / 2 + (Math.PI * i) / 12
    out.push([half + r * Math.cos(a), height / 2 + r * Math.sin(a)])
  }
  for (let i = 0; i <= 12; i++) {
    const a = Math.PI / 2 + (Math.PI * i) / 12
    out.push([-half + r * Math.cos(a), height / 2 + r * Math.sin(a)])
  }
  return out
}

const rotate = ([x, z]: Xz, angle: number): Xz => [x * Math.cos(angle) - z * Math.sin(angle), x * Math.sin(angle) + z * Math.cos(angle)]
const add = (a: Xz, b: Xz): Xz => [a[0] + b[0], a[1] + b[1]]

/** A hydraulic ram between two points of the side view, at offset `y`: a barrel and a polished rod. */
function Ram({ from, to, y = 0, radius = 0.09 }: { from: Xz; to: Xz; y?: number; radius?: number }) {
  const length = Math.hypot(to[0] - from[0], to[1] - from[1])
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0])
  return (
    <group position={[from[0], y, from[1]]} rotation={[0, -angle, 0]}>
      <mesh position={[length * 0.29, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[radius, radius, length * 0.58, 14]} />
        <meshStandardMaterial color={STEEL} roughness={0.5} metalness={0.5} />
      </mesh>
      <mesh position={[length * 0.78, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[radius * 0.55, radius * 0.55, length * 0.44, 12]} />
        <meshStandardMaterial color={ROD} roughness={0.2} metalness={0.95} />
      </mesh>
    </group>
  )
}

const TRACK_LENGTH = 4.7
const TRACK_HEIGHT = 0.96
const TRACK_WIDTH = 0.62

interface Parts {
  track: BufferGeometry
  trackFrame: BufferGeometry
  house: BufferGeometry
  hood: BufferGeometry
  counterweight: BufferGeometry
  cab: BufferGeometry
  boom: BufferGeometry
  stick: BufferGeometry
  bucket: BufferGeometry
  bucketSide: BufferGeometry
}

function makeParts(): Parts {
  return {
    track: profile(trackLoop(TRACK_LENGTH, TRACK_HEIGHT), TRACK_WIDTH, 0.03),
    trackFrame: profile(
      trackLoop(TRACK_LENGTH - 0.5, TRACK_HEIGHT - 0.34).map(([x, z]) => [x, z + 0.17] as const),
      TRACK_WIDTH + 0.02,
      0,
    ),
    // Upper structure: deck with a sloped engine cover.
    house: profile([[-2.5, 0], [-2.5, 1.0], [-0.1, 1.0], [0.1, 0.72], [1.85, 0.72], [2.0, 0.5], [2.0, 0]], 2.9, 0.07),
    hood: profile([[-2.45, 1.0], [-2.4, 1.42], [-0.6, 1.42], [-0.15, 1.05], [-0.1, 1.0]], 2.3, 0.06),
    counterweight: profile([[-3.3, 0.2], [-3.38, 0.55], [-3.3, 1.2], [-2.5, 1.32], [-2.5, 0.05]], 2.86, 0.12),
    cab: profile([[0.1, 0.72], [0.1, 2.3], [0.32, 2.42], [1.45, 2.42], [1.82, 1.5], [1.9, 0.72]], 1.02, 0.05),
    // Gooseneck boom: pivot at the origin, tip at (6, 0).
    boom: profile(
      [[-0.2, -0.3], [-0.25, 0.25], [0.3, 0.42], [2.5, 1.12], [3.2, 1.16], [5.8, 0.26], [6.15, 0.12], [6.15, -0.14], [5.8, -0.22], [3.25, 0.5], [2.6, 0.46], [0.4, -0.34]],
      0.5,
      0.04,
    ),
    stick: profile([[-0.75, 0.16], [-0.6, 0.36], [0.35, 0.42], [3.05, 0.17], [3.2, 0], [3.05, -0.16], [0.1, -0.3], [-0.6, -0.14]], 0.36, 0.035),
    // Bucket hanging from the stick tip, opening forward: the curved shell and its two side plates.
    bucket: profile(
      [[-0.1, 0.22], [-0.42, -0.25], [-0.36, -0.8], [0.05, -1.18], [0.62, -1.3], [1.1, -1.14], [1.36, -0.86], [1.2, -0.84], [0.62, -1.12], [0.15, -1.02], [-0.16, -0.7], [-0.2, -0.26], [0.32, 0.2]],
      1.2,
      0.03,
    ),
    bucketSide: profile([[-0.1, 0.22], [-0.42, -0.25], [-0.36, -0.8], [0.05, -1.18], [0.62, -1.3], [1.1, -1.14], [1.36, -0.86], [0.32, 0.2]], 0.06, 0),
  }
}

function Track({ y, parts }: { y: number; parts: Parts }) {
  const grousers = useMemo(() => {
    const out: { x: number; z: number; a: number }[] = []
    const straight = TRACK_LENGTH - TRACK_HEIGHT
    for (let i = 0; i < 17; i++) {
      const x = -straight / 2 + (straight * i) / 16
      out.push({ x, z: TRACK_HEIGHT, a: 0 }, { x, z: 0, a: 0 })
    }
    for (const end of [-1, 1]) {
      for (let i = 1; i < 6; i++) {
        const a = -Math.PI / 2 + (Math.PI * i) / 6
        out.push({
          x: end * (straight / 2 + (TRACK_HEIGHT / 2) * Math.cos(a)),
          z: TRACK_HEIGHT / 2 + (TRACK_HEIGHT / 2) * Math.sin(a),
          a: end * a + Math.PI / 2,
        })
      }
    }
    return out
  }, [])
  return (
    <group position={[0, y, 0]}>
      <mesh geometry={parts.track}>
        <meshStandardMaterial color={RUBBER} roughness={0.9} metalness={0.2} />
      </mesh>
      {/* Track frame and rollers inside the loop. */}
      <mesh geometry={parts.trackFrame}>
        <meshStandardMaterial color={STEEL} roughness={0.75} metalness={0.3} />
      </mesh>
      {[-1.9, -1.2, -0.6, 0, 0.6, 1.2, 1.9].map((x) => {
        const end = x === -1.9 || x === 1.9
        return (
          <mesh key={x} position={[x, 0, end ? TRACK_HEIGHT / 2 : 0.22]}>
            <cylinderGeometry args={[end ? 0.36 : 0.15, end ? 0.36 : 0.15, TRACK_WIDTH + 0.06, 16]} />
            <meshStandardMaterial color={YELLOW_DARK} roughness={0.6} metalness={0.3} />
          </mesh>
        )
      })}
      {/* Grouser bars across the shoes. */}
      {grousers.map((g, i) => (
        <mesh key={i} position={[g.x, 0, g.z]} rotation={[0, -g.a, 0]}>
          <boxGeometry args={[0.07, TRACK_WIDTH + 0.04, 0.09]} />
          <meshStandardMaterial color={RUBBER} roughness={0.85} metalness={0.3} />
        </mesh>
      ))}
    </group>
  )
}

/** One excavator. `swing` turns the upper structure against the tracks. */
export function Excavator({ swing = 0, parts }: { swing?: number; parts: Parts }) {
  // Joint angles of the side view, counter-clockwise from forward.
  const boomAngle = (40 * Math.PI) / 180
  const stickAngle = boomAngle - (118 * Math.PI) / 180
  const bucketAngle = stickAngle - (55 * Math.PI) / 180
  const boomPivot: Xz = [1.0, 0.95]
  const stickPivot = add(boomPivot, rotate([6.0, 0], boomAngle))
  const bucketPivot = add(stickPivot, rotate([3.1, 0], stickAngle))
  const onBoom = (p: Xz) => add(boomPivot, rotate(p, boomAngle))
  const onStick = (p: Xz) => add(stickPivot, rotate(p, stickAngle))
  return (
    <group>
      <Track y={1.28} parts={parts} />
      <Track y={-1.28} parts={parts} />
      {/* Car body between the tracks and the slew ring. */}
      <mesh position={[0, 0, 0.56]}>
        <boxGeometry args={[2.3, 2.0, 0.5]} />
        <meshStandardMaterial color={STEEL} roughness={0.8} metalness={0.3} />
      </mesh>
      <mesh position={[0, 0, 0.9]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.9, 0.95, 0.24, 28]} />
        <meshStandardMaterial color={STEEL} roughness={0.6} metalness={0.5} />
      </mesh>
      <group position={[0, 0, 1.02]} rotation={[0, 0, swing]}>
        <mesh geometry={parts.house}>
          <meshStandardMaterial color={YELLOW} roughness={0.5} metalness={0.15} />
        </mesh>
        <mesh geometry={parts.hood} position={[0, -0.25, 0]}>
          <meshStandardMaterial color={YELLOW_DARK} roughness={0.6} metalness={0.15} />
        </mesh>
        <mesh geometry={parts.counterweight}>
          <meshStandardMaterial color={STEEL} roughness={0.7} metalness={0.3} />
        </mesh>
        {/* Exhaust stack, air intake, handrail. */}
        <mesh position={[-1.75, -0.95, 1.75]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.72, 12]} />
          <meshStandardMaterial color={STEEL} roughness={0.5} metalness={0.7} />
        </mesh>
        <mesh position={[-1.1, 0.55, 1.56]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.13, 0.13, 0.3, 14]} />
          <meshStandardMaterial color={STEEL} roughness={0.6} metalness={0.5} />
        </mesh>
        {[-2.3, -1.5, -0.7, 0.05].map((x) => (
          <mesh key={x} position={[x, -1.38, 1.3]}>
            <boxGeometry args={[0.04, 0.04, 0.62]} />
            <meshStandardMaterial color={STEEL} roughness={0.5} metalness={0.6} />
          </mesh>
        ))}
        <mesh position={[-1.125, -1.38, 1.6]}>
          <boxGeometry args={[2.4, 0.04, 0.04]} />
          <meshStandardMaterial color={STEEL} roughness={0.5} metalness={0.6} />
        </mesh>
        {/* Cab on the left: frame, windscreen, side and rear glass, roof lights. */}
        <group position={[0, 0.9, 0]}>
          <mesh geometry={parts.cab}>
            <meshStandardMaterial color={YELLOW} roughness={0.5} metalness={0.15} />
          </mesh>
          <mesh position={[1.72, 0, 1.52]} rotation={[0, -0.39, 0]}>
            <boxGeometry args={[0.04, 0.86, 1.45]} />
            <meshStandardMaterial color={GLASS} roughness={0.08} metalness={0.85} />
          </mesh>
          {[-0.52, 0.52].map((y) => (
            <mesh key={y} position={[0.82, y, 1.7]}>
              <boxGeometry args={[1.25, 0.03, 1.05]} />
              <meshStandardMaterial color={GLASS} roughness={0.08} metalness={0.85} />
            </mesh>
          ))}
          <mesh position={[0.09, 0, 1.75]}>
            <boxGeometry args={[0.03, 0.8, 0.8]} />
            <meshStandardMaterial color={GLASS} roughness={0.08} metalness={0.85} />
          </mesh>
          {[-0.3, 0.3].map((y) => (
            <mesh key={y} position={[1.4, y, 2.5]}>
              <boxGeometry args={[0.12, 0.16, 0.1]} />
              <meshStandardMaterial color="#e9eef2" roughness={0.3} metalness={0.4} />
            </mesh>
          ))}
        </group>
        {/* Front attachment, on the right of the cab. */}
        <group position={[0, -0.3, 0]}>
          <group position={[boomPivot[0], 0, boomPivot[1]]} rotation={[0, -boomAngle, 0]}>
            <mesh geometry={parts.boom}>
              <meshStandardMaterial color={YELLOW} roughness={0.5} metalness={0.15} />
            </mesh>
          </group>
          <group position={[stickPivot[0], 0, stickPivot[1]]} rotation={[0, -stickAngle, 0]}>
            <mesh geometry={parts.stick}>
              <meshStandardMaterial color={YELLOW} roughness={0.5} metalness={0.15} />
            </mesh>
          </group>
          <group position={[bucketPivot[0], 0, bucketPivot[1]]} rotation={[0, -bucketAngle, 0]}>
            <mesh geometry={parts.bucket}>
              <meshStandardMaterial color={STEEL} roughness={0.55} metalness={0.6} />
            </mesh>
            {[-0.6, 0.6].map((y) => (
              <mesh key={y} geometry={parts.bucketSide} position={[0, y, 0]}>
                <meshStandardMaterial color={STEEL} roughness={0.55} metalness={0.6} />
              </mesh>
            ))}
            {[-0.48, -0.24, 0, 0.24, 0.48].map((y) => (
              <mesh key={y} position={[1.42, y, -0.9]} rotation={[0, 0.75, 0]}>
                <boxGeometry args={[0.3, 0.09, 0.07]} />
                <meshStandardMaterial color={ROD} roughness={0.35} metalness={0.9} />
              </mesh>
            ))}
          </group>
          {/* Rams: two under the boom, one on its back to the stick, one along the stick to the bucket. */}
          <Ram from={[1.75, 0.42]} to={onBoom([2.55, 0.5])} y={0.34} radius={0.1} />
          <Ram from={[1.75, 0.42]} to={onBoom([2.55, 0.5])} y={-0.34} radius={0.1} />
          <Ram from={onBoom([3.1, 1.28])} to={onStick([-0.68, 0.3])} radius={0.1} />
          <Ram from={onStick([0.35, 0.55])} to={onStick([2.65, 0.42])} radius={0.08} />
          {/* Pins at the joints. */}
          {[boomPivot, stickPivot, bucketPivot].map((p, i) => (
            <mesh key={i} position={[p[0], 0, p[1]]}>
              <cylinderGeometry args={[0.1, 0.1, 0.62, 12]} />
              <meshStandardMaterial color={STEEL} roughness={0.4} metalness={0.8} />
            </mesh>
          ))}
        </group>
      </group>
    </group>
  )
}

/** Half side of the square the machines' shadow map covers, m. */
const SHADOW_HALF_M = 40

export function Equipment({ bundle, origin }: LayerProps) {
  const invalidate = useThree((s) => s.invalidate)
  const topography = useTopography(bundle, origin)
  const machines = useRef<Group>(null)
  const parts = useMemo(makeParts, [])
  useEffect(
    () => () => {
      for (const g of Object.values(parts)) (g as BufferGeometry).dispose()
    },
    [parts],
  )
  const sites = useMemo(() => {
    if (!topography) return []
    return equipmentSites(bundle)
      .map((s) => ({ ...s, z: topography.heightAt(s.x, s.y) }))
      .filter((s) => Number.isFinite(s.z))
  }, [bundle, topography])

  // A weak light from the sun's direction whose shadow map covers only this place; the
  // terrain shows the shadow (Ground.tsx draws a shadow-only copy of itself).
  const light = useMemo(() => {
    if (sites.length === 0) return null
    const mx = sites.reduce((a, s) => a + s.x, 0) / sites.length - origin[0]
    const my = sites.reduce((a, s) => a + s.y, 0) / sites.length - origin[1]
    const mz = sites.reduce((a, s) => a + s.z, 0) / sites.length - origin[2]
    const out = new DirectionalLight('#ffffff', 0.01)
    out.castShadow = true
    out.position.set(mx + SUN[0] * 150, my + SUN[1] * 150, mz + SUN[2] * 150)
    out.target.position.set(mx, my, mz)
    out.shadow.mapSize.set(2048, 2048)
    const camera = out.shadow.camera
    camera.left = camera.bottom = -SHADOW_HALF_M
    camera.right = camera.top = SHADOW_HALF_M
    camera.near = 1
    camera.far = 320
    camera.updateProjectionMatrix()
    out.shadow.bias = -0.0004
    out.shadow.normalBias = 0.05
    return out
  }, [sites, origin])
  useEffect(() => {
    machines.current?.traverse((o) => {
      o.castShadow = true
    })
    invalidate()
    return () => light?.dispose()
  }, [light, sites, invalidate])

  return (
    <>
      <group ref={machines}>
        {sites.map((s, i) => (
          <group key={i} position={[s.x - origin[0], s.y - origin[1], s.z - origin[2]]} rotation={[0, 0, s.heading]}>
            <Excavator swing={s.swing} parts={parts} />
          </group>
        ))}
      </group>
      {light && (
        <>
          <primitive object={light} />
          <primitive object={light.target} />
        </>
      )}
    </>
  )
}

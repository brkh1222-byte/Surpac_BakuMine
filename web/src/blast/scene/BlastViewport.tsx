/**
 * The Blast BakuMine 3D view: its own canvas, camera and clock. Every layer is a pure
 * function of the bundle and the player time `t` (state.ts); frames are drawn on demand,
 * i.e. while playing, scrubbing or orbiting.
 */
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { PerspectiveCamera, Vector3 } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { SCENE_COLORS } from '../../scene/theme'
import { useViewer } from '../../state/store'
import { advance, autoSpeed, fitFrame } from '../math'
import { useBlast } from '../state'
import type { BlastBundle } from '../types'
import { Legends } from '../ui/Legends'
import { blastView, bundleOrigin, type LayerProps } from './common'
import { Damage } from './Damage'
import { Dust } from './Dust'
import { BlastShadows, BlastSound, Flashes, Shake, StayAboveGround } from './Effects'
import { Equipment } from './Equipment'
import { ContextGround, Muckpile, TerrainDetail, Topography } from './Ground'
import { Holes } from './Holes'
import { PpvMap } from './PpvMap'
import { AirblastRings, FlyrockRings, Monitors } from './Rings'
import { Throw } from './Throw'
import { blastCentre, equipmentSites } from './realism'
import { Scatter } from './Scatter'
import { Sky, SUN } from './terrain'
import { Waves } from './Waves'

const FOV = 35

/** What "fit to blast" frames, scene coordinates: the pattern and the muckpile area in front of it. */
function blastFrame(bundle: BlastBundle, origin: readonly number[]): { centre: Vector3; radius: number; direction: Vector3 } {
  const holes = bundle.timeline.holes
  const polygon = Array.isArray(bundle.design.polygon) && bundle.design.polygon.length >= 3 ? bundle.design.polygon : null
  const outline: (readonly number[])[] = polygon ?? holes.map((h) => h.collar)
  const points = outline.length > 0 ? outline : [bundle.muckpile.origin]
  const azimuth = typeof bundle.design.face_azimuth === 'number' ? bundle.design.face_azimuth : 180
  const frame = fitFrame(points, azimuth)
  // Look at mid-bench height.
  let z = 0
  for (const h of holes) z += (h.collar[2] + h.toe[2]) / 2
  z = holes.length > 0 ? z / holes.length : origin[2]
  return {
    centre: new Vector3(frame.centre[0] - origin[0], frame.centre[1] - origin[1], z - origin[2]),
    radius: frame.radius,
    direction: new Vector3(...frame.direction),
  }
}

/** Orbit controls (z up) and "fit to blast": from in front of the face, looking back at it, 35 degrees down. */
function Camera({ bundle, origin }: LayerProps) {
  const camera = useThree((s) => s.camera) as PerspectiveCamera
  const gl = useThree((s) => s.gl)
  const invalidate = useThree((s) => s.invalidate)

  useEffect(() => {
    camera.up.set(0, 0, 1)
    const controls = new OrbitControls(camera, gl.domElement)
    controls.enableDamping = false
    controls.zoomToCursor = true
    controls.screenSpacePanning = true
    controls.minDistance = 1
    controls.maxDistance = 2e5
    controls.addEventListener('change', () => invalidate())
    const fit = () => {
      const { centre, radius, direction } = blastFrame(bundle, origin)
      const distance = radius / Math.sin((FOV * Math.PI) / 360)
      camera.position.copy(centre).addScaledVector(direction, distance)
      camera.near = Math.max(0.2, distance / 2000)
      camera.far = Math.max(5e4, distance * 200)
      camera.updateProjectionMatrix()
      controls.target.copy(centre)
      controls.update()
      blastView.fitDistance = distance
      invalidate()
    }
    // From behind the excavators, at eye height above them, looking at the blast.
    const safeView = () => {
      const sites = equipmentSites(bundle)
      const ground = blastView.groundAt
      if (sites.length === 0 || !ground) return
      const [bx, by, bz] = blastCentre(bundle)
      const mx = sites.reduce((a, s) => a + s.x, 0) / sites.length
      const my = sites.reduce((a, s) => a + s.y, 0) / sites.length
      const d = Math.hypot(mx - bx, my - by) || 1
      const x = mx + ((mx - bx) / d) * 30
      const y = my + ((my - by) / d) * 30
      const z = ground(x, y)
      if (!Number.isFinite(z)) return
      camera.position.set(x - origin[0], y - origin[1], z + 6.5 - origin[2])
      camera.near = 0.5
      camera.far = Math.max(5e4, d * 200)
      camera.updateProjectionMatrix()
      controls.target.set(bx - origin[0], by - origin[1], bz - origin[2])
      controls.update()
      invalidate()
    }
    blastView.fit = fit
    blastView.safeView = safeView
    fit()
    return () => {
      controls.dispose()
      if (blastView.fit === fit) blastView.fit = null
      if (blastView.safeView === safeView) blastView.safeView = null
    }
  }, [camera, gl, invalidate, bundle, origin])
  return null
}

/** Advances the player clock on every drawn frame and asks for the next one while playing. */
function Clock() {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => useBlast.subscribe(() => invalidate()), [invalidate])
  useEffect(() => useViewer.subscribe(() => invalidate()), [invalidate])
  useFrame((_, delta) => {
    const s = useBlast.getState()
    if (!s.playing) return
    const speed = s.autoSpeed ? autoSpeed(s.t, s.lastFire) : s.speed
    // A long gap (hidden tab, first frame after a pause) must not jump the blast.
    const next = advance(s.t, Math.min(delta, 0.1), speed, s.tEnd, s.loop)
    useBlast.setState({ t: next.t, playing: next.playing })
  }, -1)
  return null
}

/** Diagram lighting, or with the satellite picture a sun where the terrain shading has it and sky light. */
function Lights() {
  const site = useBlast((s) => s.layers.satellite && s.source === 'api' && s.imagery !== null)
  if (!site) {
    return (
      <>
        <ambientLight intensity={1.0} />
        <directionalLight position={[-1, -0.15, 0.75]} intensity={2.3} />
        <directionalLight position={[0.7, 0.9, 0.25]} intensity={0.55} />
      </>
    )
  }
  return (
    <>
      <hemisphereLight args={['#dfe9f5', '#8a7a62', 1.25]} position={[0, 0, 1]} />
      <directionalLight position={[SUN[0], SUN[1], SUN[2]]} intensity={2.6} color="#fff3df" />
    </>
  )
}

function Layers({ bundle, origin }: LayerProps) {
  const layers = useBlast((s) => s.layers)
  const source = useBlast((s) => s.source)
  const topography = layers.topography && source === 'api'
  const realistic = useBlast((s) => s.layers.satellite && s.source === 'api' && s.imagery !== null)
  // Layers fill their buffers in effects: draw once more after every change of the set.
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    invalidate()
    const id = requestAnimationFrame(() => invalidate())
    return () => cancelAnimationFrame(id)
  })
  return (
    <>
      {realistic && <Sky />}
      {topography && <Topography bundle={bundle} origin={origin} />}
      {topography && realistic && <TerrainDetail bundle={bundle} origin={origin} />}
      {layers.ground && !topography && <ContextGround bundle={bundle} origin={origin} />}
      {layers.damage && <Damage bundle={bundle} origin={origin} />}
      {layers.holes && <Holes bundle={bundle} origin={origin} />}
      {layers.throw && <Throw bundle={bundle} origin={origin} />}
      {topography && layers.equipment && <Equipment bundle={bundle} origin={origin} />}
      {topography && layers.scatter && <Scatter bundle={bundle} origin={origin} />}
      {layers.dust && layers.throw && <Dust bundle={bundle} origin={origin} />}
      {realistic && <BlastShadows bundle={bundle} origin={origin} />}
      {realistic && <StayAboveGround origin={origin} />}
      {realistic && layers.dust && <Flashes bundle={bundle} origin={origin} />}
      {realistic && layers.effects && <Shake bundle={bundle} origin={origin} />}
      {realistic && layers.effects && <BlastSound bundle={bundle} origin={origin} />}
      {layers.ground && <Muckpile bundle={bundle} origin={origin} />}
      {layers.ppv && <PpvMap bundle={bundle} origin={origin} />}
      {layers.waves && <Waves bundle={bundle} origin={origin} />}
      {layers.flyrock && <FlyrockRings bundle={bundle} origin={origin} />}
      {layers.airblast && <AirblastRings bundle={bundle} origin={origin} />}
      {layers.monitors && <Monitors bundle={bundle} origin={origin} />}
    </>
  )
}

export function BlastViewport() {
  const theme = useViewer((s) => s.theme)
  const bundle = useBlast((s) => s.bundle)
  const source = useBlast((s) => s.source)
  const reason = useBlast((s) => s.reason)
  const status = useBlast((s) => s.status)
  const error = useBlast((s) => s.error)
  const origin = useMemo(() => (bundle ? bundleOrigin(bundle) : null), [bundle])

  return (
    <div className="blast-viewport" data-testid="blast-viewport">
      <Canvas flat shadows frameloop="demand" dpr={[1, 2]} gl={{ antialias: true, localClippingEnabled: true }} camera={{ fov: FOV, near: 0.5, far: 1e5, up: [0, 0, 1], position: [80, -120, 80] }}>
        <color attach="background" args={[SCENE_COLORS[theme].background]} />
        <Lights />
        <Clock />
        {bundle && origin && (
          <>
            <Camera bundle={bundle} origin={origin} />
            <Layers bundle={bundle} origin={origin} />
          </>
        )}
      </Canvas>
      <div
        className="blast-labels"
        aria-hidden="true"
        ref={(el) => {
          blastView.labels = el
        }}
      />
      <div className="blast-topbar">
        <button type="button" data-testid="blast-fit" title="Frame the blast" onClick={() => blastView.fit?.()}>
          Fit to blast
        </button>
        {source === 'api' && (
          <button type="button" data-testid="blast-safe-view" title="Stand behind the excavators in the safe zone and look at the blast" onClick={() => blastView.safeView?.()}>
            Safe-zone view
          </button>
        )}
        {status === 'loading' && (
          <span className="blast-badge" data-testid="blast-loading">
            <i className="blast-spinner" aria-hidden="true" />
            Simulating…
          </span>
        )}
        {error && <span className="blast-badge blast-badge-error" title={error}>Simulation failed · see Design</span>}
        {status === 'ready' && source === 'demo' && (
          <span className="blast-badge blast-badge-demo" data-testid="blast-demo-badge" title={reason ?? undefined}>
            Demo data · not a simulation result
          </span>
        )}
      </div>
      {bundle && <Legends bundle={bundle} />}
    </div>
  )
}

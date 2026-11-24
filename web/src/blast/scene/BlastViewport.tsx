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
import { advance, autoSpeed } from '../math'
import { useBlast } from '../state'
import type { BlastBundle } from '../types'
import { Legends } from '../ui/Legends'
import { blastView, bundleOrigin, type LayerProps } from './common'
import { Damage } from './Damage'
import { ContextGround, Muckpile, Topography } from './Ground'
import { Holes } from './Holes'
import { PpvMap } from './PpvMap'
import { AirblastRings, FlyrockRings, Monitors } from './Rings'
import { Throw } from './Throw'
import { Waves } from './Waves'

const FOV = 35

/** Sphere around the holes and the muckpile grid, scene coordinates. */
function blastSphere(bundle: BlastBundle, origin: readonly number[]): { centre: Vector3; radius: number } {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  const add = (p: readonly number[]) => {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k]
      if (p[k] > max[k]) max[k] = p[k]
    }
  }
  for (const h of bundle.timeline.holes) {
    add(h.collar)
    add(h.toe)
  }
  const m = bundle.muckpile
  const z = Number.isFinite(min[2]) ? (min[2] + max[2]) / 2 : 0
  add([m.origin[0], m.origin[1], z])
  add([m.origin[0] + (m.shape[0] - 1) * m.spacing, m.origin[1] + (m.shape[1] - 1) * m.spacing, z])
  const centre = new Vector3((min[0] + max[0]) / 2 - origin[0], (min[1] + max[1]) / 2 - origin[1], (min[2] + max[2]) / 2 - origin[2])
  return { centre, radius: Math.max(5, Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2) }
}

/** Orbit controls (z up) and "fit to blast": in front of the face, off to one side, looking down. */
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
      const { centre, radius } = blastSphere(bundle, origin)
      const az = (((bundle.design.face_azimuth as number | undefined) ?? 180) * Math.PI) / 180
      const f = new Vector3(Math.sin(az), Math.cos(az), 0)
      const u = new Vector3(Math.cos(az), -Math.sin(az), 0)
      const dir = f.multiplyScalar(1).addScaledVector(u, 0.9).add(new Vector3(0, 0, 0.75)).normalize()
      const distance = (radius / Math.sin((FOV * Math.PI) / 360)) * 1.05
      camera.position.copy(centre).addScaledVector(dir, distance)
      camera.near = Math.max(0.2, distance / 2000)
      camera.far = Math.max(5e4, distance * 200)
      camera.updateProjectionMatrix()
      controls.target.copy(centre)
      controls.update()
      invalidate()
    }
    blastView.fit = fit
    fit()
    return () => {
      controls.dispose()
      if (blastView.fit === fit) blastView.fit = null
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

function Layers({ bundle, origin }: LayerProps) {
  const layers = useBlast((s) => s.layers)
  const source = useBlast((s) => s.source)
  const topography = layers.topography && source === 'api'
  // Layers fill their buffers in effects: draw once more after every change of the set.
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    invalidate()
    const id = requestAnimationFrame(() => invalidate())
    return () => cancelAnimationFrame(id)
  })
  return (
    <>
      {topography && <Topography bundle={bundle} origin={origin} />}
      {layers.ground && !topography && <ContextGround bundle={bundle} origin={origin} />}
      {layers.damage && <Damage bundle={bundle} origin={origin} />}
      {layers.holes && <Holes bundle={bundle} origin={origin} />}
      {layers.throw && <Throw bundle={bundle} origin={origin} />}
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
  const origin = useMemo(() => (bundle ? bundleOrigin(bundle) : null), [bundle])

  return (
    <div className="blast-viewport" data-testid="blast-viewport">
      <Canvas flat frameloop="demand" dpr={[1, 2]} gl={{ antialias: true }} camera={{ fov: FOV, near: 0.5, far: 1e5, up: [0, 0, 1], position: [80, -120, 80] }}>
        <color attach="background" args={[SCENE_COLORS[theme].background]} />
        <ambientLight intensity={1.0} />
        <directionalLight position={[-1, -0.15, 0.75]} intensity={2.3} />
        <directionalLight position={[0.7, 0.9, 0.25]} intensity={0.55} />
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
        {status === 'loading' && <span className="blast-badge">Loading…</span>}
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

/**
 * What makes the site view feel like standing there, beyond the pictures of rock and dust:
 *
 * - `BlastShadows`: the sun's shadows around the blast (fragments in the air and on the
 *   pile, the bench walls, stones and shrubs). A weak light from the sun's direction owns
 *   the shadow map; `ShadowCatcher` is a copy of a ground mesh that shows only the shadow
 *   falling on it, so the terrain's own picture and shading stay as they are.
 * - `Flashes`: a short glow at each collar when its hole fires.
 * - `Shake`: the camera trembles while the ground wave of the firing holes passes it.
 * - `useBlastSound`: a low rumble when the sound of the blast reaches the camera.
 *
 * Arrival times follow from distance and wave speed (the P-wave speed of the bundle for
 * the tremble, 340 m/s for the sound). The strength of the glow, the tremble and the
 * rumble is chosen for effect: a real camera at this distance moves by millimetres.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import {
  AdditiveBlending, DirectionalLight, InstancedBufferAttribute, InstancedBufferGeometry, Mesh, PlaneGeometry, ShaderMaterial, Vector3,
  type BufferGeometry, type Plane,
} from 'three'
import { useBlast } from '../state'
import type { BlastBundle } from '../types'
import { blastView, type LayerProps } from './common'
import { blastCentre, shakeEnvelope, SOUND_SPEED_M_S } from './realism'
import { SUN } from './terrain'

/** Half side of the square around the blast that the shadow map covers, m. */
const SHADOW_HALF_M = 230

export function BlastShadows({ bundle, origin }: LayerProps) {
  const invalidate = useThree((s) => s.invalidate)
  const light = useMemo(() => {
    const [bx, by, bz] = blastCentre(bundle)
    const x = bx - origin[0]
    const y = by - origin[1]
    const z = bz - origin[2]
    const out = new DirectionalLight('#ffffff', 0.01)
    out.castShadow = true
    out.position.set(x + SUN[0] * 600, y + SUN[1] * 600, z + SUN[2] * 600)
    out.target.position.set(x, y, z)
    out.shadow.mapSize.set(4096, 4096)
    const camera = out.shadow.camera
    camera.left = camera.bottom = -SHADOW_HALF_M
    camera.right = camera.top = SHADOW_HALF_M
    camera.near = 10
    camera.far = 1300
    camera.updateProjectionMatrix()
    out.shadow.bias = -0.0006
    out.shadow.normalBias = 0.35
    return out
  }, [bundle, origin])
  useEffect(() => {
    invalidate()
    return () => light.dispose()
  }, [light, invalidate])
  return (
    <>
      <primitive object={light} />
      <primitive object={light.target} />
    </>
  )
}

/** A copy of a ground mesh that draws only the shadows falling on it. */
export function ShadowCatcher({ geometry, planes }: { geometry: BufferGeometry; planes?: Plane[] }) {
  return (
    <mesh geometry={geometry} receiveShadow renderOrder={4} frustumCulled={false}>
      <shadowMaterial
        transparent
        opacity={0.42}
        depthWrite={false}
        polygonOffset
        polygonOffsetFactor={-1}
        polygonOffsetUnits={-1}
        clippingPlanes={planes ?? null}
        clipIntersection={Boolean(planes && planes.length > 0)}
      />
    </mesh>
  )
}

const FLASH_VERTEX = /* glsl */ `
attribute vec3 aCentre;
attribute float aBirth;
uniform float uTime;
varying vec2 vCorner;
varying float vAge;
void main() {
  float dt = uTime - aBirth;
  vCorner = position.xy * 2.0;
  vAge = dt / 110.0;
  if (dt <= 0.0 || dt >= 110.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec4 mv = modelViewMatrix * vec4(aCentre, 1.0);
  mv.xy += position.xy * 2.0 * (1.6 + 5.0 * vAge);
  gl_Position = projectionMatrix * mv;
}
`
const FLASH_FRAGMENT = /* glsl */ `
varying vec2 vCorner;
varying float vAge;
void main() {
  float r = length(vCorner);
  if (r >= 1.0) discard;
  float glow = pow(1.0 - r, 2.2) * (1.0 - vAge) * (1.0 - vAge);
  gl_FragColor = vec4(vec3(1.0, 0.82, 0.55) * glow * 1.6, glow);
  #include <colorspace_fragment>
}
`

export function Flashes({ bundle, origin }: LayerProps) {
  const mesh = useMemo(() => {
    const holes = bundle.timeline.holes
    const geometry = new InstancedBufferGeometry()
    const quad = new PlaneGeometry(1, 1)
    geometry.setAttribute('position', quad.getAttribute('position'))
    geometry.setIndex(quad.getIndex())
    const centre = new Float32Array(holes.length * 3)
    const birth = new Float32Array(holes.length)
    holes.forEach((h, i) => {
      centre.set([h.collar[0] - origin[0], h.collar[1] - origin[1], h.collar[2] - origin[2] + 0.6], i * 3)
      birth[i] = h.t_fire_ms
    })
    geometry.setAttribute('aCentre', new InstancedBufferAttribute(centre, 3))
    geometry.setAttribute('aBirth', new InstancedBufferAttribute(birth, 1))
    geometry.instanceCount = holes.length
    const material = new ShaderMaterial({
      vertexShader: FLASH_VERTEX,
      fragmentShader: FLASH_FRAGMENT,
      uniforms: { uTime: { value: 0 } },
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
    })
    const out = new Mesh(geometry, material)
    out.frustumCulled = false
    out.renderOrder = 7
    quad.dispose()
    return out
  }, [bundle, origin])
  useEffect(
    () => () => {
      mesh.geometry.dispose()
      ;(mesh.material as ShaderMaterial).dispose()
    },
    [mesh],
  )
  useFrame(() => {
    ;(mesh.material as ShaderMaterial).uniforms.uTime.value = useBlast.getState().t
  })
  return <primitive object={mesh} />
}

/** Largest tremble of the camera as a fraction of its distance to what it looks at. */
const SHAKE_VIEW = 0.0016

export function Shake({ bundle, origin }: LayerProps) {
  const applied = useRef(new Vector3())
  const data = useMemo(() => {
    const [bx, by, bz] = blastCentre(bundle)
    return { centre: new Vector3(bx - origin[0], by - origin[1], bz - origin[2]), fires: bundle.timeline.holes.map((h) => h.t_fire_ms) }
  }, [bundle, origin])
  // Before the frame is drawn: take the last offset off, put the new one on. The controls
  // only move the camera on input, so the offset never accumulates.
  useFrame(({ camera }) => {
    camera.position.sub(applied.current)
    const { t } = useBlast.getState()
    const distance = camera.position.distanceTo(data.centre)
    const strength = shakeEnvelope(data.fires, t, distance, bundle.waves.cp_m_s)
    if (strength <= 0) {
      applied.current.set(0, 0, 0)
      return
    }
    const a = SHAKE_VIEW * distance * strength
    applied.current.set(a * Math.sin(t * 0.61), a * Math.sin(t * 0.83 + 1.7), a * Math.sin(t * 1.07 + 0.4))
    camera.position.add(applied.current)
  }, -2)
  useEffect(() => () => void applied.current.set(0, 0, 0), [])
  return null
}

/** The camera stays at least this far above the topography in the site view, m. */
const EYE_MIN_M = 1.6

/** Keeps the camera above the ground: a view from under the terrain is never what was meant. */
export function StayAboveGround({ origin }: Pick<LayerProps, 'origin'>) {
  useFrame(({ camera }) => {
    const ground = blastView.groundAt?.(camera.position.x + origin[0], camera.position.y + origin[1])
    if (ground === undefined || !Number.isFinite(ground)) return
    const floor = ground - origin[2] + EYE_MIN_M
    if (camera.position.z < floor) camera.position.z = floor
  }, -3)
  return null
}

/** A rumble made in the browser: filtered noise with a slow decay over a falling low tone. */
function rumble(context: AudioContext, loudness: number): void {
  const now = context.currentTime
  const seconds = 2.6
  const buffer = context.createBuffer(1, Math.floor(context.sampleRate * seconds), context.sampleRate)
  const samples = buffer.getChannelData(0)
  let last = 0
  for (let i = 0; i < samples.length; i++) {
    // Brown-ish noise: a leaky sum of white noise.
    last = 0.985 * last + 0.15 * (Math.random() * 2 - 1)
    samples[i] = last
  }
  const noise = context.createBufferSource()
  noise.buffer = buffer
  const filter = context.createBiquadFilter()
  filter.type = 'lowpass'
  filter.frequency.setValueAtTime(420, now)
  filter.frequency.exponentialRampToValueAtTime(70, now + seconds)
  const gain = context.createGain()
  gain.gain.setValueAtTime(0.0001, now)
  gain.gain.exponentialRampToValueAtTime(loudness, now + 0.04)
  gain.gain.exponentialRampToValueAtTime(0.0001, now + seconds)
  noise.connect(filter).connect(gain).connect(context.destination)
  noise.start(now)
  const tone = context.createOscillator()
  tone.frequency.setValueAtTime(62, now)
  tone.frequency.exponentialRampToValueAtTime(28, now + 1.4)
  const toneGain = context.createGain()
  toneGain.gain.setValueAtTime(0.0001, now)
  toneGain.gain.exponentialRampToValueAtTime(loudness * 0.9, now + 0.03)
  toneGain.gain.exponentialRampToValueAtTime(0.0001, now + 1.5)
  tone.connect(toneGain).connect(context.destination)
  tone.start(now)
  tone.stop(now + 1.6)
}

/** Plays the rumble once per run, when the player time passes the arrival of the sound at the camera. */
export function BlastSound({ bundle, origin }: LayerProps) {
  const state = useRef<{ context: AudioContext | null; last: number }>({ context: null, last: 0 })
  const first = useMemo(() => Math.min(...bundle.timeline.holes.map((h) => h.t_fire_ms), Infinity), [bundle])
  const centre = useMemo(() => {
    const [bx, by, bz] = blastCentre(bundle as BlastBundle)
    return new Vector3(bx - origin[0], by - origin[1], bz - origin[2])
  }, [bundle, origin])
  useFrame(({ camera }) => {
    const { t, playing } = useBlast.getState()
    const s = state.current
    const distance = camera.position.distanceTo(centre)
    const arrival = first + (distance / SOUND_SPEED_M_S) * 1000
    if (playing && Number.isFinite(arrival) && s.last < arrival && t >= arrival && t - s.last < 1500) {
      try {
        s.context ??= new AudioContext()
        void s.context.resume()
        // Quieter with distance (inverse distance from 100 m), never silent.
        rumble(s.context, Math.min(0.5, Math.max(0.08, 0.5 * (100 / Math.max(100, distance)))))
      } catch {
        // No audio device or not allowed: the view stays silent.
      }
    }
    s.last = t
  })
  useEffect(
    () => () => {
      void state.current.context?.close()
      state.current.context = null
    },
    [],
  )
  return null
}

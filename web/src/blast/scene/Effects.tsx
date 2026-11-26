/**
 * What makes the site view feel like standing there, beyond the pictures of rock and dust:
 *
 * - `BlastShadows`: the sun's shadows around the blast (fragments in the air and on the
 *   pile, the bench walls, stones and shrubs). A weak light from the sun's direction owns
 *   the shadow map; `ShadowCatcher` is a copy of a ground mesh that shows only the shadow
 *   falling on it, so the terrain's own picture and shading stay as they are.
 * - `Flashes`: a short glow at each collar when its hole fires.
 * - `BlastSound`: the sound of the blast (cracks of the holes, boom, rolling rumble, echo,
 *   falling rock), started with the first hole.
 *
 * The strength of the glow and the make-up of the sound are chosen for effect. The camera
 * itself never moves on its own.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import {
  AdditiveBlending, DirectionalLight, InstancedBufferAttribute, InstancedBufferGeometry, Mesh, PlaneGeometry, ShaderMaterial,
  type BufferGeometry, type Plane,
} from 'three'
import { useBlast } from '../state'
import { blastView, type LayerProps } from './common'
import { blastCentre } from './realism'
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

/** Brown-ish noise (a leaky sum of white noise), `seconds` long. */
function noiseBuffer(context: BaseAudioContext, seconds: number): AudioBuffer {
  const buffer = context.createBuffer(1, Math.floor(context.sampleRate * seconds), context.sampleRate)
  const samples = buffer.getChannelData(0)
  let last = 0
  for (let i = 0; i < samples.length; i++) {
    last = 0.985 * last + 0.15 * (Math.random() * 2 - 1)
    samples[i] = last
  }
  return buffer
}

/** At most this many separate cracks are made; more holes share them. */
const CRACKS = 48
/** Length of the whole sound, s. */
export const BLAST_SOUND_S = 6.5

/**
 * The sound of a bench blast, made in the browser and started with the first hole:
 * - a crack for each hole at its own firing time (a short burst of bright noise), which
 *   together give the tearing ripple of a delayed blast;
 * - a deep boom under it (a tone falling from 70 to 24 Hz) and a second, lower thump;
 * - a rumble that rolls on and darkens over five seconds;
 * - an echo of all of it from the ground around (two delays with feedback);
 * - falling rock: faint irregular clatter while the fragments fly and land.
 * `firesS` are the firing times in seconds after the first hole. Levels are chosen by ear
 * of the kind of sound, not computed from the charge: it is not a prediction of the airblast.
 */
function rumble(context: BaseAudioContext, loudness: number, firesS: readonly number[] = [0]): void {
  const now = context.currentTime + 0.02
  const master = context.createGain()
  master.gain.value = loudness
  const squash = context.createDynamicsCompressor()
  squash.threshold.value = -14
  squash.ratio.value = 5
  master.connect(squash).connect(context.destination)
  // Echo from the terrain: two taps fed back through a low-pass, so each return is duller.
  const echoIn = context.createGain()
  echoIn.gain.value = 0.42
  for (const [delaySeconds, feedback] of [[0.31, 0.38], [0.53, 0.3]] as const) {
    const delay = context.createDelay(1)
    delay.delayTime.value = delaySeconds
    const dull = context.createBiquadFilter()
    dull.type = 'lowpass'
    dull.frequency.value = 900
    const back = context.createGain()
    back.gain.value = feedback
    echoIn.connect(delay).connect(dull).connect(master)
    dull.connect(back).connect(delay)
  }
  const out = (node: AudioNode) => {
    node.connect(master)
    node.connect(echoIn)
  }

  const noise = noiseBuffer(context, BLAST_SOUND_S)
  // Cracks of the holes.
  const fires = [...firesS].sort((a, b) => a - b)
  const step = Math.max(1, Math.ceil(fires.length / CRACKS))
  for (let i = 0; i < fires.length; i += step) {
    const at = now + fires[i]
    const source = context.createBufferSource()
    source.buffer = noise
    const band = context.createBiquadFilter()
    band.type = 'bandpass'
    band.frequency.value = 900 + 1400 * ((i * 0.618) % 1)
    band.Q.value = 0.8
    const gain = context.createGain()
    gain.gain.setValueAtTime(0.0001, at)
    gain.gain.exponentialRampToValueAtTime(0.5, at + 0.004)
    gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.09)
    source.connect(band).connect(gain)
    out(gain)
    source.start(at, (i * 0.137) % (BLAST_SOUND_S - 0.2), 0.12)
  }
  const span = fires.length > 0 ? fires[fires.length - 1] : 0
  // Boom and thump.
  for (const [from, to, level, seconds, start] of [[70, 24, 1.0, 2.4, 0], [46, 20, 0.7, 3.0, span * 0.6]] as const) {
    const tone = context.createOscillator()
    tone.frequency.setValueAtTime(from, now + start)
    tone.frequency.exponentialRampToValueAtTime(to, now + start + seconds * 0.7)
    const gain = context.createGain()
    gain.gain.setValueAtTime(0.0001, now + start)
    gain.gain.exponentialRampToValueAtTime(level, now + start + 0.03)
    gain.gain.exponentialRampToValueAtTime(0.0001, now + start + seconds)
    tone.connect(gain)
    out(gain)
    tone.start(now + start)
    tone.stop(now + start + seconds + 0.1)
  }
  // Rolling rumble.
  const roll = context.createBufferSource()
  roll.buffer = noise
  const dark = context.createBiquadFilter()
  dark.type = 'lowpass'
  dark.frequency.setValueAtTime(520, now)
  dark.frequency.exponentialRampToValueAtTime(60, now + 5)
  const rollGain = context.createGain()
  rollGain.gain.setValueAtTime(0.0001, now)
  rollGain.gain.exponentialRampToValueAtTime(1.6, now + 0.05 + span * 0.5)
  rollGain.gain.exponentialRampToValueAtTime(0.02, now + 5.2)
  rollGain.gain.linearRampToValueAtTime(0, now + BLAST_SOUND_S - 0.1)
  roll.connect(dark).connect(rollGain)
  out(rollGain)
  roll.start(now)
  // Falling rock.
  const fall = context.createBufferSource()
  fall.buffer = noise
  const clatter = context.createBiquadFilter()
  clatter.type = 'bandpass'
  clatter.frequency.value = 700
  clatter.Q.value = 0.6
  const fallGain = context.createGain()
  fallGain.gain.setValueAtTime(0, now)
  for (let k = 0; k < 60; k++) {
    // Irregular little bursts between 0.7 and 4.2 s, densest in the middle.
    const at = now + 0.7 + 3.5 * ((k * 0.754877) % 1)
    const level = 0.3 * (0.3 + ((k * 0.3819) % 1)) * Math.sin(Math.PI * ((at - now - 0.7) / 3.5))
    fallGain.gain.setValueAtTime(0.0001, at)
    fallGain.gain.linearRampToValueAtTime(Math.max(0.0002, level), at + 0.008)
    fallGain.gain.linearRampToValueAtTime(0.0001, at + 0.05)
  }
  fall.connect(clatter).connect(fallGain)
  out(fallGain)
  fall.start(now, 1.3)
}

/**
 * The sound rendered off-line and measured, to check the signal without loudspeakers: peak
 * and RMS level, how long it stays above a twentieth of its peak, and the share of its
 * energy below 200 Hz (by a one-pole low-pass, cut-off 200 Hz).
 */
export async function rumbleCheck(loudness = 0.8, firesS: readonly number[] = [0]): Promise<{ peak: number; rms: number; seconds: number; lowShare: number }> {
  const rate = 44100
  const context = new OfflineAudioContext(1, Math.floor(rate * (BLAST_SOUND_S + 1)), rate)
  rumble(context, loudness, firesS)
  const samples = (await context.startRendering()).getChannelData(0)
  let peak = 0
  let energy = 0
  let low = 0
  let last = 0
  let state = 0
  const k = 1 - Math.exp((-2 * Math.PI * 200) / rate)
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i]
    peak = Math.max(peak, Math.abs(v))
    energy += v * v
    state += k * (v - state)
    low += state * state
  }
  for (let i = 0; i < samples.length; i++) if (Math.abs(samples[i]) > 0.05 * peak) last = i
  return { peak, rms: Math.sqrt(energy / samples.length), seconds: last / rate, lowShare: energy > 0 ? low / energy : 0 }
}

/** What the sound did last, for a check from the browser console (development builds). */
export const soundLog: { played: number; contextState: string; atMs: number } = { played: 0, contextState: 'none', atMs: 0 }
if (import.meta.env.DEV && typeof window !== 'undefined') {
  Object.assign(window, { __blastSound: { check: rumbleCheck, log: soundLog } })
}

/**
 * Plays the blast sound once per run, when the player time passes the first hole. It starts
 * with the picture: the delay of real sound over the distance to the camera is left out,
 * because the player shows the detonation in slow motion and a late sound reads as a fault.
 */
export function BlastSound({ bundle }: LayerProps) {
  const state = useRef<{ context: AudioContext | null; last: number }>({ context: null, last: -1 })
  const fires = useMemo(() => {
    const times = bundle.timeline.holes.map((h) => h.t_fire_ms)
    const first = times.length > 0 ? Math.min(...times) : 0
    return { first, seconds: times.map((t) => (t - first) / 1000) }
  }, [bundle])
  useFrame(() => {
    const { t, playing } = useBlast.getState()
    const s = state.current
    if (playing && s.last <= fires.first && t > fires.first && t - Math.max(s.last, 0) < 1500) {
      try {
        s.context ??= new AudioContext()
        void s.context.resume()
        rumble(s.context, 0.8, fires.seconds)
        soundLog.played += 1
        soundLog.contextState = s.context.state
        soundLog.atMs = t
      } catch {
        // No audio device or not allowed: the view stays silent.
      }
    }
    s.last = playing ? t : Math.min(t, fires.first)
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

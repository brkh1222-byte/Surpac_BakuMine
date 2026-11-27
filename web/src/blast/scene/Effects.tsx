/**
 * What makes the site view feel like standing there, beyond the pictures of rock and dust:
 *
 * - `BlastShadows`: the sun's shadows around the blast (fragments in the air and on the
 *   pile, the bench walls, stones and shrubs). A weak light from the sun's direction owns
 *   the shadow map; `ShadowCatcher` is a copy of a ground mesh that shows only the shadow
 *   falling on it, so the terrain's own picture and shading stay as they are.
 * - `Flashes`: a short glow at each collar when its hole fires.
 * - `BlastSound`: every hole booms when the player passes its firing time; a rolling rumble
 *   and falling rock follow the last hole.
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

/** Length of the tail (rolling rumble and falling rock) that follows the last hole, s. */
export const BLAST_TAIL_S = 6
/** Two booms are never started closer together than this, s: at full speed holes 17 ms apart merge. */
const BOOM_GAP_S = 0.03

interface SoundBus {
  /** Sends a node to the loudspeakers and to the echo. */
  out: (node: AudioNode) => void
  noise: AudioBuffer
}

/** Master level, a compressor against clipping, and an echo from the ground around (two delays with feedback). */
function soundBus(context: BaseAudioContext, loudness: number): SoundBus {
  const master = context.createGain()
  master.gain.value = loudness
  const squash = context.createDynamicsCompressor()
  squash.threshold.value = -14
  squash.ratio.value = 5
  master.connect(squash).connect(context.destination)
  const echoIn = context.createGain()
  echoIn.gain.value = 0.38
  for (const [delaySeconds, feedback] of [[0.31, 0.36], [0.53, 0.28]] as const) {
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
  return {
    out: (node) => {
      node.connect(master)
      node.connect(echoIn)
    },
    noise: noiseBuffer(context, BLAST_TAIL_S + 0.5),
  }
}

/**
 * One hole going off, `at` (context time): a sharp crack (a short burst of bright noise) on
 * a deep boom (a tone falling from about 85 to 30 Hz over half a second). `k` only varies
 * the pitch a little from hole to hole, so a row of holes does not sound like one note.
 */
function holeBoom(context: BaseAudioContext, bus: SoundBus, at: number, k: number): void {
  const vary = (k * 0.618034) % 1
  const crack = context.createBufferSource()
  crack.buffer = bus.noise
  const band = context.createBiquadFilter()
  band.type = 'bandpass'
  band.frequency.value = 800 + 1500 * vary
  band.Q.value = 0.8
  const crackGain = context.createGain()
  crackGain.gain.setValueAtTime(0.0001, at)
  crackGain.gain.exponentialRampToValueAtTime(0.7, at + 0.004)
  crackGain.gain.exponentialRampToValueAtTime(0.0001, at + 0.11)
  crack.connect(band).connect(crackGain)
  bus.out(crackGain)
  crack.start(at, (k * 0.137) % BLAST_TAIL_S, 0.14)
  const tone = context.createOscillator()
  tone.frequency.setValueAtTime(78 + 14 * vary, at)
  tone.frequency.exponentialRampToValueAtTime(30, at + 0.4)
  const toneGain = context.createGain()
  toneGain.gain.setValueAtTime(0.0001, at)
  toneGain.gain.exponentialRampToValueAtTime(1.0, at + 0.012)
  toneGain.gain.exponentialRampToValueAtTime(0.0001, at + 0.6)
  tone.connect(toneGain)
  bus.out(toneGain)
  tone.start(at)
  tone.stop(at + 0.65)
}

/** After the last hole: a rumble that rolls on and darkens, and faint irregular clatter of falling rock. */
function blastTail(context: BaseAudioContext, bus: SoundBus, at: number): void {
  const roll = context.createBufferSource()
  roll.buffer = bus.noise
  const dark = context.createBiquadFilter()
  dark.type = 'lowpass'
  dark.frequency.setValueAtTime(480, at)
  dark.frequency.exponentialRampToValueAtTime(60, at + 4.5)
  const rollGain = context.createGain()
  rollGain.gain.setValueAtTime(0.0001, at)
  rollGain.gain.exponentialRampToValueAtTime(1.3, at + 0.08)
  rollGain.gain.exponentialRampToValueAtTime(0.02, at + 4.8)
  rollGain.gain.linearRampToValueAtTime(0, at + BLAST_TAIL_S - 0.1)
  roll.connect(dark).connect(rollGain)
  bus.out(rollGain)
  roll.start(at, 0, BLAST_TAIL_S)
  const fall = context.createBufferSource()
  fall.buffer = bus.noise
  const clatter = context.createBiquadFilter()
  clatter.type = 'bandpass'
  clatter.frequency.value = 700
  clatter.Q.value = 0.6
  const fallGain = context.createGain()
  fallGain.gain.setValueAtTime(0, at)
  const bursts: number[] = []
  for (let k = 0; k < 60; k++) bursts.push(0.3 + 3.6 * ((k * 0.754877) % 1))
  bursts.sort((a, b) => a - b)
  bursts.forEach((offset, k) => {
    const level = 0.3 * (0.3 + ((k * 0.3819) % 1)) * Math.sin(Math.PI * ((offset - 0.3) / 3.6))
    fallGain.gain.setValueAtTime(0.0001, at + offset)
    fallGain.gain.linearRampToValueAtTime(Math.max(0.0002, level), at + offset + 0.008)
    fallGain.gain.linearRampToValueAtTime(0.0001, at + offset + 0.05)
  })
  fall.connect(clatter).connect(fallGain)
  bus.out(fallGain)
  fall.start(at, 1.3, BLAST_TAIL_S - 1.4)
}

/**
 * A whole blast rendered off-line and measured, to check the signal without loudspeakers:
 * a boom at each of `firesS` (seconds) and the tail after the last. Returns the peak and
 * RMS level, how long it stays above a twentieth of its peak, and the share of its energy
 * below 200 Hz (by a one-pole low-pass, cut-off 200 Hz).
 */
export async function rumbleCheck(loudness = 0.8, firesS: readonly number[] = [0]): Promise<{ peak: number; rms: number; seconds: number; lowShare: number }> {
  const rate = 44100
  const last = firesS.length > 0 ? Math.max(...firesS) : 0
  const context = new OfflineAudioContext(1, Math.floor(rate * (last + BLAST_TAIL_S + 1)), rate)
  const bus = soundBus(context, loudness)
  firesS.forEach((t, k) => holeBoom(context, bus, 0.02 + t, k))
  blastTail(context, bus, 0.02 + last)
  const samples = (await context.startRendering()).getChannelData(0)
  let peak = 0
  let energy = 0
  let low = 0
  let end = 0
  let state = 0
  const k = 1 - Math.exp((-2 * Math.PI * 200) / rate)
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i]
    peak = Math.max(peak, Math.abs(v))
    energy += v * v
    state += k * (v - state)
    low += state * state
  }
  for (let i = 0; i < samples.length; i++) if (Math.abs(samples[i]) > 0.05 * peak) end = i
  return { peak, rms: Math.sqrt(energy / samples.length), seconds: end / rate, lowShare: energy > 0 ? low / energy : 0 }
}

/** What the sound did in this run, for a check from the browser console (development builds). */
export const soundLog: { booms: number; merged: number; tails: number; contextState: string; lastBoomMs: number } = {
  booms: 0, merged: 0, tails: 0, contextState: 'none', lastBoomMs: 0,
}
if (import.meta.env.DEV && typeof window !== 'undefined') {
  Object.assign(window, { __blastSound: { check: rumbleCheck, log: soundLog } })
}

/**
 * The sound follows the picture: every hole booms at the moment the player time passes its
 * firing time, at any player speed (in slow motion the booms are spread out exactly as the
 * flashes are), and the tail follows the last hole. Scrubbing and jumps make no sound.
 * The delay of real sound over the distance to the camera is left out on purpose.
 */
export function BlastSound({ bundle }: LayerProps) {
  const state = useRef<{ context: AudioContext | null; bus: SoundBus | null; last: number; next: number; lastBoomAt: number }>({
    context: null, bus: null, last: 0, next: 0, lastBoomAt: -1,
  })
  const fires = useMemo(() => bundle.timeline.holes.map((h) => h.t_fire_ms).sort((a, b) => a - b), [bundle])
  useFrame(() => {
    const { t, playing } = useBlast.getState()
    const s = state.current
    // Played forward by the clock (not scrubbed or jumped): the holes passed since the last frame go off.
    const stepped = playing && t >= s.last && t - s.last < 400
    if (!stepped) {
      let n = 0
      while (n < fires.length && fires[n] <= t) n++
      // At the very beginning nothing has gone off yet: the holes that fire at time zero are still to come.
      s.next = t <= (fires[0] ?? 0) + 50 ? 0 : n
      s.last = t
      if (!playing || s.next >= fires.length || fires[s.next] > t) return
    }
    while (s.next < fires.length && fires[s.next] <= t) {
      try {
        s.context ??= new AudioContext()
        void s.context.resume()
        s.bus ??= soundBus(s.context, 0.8)
        const now = s.context.currentTime
        if (now - s.lastBoomAt >= BOOM_GAP_S) {
          holeBoom(s.context, s.bus, now + 0.005, s.next)
          s.lastBoomAt = now
          soundLog.booms += 1
          soundLog.lastBoomMs = fires[s.next]
        } else soundLog.merged += 1
        if (s.next === fires.length - 1) {
          blastTail(s.context, s.bus, now + 0.02)
          soundLog.tails += 1
        }
        soundLog.contextState = s.context.state
      } catch {
        // No audio device or not allowed: the view stays silent.
      }
      s.next += 1
    }
    s.last = t
  })
  useEffect(
    () => () => {
      void state.current.context?.close()
      state.current.context = null
      state.current.bus = null
    },
    [],
  )
  return null
}

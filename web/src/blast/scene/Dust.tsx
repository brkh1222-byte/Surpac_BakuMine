/**
 * Dust of the blast, as soft camera-facing puffs. A puff is born where something throws
 * dust up; air drag turns its start velocity into the wind's (dv/dt = (wind - v) / tau),
 * while it spreads and fades (realism.dustCentre, dustRadius; the same laws are in the vertex
 * shader, so the dust is a pure function of the player time and can be scrubbed).
 *
 * Sources (where and when are from the simulation; amounts, sizes, colours and lifetimes
 * are assumed):
 * - every hole when it fires: a column from the collar (stemming and soil) and a low surge
 *   along the ground, each hole with its own strength;
 * - every soil parcel and every third rock parcel at its launch, with part of its velocity;
 * - every parcel where and when it lands.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { Color, DoubleSide, InstancedBufferAttribute, InstancedBufferGeometry, Mesh, NormalBlending, PlaneGeometry, ShaderMaterial, Vector3 } from 'three'
import { useBlast } from '../state'
import type { LayerProps } from './common'
import { depthBelowCollars, DUST_DRAG_S, DUST_RISE, DUST_WIND, hash01, SOIL_THICKNESS_M } from './realism'
import { SUN } from './terrain'

const SOIL_DUST = '#b9a585'
const ROCK_DUST = '#aaa49a'

/**
 * The shadow of a puff: the same centre and radius, slid along the sun's ray down to the
 * height the puff started from (the ground there) and laid flat. A thin, soft, dark disc;
 * many of them add up to the shadow of the cloud.
 */
const SHADOW_VERTEX = /* glsl */ `
attribute vec3 aStart;
attribute vec3 aVelocity;
attribute vec4 aLife;
attribute vec4 aTint;
uniform float uTime;
uniform float uTau;
uniform vec3 uDrift;
uniform vec3 uSun;
varying vec2 vCorner;
varying float vAge;
varying float vBorn;
varying float vOpacity;
void main() {
  float dt = (uTime - aLife.x) / 1000.0;
  vBorn = dt;
  vCorner = position.xy * 2.0;
  vAge = dt / aLife.y;
  vOpacity = aTint.a;
  if (dt <= 0.0 || dt >= aLife.y) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  float lag = uTau * (1.0 - exp(-dt / uTau));
  vec3 centre = aStart + aVelocity * lag + uDrift * (dt - lag);
  float radius = aLife.z + aLife.w * sqrt(dt);
  float height = max(centre.z - aStart.z, 0.0);
  vec3 ground = vec3(centre.xy - uSun.xy / uSun.z * height, aStart.z + 0.35);
  ground.xy += position.xy * 2.0 * radius;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(ground, 1.0);
}
`
const SHADOW_FRAGMENT = /* glsl */ `
varying vec2 vCorner;
varying float vAge;
varying float vBorn;
varying float vOpacity;
void main() {
  float r = length(vCorner);
  if (r >= 1.0) discard;
  float life = smoothstep(0.0, 0.01, vBorn) * (1.0 - smoothstep(0.35, 1.0, vAge));
  float alpha = 0.3 * vOpacity * life * smoothstep(1.0, 0.2, r);
  if (alpha < 0.002) discard;
  gl_FragColor = vec4(0.0, 0.0, 0.0, alpha);
}
`

const VERTEX = /* glsl */ `
attribute vec3 aStart;
attribute vec3 aVelocity;
attribute vec4 aLife;   // birth ms, life s, start radius m, growth m / sqrt(s)
attribute vec4 aTint;   // rgb (linear), opacity
attribute float aSeed;
uniform float uTime;    // ms
uniform float uTau;
uniform vec3 uDrift;    // wind east, wind north, rise (m/s)
varying vec2 vCorner;
varying vec4 vTint;
varying float vAge;     // 0 at birth .. 1 at the end of its life
varying float vBorn;    // seconds since birth
varying float vSeed;
void main() {
  float dt = (uTime - aLife.x) / 1000.0;
  vBorn = dt;
  vCorner = position.xy * 2.0;
  vTint = aTint;
  vSeed = aSeed;
  vAge = dt / aLife.y;
  if (dt <= 0.0 || dt >= aLife.y) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // outside the clip volume: not drawn
    return;
  }
  float lag = uTau * (1.0 - exp(-dt / uTau));
  vec3 centre = aStart + aVelocity * lag + uDrift * (dt - lag);
  float radius = aLife.z + aLife.w * sqrt(dt);
  vec4 mv = modelViewMatrix * vec4(centre, 1.0);
  mv.xy += position.xy * 2.0 * radius;
  gl_Position = projectionMatrix * mv;
}
`

const FRAGMENT = /* glsl */ `
uniform vec3 uHaze;
varying vec2 vCorner;
varying vec4 vTint;
varying float vAge;
varying float vBorn;
varying float vSeed;
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
  float r = length(vCorner);
  if (r >= 1.0) discard;
  // A soft ball with a billowing edge and denser and thinner parts inside.
  float turn = vSeed * 6.2832;
  vec2 q = mat2(cos(turn), -sin(turn), sin(turn), cos(turn)) * vCorner * 2.4 + vSeed * 37.0;
  float billow = 0.55 * noise(q) + 0.3 * noise(q * 2.3 + 11.0) + 0.15 * noise(q * 5.1);
  float body = smoothstep(1.0, 0.1, r + 0.6 * (billow - 0.5)) * (0.45 + 1.1 * billow);
  // There within a hundredth of a second of its birth (in slow motion the eye must see the
  // dust the moment the hole fires), slow to thin out.
  float life = smoothstep(0.0, 0.01, vBorn) * (1.0 - smoothstep(0.35, 1.0, vAge));
  // The lower edge thins out, so a puff does not end in a hard line where it meets the ground.
  float alpha = vTint.a * body * life * smoothstep(-1.0, -0.25, vCorner.y);
  if (alpha < 0.004) discard;
  // Lit from above and the sun side; the first instant is bright (the hot gases).
  float light = 0.74 + 0.3 * vCorner.y + 0.1 * vCorner.x + 0.3 * (billow - 0.5);
  vec3 colour = vTint.rgb * light;
  colour = mix(colour, vec3(1.0, 0.86, 0.62), 0.55 * (1.0 - smoothstep(0.0, 0.06, vBorn)));
  colour = mix(colour, uHaze, 0.25 * smoothstep(0.3, 1.0, vAge));
  gl_FragColor = vec4(colour, alpha);
  #include <colorspace_fragment>
}
`

interface Puff {
  start: readonly number[]
  velocity: readonly number[]
  birth: number
  life: number
  radius: number
  growth: number
  tint: readonly [number, number, number]
  opacity: number
}

/** The puffs of a bundle: positions in world coordinates, times in ms. */
export function dustPuffs(bundle: LayerProps['bundle']): Puff[] {
  const soilDust = new Color(SOIL_DUST)
  const rockDust = new Color(ROCK_DUST)
  const soil: [number, number, number] = [soilDust.r, soilDust.g, soilDust.b]
  const rock: [number, number, number] = [rockDust.r, rockDust.g, rockDust.b]
  const puffs: Puff[] = []
  const holes = bundle.timeline.holes
  const collars = holes.map((h) => h.collar)
  holes.forEach((h, i) => {
    // Holes do not all vent alike: a strength per hole makes the cloud ragged.
    const strength = 0.45 + 1.2 * hash01(i, 30)
    for (let c = 0; c < 18; c++) {
      const seed = i * 16 + c
      // Fourteen puffs rise as a column from the collar; four roll out along the ground.
      const column = c < 14
      const a = hash01(seed, 34) * Math.PI * 2
      const out = column ? 1.5 + 4.5 * hash01(seed, 35) : 6 + 8 * hash01(seed, 35)
      const up = column ? (5 + 36 * hash01(seed, 31) ** 1.5) * strength : 0.4 + 1.2 * hash01(seed, 31)
      const dark = 0.74 + 0.36 * hash01(seed, 40)
      puffs.push({
        start: [h.collar[0] + (hash01(seed, 32) - 0.5) * 3, h.collar[1] + (hash01(seed, 33) - 0.5) * 3, h.collar[2] + 0.3],
        velocity: [Math.cos(a) * out, Math.sin(a) * out, up],
        // The first puffs are born with the hole; the rest follow within a few hundredths of a second.
        birth: h.t_fire_ms + (c < 4 ? 0 : (column ? 45 : 20) * hash01(seed, 36)),
        life: (column ? 7 : 5) + 5 * hash01(seed, 37),
        radius: column ? 0.7 + 0.9 * hash01(seed, 38) : 1.2 + 0.8 * hash01(seed, 38),
        growth: (column ? 2.3 : 2.9) + 1.5 * hash01(seed, 39),
        tint: [soil[0] * dark, soil[1] * dark, soil[2] * dark],
        opacity: (column ? 0.17 : 0.12) + 0.1 * hash01(seed, 51),
      })
    }
  })
  const p = bundle.throw.particles
  for (let i = 0; i < p.t0_ms.length; i++) {
    const isSoil = depthBelowCollars(p.p0[i], collars) < SOIL_THICKNESS_M
    const v = p.v0[i]
    if (isSoil || i % 3 === 0) {
      for (let c = 0; c < (isSoil ? 2 : 1); c++) {
        const seed = i * 4 + c
        const share = 0.35 + 0.4 * hash01(seed, 41)
        puffs.push({
          start: p.p0[i],
          velocity: [v[0] * share, v[1] * share, v[2] * share + (isSoil ? 2.5 : 1)],
          birth: p.t0_ms[i] + 8 * hash01(seed, 42),
          life: 6 + 5 * hash01(seed, 43),
          radius: 0.7 + 0.7 * hash01(seed, 44),
          growth: isSoil ? 1.9 : 1.4,
          tint: isSoil ? soil : rock,
          opacity: isSoil ? 0.3 : 0.22,
        })
      }
    }
    if (i % 2 === 0) {
      const seed = i * 4 + 3
      puffs.push({
        start: [p.p_land[i][0], p.p_land[i][1], p.p_land[i][2] + 0.3],
        velocity: [v[0] * 0.15 + (hash01(seed, 45) - 0.5) * 2, v[1] * 0.15 + (hash01(seed, 46) - 0.5) * 2, 1.2 + 1.5 * hash01(seed, 47)],
        birth: p.t_land_ms[i],
        life: 5 + 4 * hash01(seed, 48),
        radius: 0.6 + 0.5 * hash01(seed, 49),
        growth: 1.2 + 0.6 * hash01(seed, 50),
        tint: isSoil ? soil : rock,
        opacity: 0.2,
      })
    }
  }
  return puffs
}

export function Dust({ bundle, origin }: LayerProps) {
  const invalidate = useThree((s) => s.invalidate)
  const mesh = useMemo(() => {
    const puffs = dustPuffs(bundle)
    const n = puffs.length
    const geometry = new InstancedBufferGeometry()
    const quad = new PlaneGeometry(1, 1)
    geometry.setAttribute('position', quad.getAttribute('position'))
    geometry.setIndex(quad.getIndex())
    const start = new Float32Array(n * 3)
    const velocity = new Float32Array(n * 3)
    const life = new Float32Array(n * 4)
    const tint = new Float32Array(n * 4)
    const seed = new Float32Array(n)
    puffs.forEach((p, k) => {
      for (let j = 0; j < 3; j++) {
        start[k * 3 + j] = p.start[j] - origin[j]
        velocity[k * 3 + j] = p.velocity[j]
        tint[k * 4 + j] = p.tint[j]
      }
      life.set([p.birth, p.life, p.radius, p.growth], k * 4)
      tint[k * 4 + 3] = p.opacity
      seed[k] = hash01(k, 60)
    })
    geometry.setAttribute('aStart', new InstancedBufferAttribute(start, 3))
    geometry.setAttribute('aVelocity', new InstancedBufferAttribute(velocity, 3))
    geometry.setAttribute('aLife', new InstancedBufferAttribute(life, 4))
    geometry.setAttribute('aTint', new InstancedBufferAttribute(tint, 4))
    geometry.setAttribute('aSeed', new InstancedBufferAttribute(seed, 1))
    geometry.instanceCount = n
    const material = new ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: {
        uTime: { value: 0 },
        uTau: { value: DUST_DRAG_S },
        uDrift: { value: [DUST_WIND[0], DUST_WIND[1], DUST_RISE] },
        uHaze: { value: new Color('#cfd6da') },
      },
      transparent: true,
      depthWrite: false,
      blending: NormalBlending,
      side: DoubleSide,
    })
    const out = new Mesh(geometry, material)
    out.frustumCulled = false
    out.renderOrder = 6
    const shadowMaterial = new ShaderMaterial({
      vertexShader: SHADOW_VERTEX,
      fragmentShader: SHADOW_FRAGMENT,
      uniforms: {
        uTime: material.uniforms.uTime,
        uTau: material.uniforms.uTau,
        uDrift: material.uniforms.uDrift,
        uSun: { value: new Vector3(...SUN).normalize() },
      },
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    })
    const shadow = new Mesh(geometry, shadowMaterial)
    shadow.frustumCulled = false
    shadow.renderOrder = 5
    out.add(shadow)
    quad.dispose()
    return out
  }, [bundle, origin])
  useEffect(() => {
    invalidate()
    return () => {
      mesh.geometry.dispose()
      ;(mesh.material as ShaderMaterial).dispose()
      for (const child of mesh.children) ((child as Mesh).material as ShaderMaterial).dispose()
    }
  }, [mesh, invalidate])
  useFrame(() => {
    ;(mesh.material as ShaderMaterial).uniforms.uTime.value = useBlast.getState().t
  })
  return <primitive object={mesh} />
}

/**
 * The realistic look of the ground under the satellite picture. Everything here is a display
 * effect added by the viewer; none of it is measured and none of it changes the surface:
 *
 * - relief shading: the picture is brightened / darkened by the slope towards a fixed sun
 *   (south-east, 45 degrees up, where the sun of a mid-morning satellite pass stands north of
 *   the tropics), relative to flat ground, which keeps the picture's own brightness;
 * - fine relief from the picture: the brightness of the picture is read as a small height
 *   (the "bump map" of Blinn 1978, with the screen-space derivatives of Mikkelsen 2010), so
 *   bushes, rills and tracks catch the light;
 * - sharpening: an unsharp mask on the picture (the picture plus 0.9 of its difference
 *   from the mean of four neighbours 1.5 pixels away);
 * - grain: fractal value noise multiplied in at close range, where one pixel of the picture
 *   covers more than one pixel of the screen; within 150 m also dark pebble specks and a
 *   rough ground (a few centimetres of the same noise) that catches the light;
 * - fresh rock: ground that was cut (the bench face and floor) or broken (the muckpile) is
 *   drawn as procedural rock (bands along the bedding, mottling, grit) in the vertex colour, not as the picture
 *   of the old surface;
 * - haze: colours fade to the horizon colour with distance, 1 - exp(-d / HAZE_LENGTH).
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BackSide, Color, DoubleSide, ShaderMaterial, Vector2, Vector3, type Plane, type Texture } from 'three'

/** Direction to the sun, scene axes (x east, y north, z up). */
export const SUN: readonly [number, number, number] = [0.5, -0.5, Math.SQRT1_2]
/** Distance at which the haze has taken 63 % of the colour, m. */
export const HAZE_LENGTH = 12000
/** Fresh rock of a cut and broken rock of the muckpile (sRGB hex). */
export const ROCK_COLOR = '#8f877a'
export const MUCK_COLOR = '#958c7e'
const HORIZON = '#cfd6da'
const ZENITH = '#4f86c6'
const BELOW = '#8f8672'

/**
 * Brightness factor of ground with unit normal `n` under a sun at `sun` (unit): 1 on flat
 * ground, more on slopes that face the sun, less on those that face away; limited to
 * [0.5, 1.35]. The fragment shader uses the same expression.
 */
export function reliefShade(n: readonly number[], sun: readonly number[] = SUN, strength = 0.85): number {
  const lit = n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]
  return Math.min(1.35, Math.max(0.5, 1 + strength * (lit - sun[2])))
}

/** Share of the colour replaced by haze at distance `d` m. */
export function hazeAt(d: number, length = HAZE_LENGTH): number {
  return 1 - Math.exp(-Math.max(0, d) / length)
}

/** 0 on ground cut by less than 0.2 m, 1 from 1 m of cut (smoothstep between). */
export function cutRock(cut: number | null | undefined): number {
  if (typeof cut !== 'number' || !Number.isFinite(cut)) return 0
  const t = Math.min(1, Math.max(0, (cut - 0.2) / 0.8))
  return t * t * (3 - 2 * t)
}

const NOISE = /* glsl */ `
float hash(vec3 p) {
  p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
  float a = 0.5;
  float s = 0.0;
  for (int i = 0; i < 4; i++) {
    s += a * vnoise(p);
    p = p * 2.03 + 7.1;
    a *= 0.5;
  }
  return s / 0.9375;
}
`

const VERTEX = /* glsl */ `
attribute vec3 color;
attribute float rock;
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vNormal;
varying vec3 vTint;
varying float vRock;
varying float vDist;
#include <clipping_planes_pars_vertex>
void main() {
  vUv = uv;
  vPos = position;
  vNormal = normal;
  vTint = color;
  vRock = rock;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vDist = length(mvPosition.xyz);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`

const FRAGMENT = /* glsl */ `
uniform sampler2D map;
uniform vec3 sun;
uniform vec3 haze;
uniform float hazeLength;
uniform float opacity;
uniform float bump;
uniform vec2 texel;
uniform float sharpen;
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vNormal;
varying vec3 vTint;
varying float vRock;
varying float vDist;
#include <clipping_planes_pars_fragment>
${NOISE}
void main() {
  #include <clipping_planes_fragment>
  vec3 n = normalize(vNormal);
  if (n.z < 0.0) n = -n;
  vec3 dpdx = dFdx(vPos);
  vec3 dpdy = dFdy(vPos);

  // The picture, with grain at close range.
  vec3 base = texture2D(map, vUv).rgb;
  // Unsharp mask: the picture minus its local mean, added back, makes edges crisper.
  vec3 mean = 0.25 * (texture2D(map, vUv + vec2(texel.x, 0.0)).rgb + texture2D(map, vUv - vec2(texel.x, 0.0)).rgb
                    + texture2D(map, vUv + vec2(0.0, texel.y)).rgb + texture2D(map, vUv - vec2(0.0, texel.y)).rgb);
  base = clamp(base + sharpen * (base - mean), 0.0, 1.0);
  float close = 1.0 - smoothstep(30.0, 450.0, vDist);
  float grain = 0.6 * fbm(vPos * 0.9) + 0.4 * fbm(vPos * 5.0);
  base *= 1.0 + close * 0.3 * (grain - 0.5);

  // Pebbles: dark specks within about 150 m, where the picture has no detail left.
  float near = 1.0 - smoothstep(25.0, 150.0, vDist);
  float speck = smoothstep(0.74, 0.84, vnoise(vPos * 5.5)) * near;
  base *= 1.0 - 0.32 * speck;
  // Patches of gravel (greyer, rougher) and of sand (paler, smoother) a few metres across,
  // fine sand ripples, and small bright stones: within 150 m only.
  float soil = fbm(vPos * 0.23 + 3.7);
  float gravel = smoothstep(0.52, 0.66, soil) * near;
  float sand = smoothstep(0.48, 0.34, soil) * near;
  float lum = dot(base, vec3(0.299, 0.587, 0.114));
  base = mix(base, vec3(lum) * vec3(0.98, 0.96, 0.93), 0.35 * gravel);
  base = mix(base, base * vec3(1.1, 1.06, 0.98), 0.5 * sand);
  float pebble = smoothstep(0.8, 0.9, vnoise(vPos * 9.0 + 5.1)) * near * (0.4 + gravel);
  base = mix(base, vec3(0.62, 0.6, 0.56), 0.45 * pebble);
  float ripple = sin(dot(vPos.xy, vec2(2.3, 1.1)) + 4.0 * fbm(vPos * 0.6)) * sand;

  // Fine relief: the brightness of the picture as a height of "bump" metres, plus near the
  // camera a rough ground of the same noise as the grain (a few centimetres) and the pebbles.
  float h = dot(base, vec3(0.299, 0.587, 0.114)) * bump + near * (0.22 * grain + 0.05 * speck + 0.04 * vnoise(vPos * 14.0) + 0.07 * pebble + 0.012 * ripple + 0.1 * gravel * vnoise(vPos * 7.0));
  vec3 r1 = cross(dpdy, n);
  vec3 r2 = cross(n, dpdx);
  float det = dot(dpdx, r1);
  vec3 gradient = sign(det) * (dFdx(h) * r1 + dFdy(h) * r2);
  vec3 ns = normalize(abs(det) * n - gradient);

  // Fresh or broken rock: bands along the bedding, mottling, grit; facets catch the light.
  vec3 facet = normalize(cross(dpdx, dpdy));
  if (facet.z < 0.0) facet = -facet;
  float bands = fbm(vec3(vPos.xy * 0.05, vPos.z * 0.7));
  float patches = fbm(vPos * 0.35);
  float grit = 0.6 * fbm(vPos * 2.5) + 0.4 * vnoise(vPos * 11.0);
  vec3 stone = vTint * (0.82 + 0.36 * bands) * (0.88 + 0.24 * patches) * (0.84 + 0.32 * grit);

  vec3 colour = base;
  // The rock itself is rough: blocks a metre or two across with sharper small breaks, read
  // as a height and lit like the fine relief of the picture.
  float hr = 0.45 * fbm(vec3(vPos.xy * 0.55, vPos.z * 0.9)) + 0.12 * vnoise(vPos * 2.6) + 0.04 * vnoise(vPos * 9.0);
  vec3 rockGradient = sign(det) * (dFdx(hr) * r1 + dFdy(hr) * r2);
  vec3 rough = normalize(abs(det) * normalize(mix(n, facet, 0.6)) - rockGradient);
  stone *= 0.86 + 0.28 * smoothstep(0.25, 0.75, fbm(vec3(vPos.xy * 0.55, vPos.z * 0.9)));
  vec3 nn = normalize(mix(ns, rough, vRock));
  colour = mix(colour, stone, vRock);
  colour *= clamp(1.0 + 0.85 * (dot(nn, sun) - sun.z), 0.5, 1.35);
  // A slightly firmer picture: more contrast about mid grey and a little more colour.
  float grey = dot(colour, vec3(0.299, 0.587, 0.114));
  colour = mix(vec3(grey), colour, 1.12);
  colour = clamp((colour - 0.18) * 1.08 + 0.18, 0.0, 1.0);
  colour = mix(colour, haze, 1.0 - exp(-vDist / hazeLength));
  gl_FragColor = vec4(colour, opacity);
  #include <colorspace_fragment>
}
`

/** The material of ground under the satellite picture. Attributes: uv, normal, and optionally color + rock. */
export function TerrainMaterial({
  texture,
  opacity = 1,
  planes,
  offset = false,
}: {
  texture: Texture
  opacity?: number
  planes?: Plane[]
  offset?: boolean
}) {
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: VERTEX,
        fragmentShader: FRAGMENT,
        uniforms: {
          map: { value: null },
          sun: { value: new Vector3(...SUN).normalize() },
          haze: { value: new Color(HORIZON) },
          hazeLength: { value: HAZE_LENGTH },
          opacity: { value: 1 },
          bump: { value: 1.6 },
          texel: { value: new Vector2(1 / 2048, 1 / 2048) },
          sharpen: { value: 0.9 },
        },
        side: DoubleSide,
        clipping: true,
      }),
    [],
  )
  useEffect(() => () => material.dispose(), [material])
  material.uniforms.map.value = texture
  const image = texture.image as { width?: number; height?: number } | undefined
  if (image?.width && image.height) material.uniforms.texel.value.set(1.5 / image.width, 1.5 / image.height)
  material.uniforms.opacity.value = opacity
  material.transparent = opacity < 1
  material.clippingPlanes = planes ?? null
  material.clipIntersection = Boolean(planes && planes.length > 0)
  material.polygonOffset = offset
  material.polygonOffsetFactor = offset ? 2 : 0
  material.polygonOffsetUnits = offset ? 2 : 0
  return <primitive object={material} attach="material" />
}

const SKY_VERTEX = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position;
  vec4 p = projectionMatrix * vec4(mat3(viewMatrix) * position, 1.0);
  gl_Position = p.xyww;
}
`

const SKY_FRAGMENT = /* glsl */ `
uniform vec3 horizon;
uniform vec3 zenith;
uniform vec3 below;
uniform vec3 sun;
varying vec3 vDir;
${NOISE}
void main() {
  vec3 d = normalize(vDir);
  vec3 colour = mix(horizon, zenith, pow(max(d.z, 0.0), 0.55));
  // Fair-weather clouds on a flat layer overhead: denser patches of fractal noise, thinning
  // to the horizon, lit on the sun's side and grey underneath.
  if (d.z > 0.015) {
    vec2 p = d.xy / (d.z + 0.08);
    float cover = 0.6 * fbm(vec3(p * 0.55, 1.7)) + 0.4 * fbm(vec3(p * 1.7, 4.2));
    float cloud = smoothstep(0.5, 0.72, cover) * smoothstep(0.015, 0.16, d.z);
    float lit = 0.5 + 0.5 * fbm(vec3(p * 0.55 + sun.xy * 0.25, 1.7));
    vec3 white = mix(vec3(0.74, 0.77, 0.82), vec3(1.0, 0.99, 0.96), smoothstep(0.35, 0.75, lit));
    colour = mix(colour, white, 0.88 * cloud);
  }
  colour = mix(colour, below, smoothstep(0.0, -0.12, d.z));
  // The sun: a small bright disc, a glow around it and a wide warm haze.
  float towards = max(dot(d, sun), 0.0);
  colour += vec3(1.0, 0.95, 0.85) * (0.22 * pow(towards, 24.0) + 0.5 * pow(towards, 400.0) + 2.5 * smoothstep(0.99985, 0.99996, towards));
  gl_FragColor = vec4(colour, 1.0);
  #include <colorspace_fragment>
}
`

/** A sky behind everything: pale at the horizon, blue overhead, ground haze below the horizon. */
export function Sky() {
  const invalidate = useThree((s) => s.invalidate)
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: SKY_VERTEX,
        fragmentShader: SKY_FRAGMENT,
        uniforms: {
          horizon: { value: new Color(HORIZON) },
          zenith: { value: new Color(ZENITH) },
          below: { value: new Color(BELOW) },
          sun: { value: new Vector3(...SUN).normalize() },
        },
        side: BackSide,
        depthTest: false,
        depthWrite: false,
      }),
    [],
  )
  useEffect(() => {
    invalidate()
    return () => {
      material.dispose()
      invalidate()
    }
  }, [material, invalidate])
  return (
    <mesh material={material} renderOrder={-1000} frustumCulled={false}>
      <sphereGeometry args={[1, 32, 16]} />
    </mesh>
  )
}

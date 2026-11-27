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
 * - drone-level detail: the picture resolves about a metre, so everything finer is made
 *   here and drawn at every viewing height, each detail only while it covers a few pixels:
 *   tone drifting over metres, patches of gravel and sand, grain down to centimetres, dark
 *   and pale stones, and plants (soft irregular cover, no outlines, with their
 *   shadows, grass tufts), dense where the picture is green or dark and sparse on bare
 *   ground. The same details are a height that the sun models;
 * - fresh rock: ground that was cut (the bench face and floor) or broken (the muckpile) is
 *   drawn as procedural rock (bands along the bedding, mottling, grit) in the vertex colour, not as the picture
 *   of the old surface;
 * - haze: colours fade to the horizon colour with distance, 1 - exp(-d / HAZE_LENGTH).
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { BackSide, Color, DoubleSide, ShaderMaterial, SRGBColorSpace, Vector2, Vector3, Vector4, type Plane, type Texture } from 'three'

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

/** Shade under the clouds: soft darker patches a few hundred metres across, fixed on the ground. */
const CLOUD_SHADE = /* glsl */ `
float cloudShade(vec2 xy) {
  float cover = 0.6 * fbm(vec3(xy * 0.0017, 1.7)) + 0.4 * fbm(vec3(xy * 0.0052, 4.2));
  return 1.0 - 0.2 * smoothstep(0.5, 0.7, cover);
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
${CLOUD_SHADE}
void main() {
  #include <clipping_planes_fragment>
  vec3 n = normalize(vNormal);
  if (n.z < 0.0) n = -n;
  vec3 dpdx = dFdx(vPos);
  vec3 dpdy = dFdy(vPos);

  // The picture. Seen close, its colours change in soft blurs a metre wide; real ground
  // cover ends in ragged edges. The place the picture is read at is therefore shifted by up
  // to a metre with two noises, which breaks every soft edge into a ragged one.
  float fp0 = max(length(dpdx), length(dpdy));
  float uvPerMetre = min(length(fwidth(vUv)) / max(length(fwidth(vPos.xy)), 1e-6), 2.0 * texel.x);
  float ragged = smoothstep(1.5 * fp0, 5.0 * fp0, 1.2);
  vec2 shift = (vec2(vnoise(vPos * 1.7 + 3.0), vnoise(vPos * 1.7 + 11.0)) - 0.5) * 0.4
             + (vec2(vnoise(vPos * 0.45 + 5.0), vnoise(vPos * 0.45 + 17.0)) - 0.5) * 0.55;
  vec2 uv = vUv + shift * uvPerMetre * ragged;
  vec3 base = texture2D(map, uv).rgb;
  // Unsharp mask: the picture minus its local mean, added back, makes edges crisper.
  vec3 mean = 0.25 * (texture2D(map, uv + vec2(texel.x, 0.0)).rgb + texture2D(map, uv - vec2(texel.x, 0.0)).rgb
                    + texture2D(map, uv + vec2(0.0, texel.y)).rgb + texture2D(map, uv - vec2(0.0, texel.y)).rgb);
  base = clamp(base + sharpen * (0.35 + 0.65 * ragged) * (base - mean), 0.0, 1.0);
  // A satellite looks through the whole atmosphere: its picture has a pale bluish veil and
  // flat colours. A drone at a hundred metres has neither: the veil is taken off, and
  // contrast and colour are raised.
  base = max(base - vec3(0.006, 0.009, 0.016), 0.0) * 1.04;
  float lum0 = dot(base, vec3(0.299, 0.587, 0.114));
  base = clamp(mix(vec3(lum0), base, 1.04), 0.0, 1.0);
  base = clamp((base - 0.2) * 1.02 + 0.2, 0.0, 1.0);

  // ---- Drone-level detail -------------------------------------------------------------
  // The satellite picture gives the colour of the ground at about a metre; everything finer
  // is made here. A detail of size l (metres) is drawn only while it covers a few pixels
  // (SEEN), so the ground stays crisp at every height without shimmering.
  float fp = max(length(dpdx), length(dpdy)); // metres of ground per pixel
  #define SEEN(l) smoothstep(1.5 * fp, 5.0 * fp, l)
  float lum = dot(base, vec3(0.299, 0.587, 0.114));
  // Where the picture is greener or darker than bare soil, plants grow.
  float veg = clamp(0.75 * smoothstep(-0.004, 0.03, base.g - 0.5 * (base.r + base.b)) + 0.55 * smoothstep(0.34, 0.16, lum), 0.0, 1.0);

  // Bare ground: tone drifting over metres, patches of gravel and of sand, grain down to
  // centimetres, dark and pale stones.
  float tone = 0.18 * (vnoise(vPos * 0.125 + 1.3) - 0.5) * SEEN(8.0) + 0.16 * (vnoise(vPos * 0.34 + 7.1) - 0.5) * SEEN(3.0);
  float soil = fbm(vPos * 0.23 + 3.7);
  float gravel = smoothstep(0.52, 0.66, soil) * SEEN(4.0) * (1.0 - veg);
  float sand = smoothstep(0.48, 0.34, soil) * SEEN(4.0) * (1.0 - veg);
  float grain = (vnoise(vPos * 2.2) - 0.5) * SEEN(0.45) + 0.8 * (vnoise(vPos * 6.5) - 0.5) * SEEN(0.15) + 0.6 * (vnoise(vPos * 17.0) - 0.5) * SEEN(0.06);
  float darkStone = smoothstep(0.7, 0.92, vnoise(vPos * 4.5 + 5.1)) * SEEN(0.22) * (1.0 - veg);
  float paleStone = smoothstep(0.72, 0.94, vnoise(vPos * 7.5 + 9.4)) * SEEN(0.13) * (0.35 + gravel) * (1.0 - veg);
  base *= 1.0 + tone;
  // From a few hundred metres the fine grain is too small to see and the magnified picture
  // is all soft blots, like a watercolour. Texture one to three metres across, irregular and
  // weak, fills that range the way real ground does.
  float middle = 0.14 * (fbm(vPos * 0.2 + 4.0) - 0.5) * SEEN(4.5) + 0.24 * (fbm(vPos * 0.55 + 1.0) - 0.5) * SEEN(1.6) + 0.18 * (fbm(vPos * 1.15 + 2.0) - 0.5) * SEEN(0.8);
  base *= 1.0 + middle;
  base = mix(base, vec3(lum) * vec3(0.98, 0.96, 0.93), 0.35 * gravel);
  base = mix(base, base * vec3(1.1, 1.06, 0.98), 0.5 * sand);
  base *= 1.0 + 0.17 * grain * (1.0 - 0.4 * veg);
  base *= 1.0 - 0.08 * darkStone;
  base = mix(base, vec3(0.62, 0.6, 0.56), 0.1 * paleStone);

  // Plants, as they look from above: no outlines, only a ground that is greener, darker and
  // more broken where the picture says plants grow. Irregular clumps come from fractal noise
  // read through a second noise (so no shape repeats and none is round); they are blended in
  // softly and stay low, so the sun does not emboss them.
  vec3 bent = vPos + 0.9 * vec3(vnoise(vPos * 0.8 + 4.1) - 0.5, vnoise(vPos * 0.8 + 12.6) - 0.5, 0.0);
  float stand = 0.5 + smoothstep(0.3, 0.75, fbm(vPos * 0.045 + 8.8));
  float cover = clamp(mix(0.02, 0.5, veg) * stand, 0.0, 0.7); // share of the ground under plants
  float clumpNoise = 0.6 * fbm(bent * 1.3 + 2.2) + 0.4 * vnoise(bent * 4.2 + 6.0);
  float shrub = smoothstep(0.66 - 0.32 * cover, 0.8 - 0.32 * cover, clumpNoise) * SEEN(0.45);
  float tuft = smoothstep(0.5, 0.78, vnoise(bent * 9.0 + 4.4)) * SEEN(0.12) * (0.15 + 0.85 * veg) * (1.0 - shrub);
  float shrubShadow = smoothstep(0.66 - 0.32 * cover, 0.8 - 0.32 * cover, 0.6 * fbm((bent - vec3(sun.xy, 0.0) * 0.25) * 1.3 + 2.2) + 0.4 * vnoise((bent - vec3(sun.xy, 0.0) * 0.25) * 4.2 + 6.0))
                     * SEEN(0.45) * (1.0 - shrub);
  vec3 leaf = mix(vec3(0.1, 0.115, 0.055), vec3(0.2, 0.2, 0.1), vnoise(vPos * 5.0 + 1.9));
  // The satellite renders plants in a bluish green; dry steppe is olive.
  base = mix(base, lum * vec3(1.04, 1.07, 0.72), 0.38 * veg);
  // From far above, bushes are small dark specks with a shadow beside them, and that speckle
  // is what makes ground look real at that height. They are irregular (noise read through
  // noise), a metre or two across, follow the plant cover, and fade out when the camera
  // comes close, where separate shapes would look drawn.
  float far = smoothstep(0.1, 0.26, fp);
  float seenSpeck = smoothstep(0.6 * fp, 1.8 * fp, 1.3);
  vec3 drift = vPos + 1.4 * vec3(vnoise(vPos * 0.5 + 3.0) - 0.5, vnoise(vPos * 0.5 + 9.0) - 0.5, 0.0);
  float limit = mix(0.72, 0.5, clamp(1.6 * cover, 0.0, 1.0));
  float speck = smoothstep(limit, limit + 0.07, 0.55 * fbm(drift * 0.85 + 6.3) + 0.45 * vnoise(drift * 2.3 + 1.1)) * far * seenSpeck;
  vec3 beside = drift - vec3(sun.xy, 0.0) * 0.6;
  float speckShadow = smoothstep(limit, limit + 0.07, 0.55 * fbm(beside * 0.85 + 6.3) + 0.45 * vnoise(beside * 2.3 + 1.1)) * far * seenSpeck * (1.0 - speck);
  base = mix(base, mix(base * vec3(0.62, 0.7, 0.45), leaf * (0.8 + 1.1 * lum), 0.45), 0.4 * shrub);
  base *= 1.0 - 0.14 * shrubShadow;
  base = mix(base, leaf * (0.75 + 1.1 * lum), 0.62 * speck);
  base *= 1.0 - 0.2 * speckShadow;
  base = mix(base, base * vec3(0.88, 0.93, 0.78), 0.14 * tuft);

  // All of it as a height in metres, so the sun models it: the picture's own brightness
  // (weak), a little grain and the plant cover. Small stones and tufts are colour only: lit as
  // relief, each one turns into a dark crescent and the ground looks drawn.
  // The picture's own brightness is read as relief only up close: from afar it embosses
  // every blot of the picture.
  float h = lum * bump * SEEN(1.2) + 0.025 * grain + 0.04 * shrub + 0.05 * middle;
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

  // The floor of a cut is not a clean slab: irregular patches of rubble and chips lie on
  // grit and dust that changes tone over a few metres. No shape is round or outlined.
  float level = smoothstep(0.6, 0.88, facet.z);
  vec3 swept = vPos + 0.7 * vec3(vnoise(vPos * 1.1 + 2.0) - 0.5, vnoise(vPos * 1.1 + 8.0) - 0.5, 0.0);
  float rubble = smoothstep(0.56, 0.74, 0.6 * fbm(swept * 1.6 + 5.5) + 0.4 * vnoise(swept * 5.0 + 1.2)) * SEEN(0.4) * level;
  float chip = smoothstep(0.62, 0.8, vnoise(swept * 11.0 + 3.3)) * SEEN(0.1) * level;
  float dust = fbm(vec3(vPos.xy * 0.16, 3.1));
  float fineGrit = (vnoise(vPos * 6.5) - 0.5) * SEEN(0.15) + 0.7 * (vnoise(vPos * 18.0) - 0.5) * SEEN(0.055);
  stone *= 0.84 + 0.34 * smoothstep(0.3, 0.7, dust) * SEEN(6.0);
  stone *= 1.0 + 0.14 * fineGrit;
  stone *= 1.0 + 0.16 * (rubble - 0.4) * (0.5 + vnoise(swept * 2.4));
  stone *= 1.0 + 0.07 * (chip - 0.3);

  vec3 colour = base;
  // The rock itself is rough: blocks a metre or two across with sharper small breaks, read
  // as a height and lit like the fine relief of the picture.
  float hr = 0.45 * (1.0 - 0.6 * level) * fbm(vec3(vPos.xy * 0.55, vPos.z * 0.9)) + 0.12 * vnoise(vPos * 2.6) + 0.04 * vnoise(vPos * 9.0)
           + 0.012 * rubble + 0.008 * fineGrit;
  vec3 rockGradient = sign(det) * (dFdx(hr) * r1 + dFdy(hr) * r2);
  vec3 rough = normalize(abs(det) * normalize(mix(n, facet, 0.6)) - rockGradient);
  stone *= 0.86 + 0.28 * smoothstep(0.25, 0.75, fbm(vec3(vPos.xy * 0.55, vPos.z * 0.9)));
  vec3 nn = normalize(mix(ns, rough, vRock));
  colour = mix(colour, stone, vRock);
  colour *= clamp(1.0 + 0.85 * (dot(nn, sun) - sun.z), 0.5, 1.35);
  // A slightly firmer picture: more contrast about mid grey and a little more colour.
  float grey = dot(colour, vec3(0.299, 0.587, 0.114));
  colour = mix(vec3(grey), colour, 1.03);
  colour = clamp((colour - 0.18) * 1.03 + 0.18, 0.0, 1.0);
  colour *= cloudShade(vPos.xy);
  colour = mix(colour, haze, 1.0 - exp(-vDist / hazeLength));
  // Sensor grain, per pixel of the screen: a photograph is never perfectly smooth.
  colour *= 1.0 + 0.035 * (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5);
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
          bump: { value: 0.45 },
          texel: { value: new Vector2(1 / 2048, 1 / 2048) },
          sharpen: { value: 0.55 },
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

const FAR_VERTEX = /* glsl */ `
varying vec3 vPos;
varying vec3 vNormal;
varying float vDist;
void main() {
  vPos = position;
  vNormal = normal;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vDist = length(mvPosition.xyz);
  gl_Position = projectionMatrix * mvPosition;
}
`

const FAR_FRAGMENT = /* glsl */ `
uniform vec3 ground;
uniform vec3 sun;
uniform vec3 haze;
uniform float hazeLength;
uniform vec4 inner;   // xmin, ymin, xmax, ymax of the detailed terrain: not drawn there
varying vec3 vPos;
varying vec3 vNormal;
varying float vDist;
${NOISE}
${CLOUD_SHADE}
void main() {
  if (vPos.x > inner.x && vPos.x < inner.z && vPos.y > inner.y && vPos.y < inner.w) discard;
  vec3 n = normalize(vNormal);
  if (n.z < 0.0) n = -n;
  // The mean colour of the satellite picture, in broad paler and greener patches.
  float tone = fbm(vec3(vPos.xy * 0.0021, 2.3));
  float fine = fbm(vec3(vPos.xy * 0.012, 6.1));
  vec3 colour = ground * mix(vec3(0.9, 0.94, 0.86), vec3(1.12, 1.06, 0.98), smoothstep(0.3, 0.7, tone)) * (0.9 + 0.2 * fine);
  colour *= clamp(1.0 + 1.1 * (dot(n, sun) - sun.z), 0.5, 1.35);
  colour *= cloudShade(vPos.xy);
  colour = mix(colour, haze, 1.0 - exp(-vDist / hazeLength));
  gl_FragColor = vec4(colour, 1.0);
  #include <colorspace_fragment>
}
`

/** Mean colour of a mosaic canvas (linear), for ground beyond the picture. */
function meanColour(texture: Texture): Color {
  const out = new Color('#a39478')
  try {
    const small = document.createElement('canvas')
    small.width = 8
    small.height = 8
    const context = small.getContext('2d')
    if (!context) return out
    context.drawImage(texture.image as CanvasImageSource, 0, 0, 8, 8)
    const data = context.getImageData(0, 0, 8, 8).data
    let r = 0
    let g = 0
    let b = 0
    for (let i = 0; i < data.length; i += 4) {
      r += data[i]
      g += data[i + 1]
      b += data[i + 2]
    }
    const n = data.length / 4
    out.setRGB(r / n / 255, g / n / 255, b / n / 255, SRGBColorSpace)
  } catch {
    // The canvas cannot be read back: the default colour stays.
  }
  return out
}

/**
 * Material of the far terrain around the detailed one: real ground shape (the elevation
 * model at a coarse spacing) without a picture. It has the mean colour of the satellite
 * picture in broad patches, the same relief shading, cloud shade and haze, and is not drawn
 * inside `inner` (xmin, ymin, xmax, ymax in scene coordinates), where the detailed terrain is.
 */
export function FarTerrainMaterial({ texture, inner }: { texture: Texture; inner: readonly [number, number, number, number] }) {
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: FAR_VERTEX,
        fragmentShader: FAR_FRAGMENT,
        uniforms: {
          ground: { value: new Color() },
          sun: { value: new Vector3(...SUN).normalize() },
          haze: { value: new Color(HORIZON) },
          hazeLength: { value: HAZE_LENGTH },
          inner: { value: new Vector4() },
        },
        side: DoubleSide,
      }),
    [],
  )
  useEffect(() => () => material.dispose(), [material])
  const ground = useMemo(() => meanColour(texture), [texture])
  material.uniforms.ground.value.copy(ground)
  material.uniforms.inner.value.set(inner[0], inner[1], inner[2], inner[3])
  return <primitive object={material} attach="material" />
}

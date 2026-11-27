/**
 * Data colours: sequential ramps, the categorical palette and the "no value" colour.
 *
 * Everything here is sRGB in 0..1 (what CSS and the legend use). GPU buffers need
 * linear values: convert with `srgbToLinear` when filling a colour attribute (the
 * canvas is `flat`, so a buffer colour then appears on screen exactly as its legend
 * swatch).
 *
 * Ramps are per theme because one ramp cannot contrast with both a near-black and a
 * near-white canvas: each runs low = cool and dark to high = warm and bright with
 * strictly increasing lightness, and both of its ends keep at least 2:1 contrast
 * against that theme's canvas (ramps.test.ts checks both properties).
 */
import type { Theme } from '../state/store'

export type Rgb = readonly [number, number, number]
export type RampName = 'grade' | 'elevation'

export function hexToRgb(hex: string): Rgb {
  const n = parseInt(hex.slice(1), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

export function rgbToHex(rgb: Rgb): string {
  const byte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255)
  return '#' + ((1 << 24) | (byte(rgb[0]) << 16) | (byte(rgb[1]) << 8) | byte(rgb[2])).toString(16).slice(1)
}

export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

/** WCAG relative luminance of an sRGB colour. */
export function luminance(rgb: Rgb): number {
  return 0.2126 * srgbToLinear(rgb[0]) + 0.7152 * srgbToLinear(rgb[1]) + 0.0722 * srgbToLinear(rgb[2])
}

/** WCAG contrast ratio, 1..21. */
export function contrast(a: Rgb, b: Rgb): number {
  const la = luminance(a)
  const lb = luminance(b)
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

/** OKLab lightness, 0..1: the perceptual quantity a sequential ramp must order by. */
export function oklabLightness(rgb: Rgb): number {
  const r = srgbToLinear(rgb[0])
  const g = srgbToLinear(rgb[1])
  const b = srgbToLinear(rgb[2])
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s
}

/** Ramp stops, low to high, evenly spaced. */
export const RAMP_STOPS: Record<RampName, Record<Theme, readonly string[]>> = {
  // Grades: blue -> teal -> green -> yellow-green -> yellow.
  grade: {
    dark: ['#4a66d9', '#2e93c9', '#2fb38a', '#a3c540', '#ffdc4a'],
    light: ['#27348f', '#1c6da6', '#198a6a', '#7d9a1c', '#d39a00'],
  },
  // Elevation: deep green -> olive -> tan -> pale; reads as terrain, distinct from grades.
  elevation: {
    dark: ['#3f7a5a', '#6f9a58', '#a8ad62', '#d3bd84', '#f1e3c4'],
    light: ['#1f5a40', '#4f7a38', '#8a8a3a', '#b3915a', '#cfa98a'],
  },
}

/** Colour at `t` in 0..1 (clamped), interpolated between the stops in sRGB. */
export function rampColor(name: RampName, theme: Theme, t: number): Rgb {
  const stops = RAMP_STOPS[name][theme]
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(x))
  const a = hexToRgb(stops[i])
  const b = hexToRgb(stops[i + 1])
  const f = x - i
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** CSS gradient of a ramp, low at the start. */
export function rampGradient(name: RampName, theme: Theme, direction = 'to right'): string {
  return `linear-gradient(${direction}, ${RAMP_STOPS[name][theme].join(', ')})`
}

/**
 * Categorical colours, 20 per theme, assigned to categories in list order.
 *
 * Entries 1-8 are the original palette (checked for colour-blind separation with the
 * dataviz validator) and are unchanged, so an attribute with up to eight categories
 * looks as it always did. Entries 9-20 were added for rock units by greedy
 * farthest-point selection in CIELAB (D65): from the sRGB colours on a 5/255 grid with
 * chroma 22..72, lightness 30..82 (light theme) or 42..88 (dark theme), contrast of at
 * least 2.2 against the theme's canvas and CIE76 distance of at least 20 from the
 * "other", "no value" and neutral block greys, take the one farthest from everything
 * already chosen, twelve times. The dark entry keeps the hue (within 18 degrees) of the
 * light entry with the same index, so a unit reads as the same colour in both themes.
 *
 * Measured (colour.test.ts): CIE76 distance >= 30 between list neighbours and between
 * any pair that involves an added entry; >= 20 between any pair at all (the limit is
 * set by two pairs of the original eight). Twenty colours cannot all be told apart
 * with a colour-vision deficiency: only the first eight were validated for that.
 *
 * More than 20 categories: the palette cycles (category i gets entry i mod 20), so
 * colours repeat from the 21st category on; give the data a colour map to avoid it.
 */
export const CATEGORICAL: Record<Theme, readonly string[]> = {
  light: [
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948',
    '#55461e', '#2db4d2', '#5f4164', '#a5aa55', '#9b0082', '#005555', '#d79678', '#aaa0cd',
    '#e182f0', '#910541', '#7d2d00', '#05551e',
  ],
  dark: [
    '#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#0ca30c', '#9085e9', '#e66767',
    '#ebd7af', '#3cdcff', '#ffcdfa', '#d7e655', '#7d5a78', '#196e78', '#875f37', '#9141b4',
    '#ff8ceb', '#d29696', '#ffaa73', '#41faa0',
  ],
}

/** A category index that is not one (negative, NaN): a neutral grey outside the palette. */
export const OTHER_COLOR: Record<Theme, string> = { light: '#6f6e69', dark: '#a3a29b' }

/** Intervals without a value: a neutral grey, clearly not on any ramp. */
export const MISSING_COLOR: Record<Theme, string> = { light: '#9a9c9a', dark: '#6b7178' }

/** Palette colour of category `index`; cycles past the end (see `CATEGORICAL`). */
export function categoryColor(theme: Theme, index: number): string {
  const palette = CATEGORICAL[theme]
  return Number.isFinite(index) && index >= 0 ? palette[Math.floor(index) % palette.length] : OTHER_COLOR[theme]
}

/**
 * A colour supplied by the data, as lower-case `#rrggbb`; null when it is absent or not
 * in that form (the palette is used instead).
 */
export function explicitColor(colours: Readonly<Record<string, string>> | undefined, category: string): string | null {
  if (!colours || !Object.prototype.hasOwnProperty.call(colours, category)) return null
  const value = colours[category]
  return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value) ? value.toLowerCase() : null
}

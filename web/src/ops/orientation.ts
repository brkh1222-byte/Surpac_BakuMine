/**
 * Orientation entered as a geologist measures a plane: dip, dip azimuth and (optionally) the
 * pitch of the long axis in the plane. Mirror of `bakumine/core/estimation/orientation.py`
 * (the server does the conversion that is used; these helpers show the user what the input
 * means and are checked against the same numbers as the Python tests).
 *
 *   minor axis      = pole of the plane, pointing up
 *   pitch 0         : major axis horizontal along strike (dip azimuth - 90), semi-major
 *                     straight down dip
 *   pitch p         : the major axis turns in the plane from that strike direction towards
 *                     down dip (90 = down dip); the semi-major axis stays perpendicular to it
 *
 * Vectors are (east, north, up); angles in degrees.
 */

export type Vec3 = [number, number, number]

export interface PlaneInput {
  dip: number
  dip_azimuth: number
  pitch: number
}

const RAD = Math.PI / 180

/** Parses the three text fields; an error message when they are not a valid plane. */
export function parsePlane(dip: string, dipAzimuth: string, pitch: string): PlaneInput | string {
  if (dip.trim() === '' || dipAzimuth.trim() === '') return 'Give the dip and the dip azimuth'
  const d = Number(dip)
  const a = Number(dipAzimuth)
  const p = pitch.trim() === '' ? 0 : Number(pitch)
  if (!Number.isFinite(d) || d < 0 || d > 90) return 'Dip: a number from 0 to 90'
  if (!Number.isFinite(a)) return 'Dip azimuth: a number (degrees clockwise from north)'
  if (!Number.isFinite(p)) return 'Pitch: a number (degrees from strike, in the plane)'
  return { dip: d, dip_azimuth: a, pitch: p }
}

/** Unit vectors of the major, semi-major and minor axes of a plane. */
export function planeAxes(plane: PlaneInput): { major: Vec3; semiMajor: Vec3; minor: Vec3 } {
  const a = plane.dip_azimuth * RAD
  const d = plane.dip * RAD
  const p = plane.pitch * RAD
  const downDip: Vec3 = [Math.sin(a) * Math.cos(d), Math.cos(a) * Math.cos(d), -Math.sin(d)]
  const strike: Vec3 = [-Math.cos(a), Math.sin(a), 0]
  const pole: Vec3 = [Math.sin(a) * Math.sin(d), Math.cos(a) * Math.sin(d), Math.cos(d)]
  const mix = (s: number, t: number): Vec3 => [s * strike[0] + t * downDip[0], s * strike[1] + t * downDip[1], s * strike[2] + t * downDip[2]]
  return { major: mix(Math.cos(p), Math.sin(p)), semiMajor: mix(-Math.sin(p), Math.cos(p)), minor: pole }
}

const tidy = (value: number) => {
  const rounded = Math.round(value * 1e6) / 1e6
  return rounded === 0 ? 0 : rounded
}

/** (azimuth, plunge) of an axis at its downward end; a horizontal axis reports 0 - 180. */
export function trendPlunge(axis: Vec3): { azimuth: number; plunge: number } {
  let [x, y, z] = axis
  const eps = 1e-12
  if (z > eps || (Math.abs(z) <= eps && (x < -eps || (Math.abs(x) <= eps && y < 0)))) [x, y, z] = [-x, -y, -z]
  const plunge = Math.asin(Math.min(1, Math.max(-1, -z))) / RAD
  const azimuth = plunge > 90 - 1e-9 ? 0 : ((Math.atan2(x, y) / RAD) % 360 + 360) % 360
  return { azimuth: tidy(azimuth) % 360, plunge: tidy(plunge) }
}

/**
 * (bearing, plunge, dip) of the project's rotation convention (`surpac_zxy_lrl` in
 * bakumine/core/estimation/rotation.py) that reproduce the axes of the plane.
 */
export function planeAngles(plane: PlaneInput): [number, number, number] {
  const { major, semiMajor, minor } = planeAxes(plane)
  const plunge = Math.asin(Math.min(1, Math.max(-1, major[2])))
  let bearing: number
  let dip: number
  if (Math.abs(Math.cos(plunge)) < 1e-9) {
    dip = 0
    bearing = -Math.atan2(semiMajor[1], semiMajor[0])
  } else {
    bearing = Math.atan2(major[0], major[1])
    dip = Math.atan2(semiMajor[2], minor[2])
  }
  const b = ((bearing / RAD) % 360 + 360) % 360
  return [tidy(b) % 360, tidy(plunge / RAD), tidy(dip / RAD)]
}

const deg = (value: number, digits = 3) => String(Math.round(value)).padStart(digits, '0')

/** One line for the dialogs: what the plane means as ellipsoid axes. */
export function describePlane(plane: PlaneInput): string {
  const axes = planeAxes(plane)
  const part = (label: string, axis: Vec3) => {
    const t = trendPlunge(axis)
    return `${label} ${deg(t.azimuth)}/${deg(t.plunge, 2)}`
  }
  return `Axes (azimuth/plunge): ${part('major', axes.major)}, ${part('semi-major', axes.semiMajor)}, ${part('minor', axes.minor)}`
}

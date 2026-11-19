/** One-line description of a section: "N 5 300 650 · looking north · ±12.5 m". */
import { formatCoord } from '../scene/format'
import { compass, lookAzimuth, sectionKind, type SectionPlane, type Slab } from './plane'

function trim(value: number): string {
  return String(Math.round(value * 100) / 100)
}

export function describeThickness(slab: Slab): string {
  return slab.front === slab.back ? `±${trim(slab.front)} m` : `+${trim(slab.front)} / −${trim(slab.back)} m`
}

export function describeSection(plane: SectionPlane, slab: Slab): string {
  const [e, n, z] = plane.centre
  const thickness = describeThickness(slab)
  const looking = `looking ${compass(lookAzimuth(plane))}`
  switch (sectionKind(plane)) {
    case 'we':
      return `N ${formatCoord(n, n % 1 === 0 ? 0 : 1)} · ${looking} · ${thickness}`
    case 'sn':
      return `E ${formatCoord(e, e % 1 === 0 ? 0 : 1)} · ${looking} · ${thickness}`
    case 'plan':
      return `Z ${formatCoord(z, z % 1 === 0 ? 0 : 1)} · plan · ${thickness}`
    case 'vertical':
      return `Az ${trim(plane.azimuth)}° through E ${formatCoord(e, 0)} N ${formatCoord(n, 0)} · ${looking} · ${thickness}`
    default:
      return `Az ${trim(plane.azimuth)}° dip ${trim(plane.dip)}° through E ${formatCoord(e, 0)} N ${formatCoord(n, 0)} Z ${formatCoord(z, 0)} · ${thickness}`
  }
}

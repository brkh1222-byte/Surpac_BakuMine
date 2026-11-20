/**
 * Block model: every block of the current selection as a true-size box (sub-blocks at
 * their own size), coloured by one attribute.
 *
 * Drawing: ONE instanced draw call of a unit cube with a custom shader. Per block the
 * GPU holds the centre (scene coordinates, float32), the level (which gives the size
 * from a small uniform table) and one float: the attribute value. The colour comes from
 * a 256-texel table, so changing the legend range, ramp / classes, theme, edges, opacity
 * or the shrink factor only rewrites uniforms (blocks/colour.ts). A new filter or colour
 * attribute replaces the instance buffers, which are compact (17 bytes per block).
 *
 * Which blocks are here is decided by the server (`where`, level of detail: the exposed
 * shell for large selections; bakumine/api/blocks.py), so hidden interior faces are
 * mostly never sent. Of each block only the three faces turned towards the eye are
 * drawn (the vertex shader mirrors a three-face geometry): half the vertices of a cube.
 *
 * The model rotation is a uniform; the vertical exaggeration is the content group's
 * scale; the section slab is three's clipping planes (`clipping: true`), applied by
 * <SceneContent> like for every other layer. While a section is on, whole cubes are
 * drawn and the fragment shader closes every cut block with its cut face: for a back
 * face it finds the stretch of the view ray inside the block AND the slab, so a block is
 * shown wherever the slab passes through it - also a slab thinner than the block.
 * Blocks are cut at the section plane itself (where the mesh outlines are), so of the
 * slab they fill the part behind the plane.
 *
 * Picking is analytic (blocks/pick.ts): the pointer ray walks the parent grid in the
 * model frame. No three.js raycast, no id buffer.
 */
import { useThree } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo } from 'react'
import {
  BoxGeometry,
  BufferGeometry,
  Color,
  DataTexture,
  DoubleSide,
  Float32BufferAttribute,
  EdgesGeometry,
  Group,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  LinearFilter,
  Matrix3,
  Matrix4,
  Mesh,
  NoColorSpace,
  RGBAFormat,
  ShaderMaterial,
  Sphere,
  Vector3,
  Vector4,
} from 'three'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { fetchBlockDetail, type BlockSet } from '../api/blocks'
import {
  blockLegend,
  countMissing,
  MISSING_BELOW,
  rampTexture,
  shaderColouring,
  TABLE_SIZE,
  valueBuffer,
} from '../blocks/colour'
import { buildPickIndex, pickBlock } from '../blocks/pick'
import { blockGeometry, blockId, levelSizes, sceneCentres, worldToLocal } from '../blocks/structure'
import { hexToRgb, MISSING_COLOR } from '../colour/ramps'
import { INK } from '../drillholes/colouring'
import { useBlocks } from '../state/blocks'
import { useDisplay } from '../state/display'
import { frameOf } from '../section/plane'
import { useSection } from '../state/section'
import { useViewer } from '../state/store'
import { SECTION_PLANES, sectionClipActive, updateSectionPlanes } from './clipping'
import { useLayerObject, type LayerObject } from './layers'
import { view, type Picker } from './view'

const MAX_LEVELS = 9

const VERTEX = /* glsl */ `
#include <common>
#include <clipping_planes_pars_vertex>
attribute vec3 aCentre;
attribute float aValue;
attribute float aLevel;
uniform mat3 uRotation;
uniform vec3 uSizes[${MAX_LEVELS}];
uniform float uShrink;
uniform float uShowMissing;
uniform vec4 uEye;
varying float vValue;
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewNormal;
varying vec3 vViewPosition;
#if NUM_CLIPPING_PLANES > 0
varying vec3 vCube;
varying vec3 vCubeRay;
#endif
void main() {
  vec3 size = uSizes[int(aLevel + 0.5)] * uShrink;
  vec3 corner = position;
  vec3 faceNormal = normal;
  #ifdef THREE_FACES
    // The geometry is the +i, +j and +k faces only; each is moved to the side of the
    // block the eye is on (uEye: scene-space eye position, or direction for w = 0).
    vec3 toEye = uEye.w > 0.5 ? uEye.xyz - aCentre : uEye.xyz;
    vec3 side = vec3(
      dot(uRotation[0], toEye) < 0.0 ? -1.0 : 1.0,
      dot(uRotation[1], toEye) < 0.0 ? -1.0 : 1.0,
      dot(uRotation[2], toEye) < 0.0 ? -1.0 : 1.0
    );
    corner *= side;
    faceNormal *= side;
  #endif
  vec3 scenePosition = aCentre + uRotation * (corner * size);
  vec4 mvPosition = modelViewMatrix * vec4(scenePosition, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  // Blocks without a value, when they are not shown: moved out of the clip volume.
  if (aValue < ${MISSING_BELOW.toExponential(1)} && uShowMissing < 0.5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  vValue = aValue;
  vUv = uv;
  vNormal = uRotation * faceNormal;
  vViewNormal = normalMatrix * vNormal;
  vViewPosition = -mvPosition.xyz;
  #include <clipping_planes_vertex>
  #if NUM_CLIPPING_PLANES > 0
    // For the section caps: this vertex in the block's own unit-cube coordinates, and
    // the eye (perspective) or the viewing direction (orthographic) in the same.
    mat3 cubeToView = mat3(modelViewMatrix) * uRotation * mat3(size.x, 0.0, 0.0, 0.0, size.y, 0.0, 0.0, 0.0, size.z);
    mat3 viewToCube = inverse(cubeToView);
    vCube = corner;
    vCubeRay = isOrthographic ? viewToCube * vec3(0.0, 0.0, -1.0) : -(viewToCube * (modelViewMatrix * vec4(aCentre, 1.0)).xyz);
  #endif
}
`

const FRAGMENT = /* glsl */ `
#include <clipping_planes_pars_fragment>
uniform sampler2D uTable;
uniform float uMin;
uniform float uMax;
uniform float uSteps;
uniform float uCategorical;
uniform vec3 uMissingColor;
uniform float uEdges;
uniform float uOpacity;
varying float vValue;
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewNormal;
varying vec3 vViewPosition;
#if NUM_CLIPPING_PLANES > 0
uniform mat4 projectionMatrix;
uniform float uCutShift;
varying vec3 vCube;
varying vec3 vCubeRay;
#endif
void main() {
  vec3 toEye = isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(vViewPosition);
  // Where on the face, in pixels from its border (screen-space derivatives).
  vec2 width = max(fwidth(vUv), vec2(1e-6));
  vec2 border = min(vUv, 1.0 - vUv) / width;
  float borderPixels = min(border.x, border.y);
  float facePixels = min(1.0 / width.x, 1.0 / width.y);
  bool cap = false;
  vec3 capNormal = vec3(0.0, 0.0, 1.0);

  #if NUM_CLIPPING_PLANES > 0
    gl_FragDepth = gl_FragCoord.z;
    if (gl_FrontFacing) {
      for (int i = 0; i < NUM_CLIPPING_PLANES; i++) {
        vec4 plane = clippingPlanes[i];
        // Plane 0 is the slab face towards the viewer: blocks are cut at the section
        // plane itself instead (uCutShift is the slab's front thickness).
        if (i == 0) plane.w -= uCutShift;
        if (dot(vClipPosition, plane.xyz) > plane.w) discard;
      }
    } else {
      // A back face is where the view ray LEAVES the block. Walk back along the ray
      // (parameter t) to where it entered; the part of that stretch inside the slab is
      // what the section shows of this block. If the slab cuts the entry off, the
      // nearest visible point is on a slab plane: the cut face of the block (its cap).
      vec3 rayCube = isOrthographic ? vCubeRay : vCube - vCubeRay;
      vec3 rayView = isOrthographic ? vec3(0.0, 0.0, -1.0) : -vClipPosition;
      // (varyings are extrapolated at the rim of a face: keep the point on the cube)
      vec3 exit = clamp(vCube, -0.5, 0.5);
      // An axis the ray barely moves along cannot be where it entered; dividing by it
      // would turn rounding noise at the block borders into holes.
      vec3 along = abs(rayCube);
      float least = 1e-4 * max(along.x, max(along.y, along.z));
      vec3 entry = vec3(
        along.x > least ? (exit.x + 0.5 * sign(rayCube.x)) / rayCube.x : 1e30,
        along.y > least ? (exit.y + 0.5 * sign(rayCube.y)) / rayCube.y : 1e30,
        along.z > least ? (exit.z + 0.5 * sign(rayCube.z)) / rayCube.z : 1e30
      );
      float tFar = min(entry.x, min(entry.y, entry.z));
      float tNear = 0.0;
      for (int i = 0; i < NUM_CLIPPING_PLANES; i++) {
        vec4 plane = clippingPlanes[i];
        if (i == 0) plane.w -= uCutShift;
        float a = plane.w - dot(vClipPosition, plane.xyz);
        float b = -dot(rayView, plane.xyz);
        if (b > 0.0) {
          tNear = max(tNear, -a / b);
        } else if (b < 0.0) {
          if (-a / b < tFar) {
            tFar = -a / b;
            cap = true;
            capNormal = plane.xyz;
          }
        } else if (a < 0.0) {
          discard;
        }
      }
      if (tNear > tFar) discard;
      if (cap) {
        // The cut face is ON the slab plane, in front of everything the slab shows:
        // give it that depth (not the back face's), so nothing behind it shows through.
        vec4 clip = projectionMatrix * vec4(-vClipPosition - tFar * rayView, 1.0);
        gl_FragDepth = clamp(0.5 * clip.z / clip.w + 0.5, 0.0, 1.0);
        // Border of the cut face: where the cut point nears a face of the block.
        vec3 cut = exit - tFar * rayCube;
        vec3 w = fwidth(cut);
        borderPixels = 1e9;
        facePixels = 1e9;
        for (int k = 0; k < 3; k++) {
          // (an axis the cut does not move along is the one the plane is square to)
          if (w[k] > 1e-5) {
            borderPixels = min(borderPixels, (0.5 - abs(cut[k])) / w[k]);
            facePixels = min(facePixels, 1.0 / w[k]);
          }
        }
      }
    }
  #endif

  vec3 color;
  if (vValue < ${MISSING_BELOW.toExponential(1)}) {
    color = uMissingColor;
  } else {
    float t;
    if (uCategorical > 0.5) {
      t = (min(vValue, ${TABLE_SIZE - 1}.0) + 0.5) / ${TABLE_SIZE}.0;
    } else {
      float span = uMax - uMin;
      float x = span > 0.0 ? clamp((vValue - uMin) / span, 0.0, 1.0) : (vValue >= uMax ? 1.0 : 0.0);
      if (uSteps > 0.5) x = (min(floor(x * uSteps), uSteps - 1.0) + 0.5) / uSteps;
      t = x * ${(TABLE_SIZE - 1) / TABLE_SIZE} + ${0.5 / TABLE_SIZE};
    }
    color = texture2D(uTable, vec2(t, 0.5)).rgb;
  }
  // Flat shading from the face normal, the brighter of two lights. A fixed one from
  // above and the west (like the meshes): top faces are the legend colour itself, the
  // sides darker by fixed amounts. And a headlight: a face seen square-on - every face of
  // a section view, cut faces included - is the legend colour too.
  vec3 n = normalize(vNormal);
  float facing = abs(dot(normalize(cap ? capNormal : vViewNormal), toEye));
  float head = 0.7 + 0.3 * facing * facing;
  float fixedLight = 0.8 + 0.2 * n.z - 0.08 * n.x - 0.03 * n.y;
  #ifdef THREE_FACES
    float shade = max(fixedLight, head);
  #else
    float shade = gl_FrontFacing ? max(fixedLight, head) : head;
  #endif
  // Block edges; faces only a few pixels across would be all edge, so they fade out there.
  float line = 1.0 - smoothstep(0.35, 1.35, borderPixels);
  float edge = line * uEdges * smoothstep(3.0, 8.0, facePixels);
  // The colour table is sRGB and the canvas is not tone mapped: written as is.
  gl_FragColor = vec4(color * shade * (1.0 - 0.5 * edge), uOpacity);
}
`

/**
 * Blocks are cut by the section PLANE, where the mesh outlines are drawn, not by the
 * slab face in front of it: a section then shows the blocks as the plane meets them.
 * This is how far the first slab plane (SECTION_PLANES[0], "distance <= front") has to
 * move back for that, in render-space units.
 */
function sectionCutShift(): number {
  const section = useSection.getState()
  if (!section.enabled) return 0
  const { n } = frameOf(section.plane)
  return section.front / Math.hypot(n[0], n[1], n[2] / useViewer.getState().verticalExaggeration)
}

/** The +i, +j and +k faces of the unit cube: the shader mirrors them towards the eye. */
function threeFaces(): BufferGeometry {
  const position: number[] = []
  const normal: number[] = []
  const uv: number[] = []
  const index: number[] = []
  for (let axis = 0; axis < 3; axis++) {
    const u = (axis + 1) % 3
    const v = (axis + 2) % 3
    const first = position.length / 3
    for (const [a, b] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const p = [0, 0, 0]
      p[axis] = 0.5
      p[u] = a - 0.5
      p[v] = b - 0.5
      position.push(...p)
      const n = [0, 0, 0]
      n[axis] = 1
      normal.push(...n)
      uv.push(a, b)
    }
    index.push(first, first + 1, first + 2, first + 2, first + 1, first + 3)
  }
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(position, 3))
  geometry.setAttribute('normal', new Float32BufferAttribute(normal, 3))
  geometry.setAttribute('uv', new Float32BufferAttribute(uv, 2))
  geometry.setIndex(index)
  return geometry
}

/**
 * `base` (the whole unit cube, or its three faces) with the instance attributes, which
 * the two geometries of a set share: `count` blocks.
 */
function makeGeometry(base: BufferGeometry, instances: Record<string, InstancedBufferAttribute>, count: number, sphere: Sphere): InstancedBufferGeometry {
  const geometry = new InstancedBufferGeometry()
  // Copies, not the shared attributes: disposing a geometry frees the GPU buffers of
  // all its attributes, and the next set's geometry must not lose its cube with them.
  geometry.setIndex(base.getIndex()?.clone() ?? null)
  for (const name of ['position', 'normal', 'uv']) geometry.setAttribute(name, base.getAttribute(name).clone())
  for (const [name, attribute] of Object.entries(instances)) geometry.setAttribute(name, attribute)
  geometry.instanceCount = count
  geometry.boundingSphere = sphere
  return geometry
}

export interface BlocksLayerProps {
  object: LayerObject
  /** Data endpoint of the model (block details are fetched from `${href}/detail`). */
  href: string
  set: BlockSet
  /** `performance.now()` when the request for `set` was sent. */
  requestedAt: number
  /** Time the request took until `set` was decoded, ms. */
  requestMs: number
}

export function BlocksLayer({ object, href, set, requestedAt, requestMs }: BlocksLayerProps) {
  const { ready, visible, origin } = useLayerObject(object)
  const invalidate = useThree((s) => s.invalidate)
  const theme = useViewer((s) => s.theme)
  const model = useBlocks((s) => s.models[object.id])
  const sectionOn = useSection((s) => s.enabled)

  const definition = set.definition
  // The column drawn is the one this set carries (the store may already ask for another).
  const column = useMemo(() => set.columns.find((c) => c.name === model?.colorBy) ?? set.columns[0] ?? null, [set, model?.colorBy])
  const info = model?.attributes.find((a) => a.name === column?.name)
  const settings = column ? model?.settings[column.name] : undefined

  const shapes = useMemo(() => ({ cube: new BoxGeometry(1, 1, 1) as BufferGeometry, faces: threeFaces() }), [])
  useEffect(
    () => () => {
      shapes.cube.dispose()
      shapes.faces.dispose()
    },
    [shapes],
  )

  // ---- instance buffers: rebuilt per set (and per scene origin), never per display change
  const built = useMemo(() => {
    if (!ready) return null
    const start = performance.now()
    const centres = sceneCentres(definition, set.structure, origin)
    const values = valueBuffer(column, set.count)
    const instances = {
      aCentre: new InstancedBufferAttribute(centres, 3),
      aLevel: new InstancedBufferAttribute(set.structure.level, 1),
      aValue: new InstancedBufferAttribute(values, 1),
    }
    // Bounding sphere of the whole model in scene space (the content group scales z).
    const bounds = object.bounds
    const centre = new Vector3(
      (bounds.min[0] + bounds.max[0]) / 2 - origin[0],
      (bounds.min[1] + bounds.max[1]) / 2 - origin[1],
      (bounds.min[2] + bounds.max[2]) / 2 - origin[2],
    )
    const sphere = new Sphere(centre, Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]) / 2 + 1)
    return {
      // Normally three faces per block; all six while a section may cut blocks open.
      faces: makeGeometry(shapes.faces, instances, set.count, sphere),
      cube: makeGeometry(shapes.cube, instances, set.count, sphere),
      centres,
      values,
      missing: countMissing(values),
      pick: buildPickIndex(definition, set.structure),
      buildMs: performance.now() - start,
    }
  }, [ready, origin, set, column, definition, object.bounds, shapes])
  useEffect(
    () => () => {
      built?.faces.dispose()
      built?.cube.dispose()
    },
    [built],
  )

  const spec = useMemo(() => blockLegend(info, settings, (built?.missing ?? 0) > 0), [info, settings, built?.missing])

  // ---- the mesh, its material and the selection outline: created once
  const scene = useMemo(() => {
    const table = new DataTexture(new Uint8Array(TABLE_SIZE * 4), TABLE_SIZE, 1, RGBAFormat)
    table.minFilter = table.magFilter = LinearFilter
    table.colorSpace = NoColorSpace
    table.generateMipmaps = false
    const material = new ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      clipping: true,
      // Mirrored faces change their winding, and cut blocks are seen from inside.
      side: DoubleSide,
      defines: { THREE_FACES: 1 },
      uniforms: {
        uRotation: { value: new Matrix3() },
        uSizes: { value: Array.from({ length: MAX_LEVELS }, () => new Vector3(1, 1, 1)) },
        uShrink: { value: 1 },
        uShowMissing: { value: 0 },
        uTable: { value: table },
        uMin: { value: 0 },
        uMax: { value: 1 },
        uSteps: { value: 0 },
        uCategorical: { value: 0 },
        uMissingColor: { value: new Color() },
        uEdges: { value: 1 },
        uOpacity: { value: 1 },
        uEye: { value: new Vector4(0, 0, 1, 0) },
        uCutShift: { value: 0 },
      },
    })
    const mesh = new Mesh(new BufferGeometry(), material)
    mesh.name = `blocks:${object.id}`
    // Picked by the layer itself; the bounding sphere is the model's, set per geometry.
    mesh.userData.noRaycast = true
    mesh.visible = false
    // Where the eye is, in the mesh's own (scene) space: the shader turns the three
    // faces of each block towards it.
    const inverse = new Matrix4()
    mesh.onBeforeRender = (_renderer, _scene, camera) => {
      const eye = material.uniforms.uEye.value as Vector4
      inverse.copy(mesh.matrixWorld).invert()
      const v = new Vector3()
      if ((camera as { isOrthographicCamera?: boolean }).isOrthographicCamera) {
        // Towards the eye: the camera's own +z axis.
        v.setFromMatrixColumn(camera.matrixWorld, 2).transformDirection(inverse)
        eye.set(v.x, v.y, v.z, 0)
      } else {
        v.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(inverse)
        eye.set(v.x, v.y, v.z, 1)
      }
      material.uniforms.uCutShift.value = sectionCutShift()
      material.uniformsNeedUpdate = true
    }

    const outlineGeometry = new LineSegmentsGeometry()
    outlineGeometry.setPositions(new EdgesGeometry(new BoxGeometry(1, 1, 1)).getAttribute('position').array as Float32Array)
    const outlineMaterial = new LineMaterial({ linewidth: 2.5, depthTest: false, transparent: true })
    const outline = new LineSegments2(outlineGeometry, outlineMaterial)
    outline.name = 'block-highlight'
    outline.renderOrder = 4
    outline.matrixAutoUpdate = false
    outline.frustumCulled = false
    outline.visible = false
    outline.userData.noRaycast = true
    outline.userData.helper = true
    outline.userData.noClip = true

    const group = new Group()
    group.name = `block-model:${object.id}`
    group.add(mesh, outline)
    const dispose = () => {
      material.dispose()
      table.dispose()
      outlineGeometry.dispose()
      outlineMaterial.dispose()
    }
    return { group, mesh, material, table, outline, outlineMaterial, dispose }
  }, [object.id])
  useEffect(() => () => scene.dispose(), [scene])

  // ---- new blocks
  useLayoutEffect(() => {
    if (!built) return
    scene.mesh.geometry = sectionOn ? built.cube : built.faces
    scene.mesh.visible = true
    const timing = { requestMs, buildMs: built.buildMs, toDrawMs: -1, toFrameMs: -1 }
    // The first frame that draws this set: when the draw call is issued, and when the
    // GPU has finished it (a one-pixel read-back in the next task waits for that).
    scene.mesh.onAfterRender = (renderer) => {
      scene.mesh.onAfterRender = () => {}
      timing.toDrawMs = performance.now() - requestedAt
      setTimeout(() => {
        const context = renderer.getContext()
        context.readPixels(0, 0, 1, 1, context.RGBA, context.UNSIGNED_BYTE, new Uint8Array(4))
        timing.toFrameMs = performance.now() - requestedAt
      }, 0)
    }
    const levels = new Set<number>()
    const level = set.structure.level
    for (let i = 0; i < set.count && levels.size < MAX_LEVELS; i++) levels.add(level[i])
    const sizes = levelSizes(definition)
    let maxAbs = 0
    for (let i = 0; i < built.centres.length; i++) maxAbs = Math.max(maxAbs, Math.abs(built.centres[i]))
    view.blocks.set(object.id, {
      count: set.count,
      selected: set.selected,
      total: set.total,
      lod: set.lod,
      clipped: set.clipped,
      attribute: column?.name ?? null,
      sizes: [...levels].sort().map((l) => [sizes[l * 3], sizes[l * 3 + 1], sizes[l * 3 + 2]]),
      maxAbsCentre: maxAbs,
      bytesPerBlock: set.bytesPerBlock,
      timing,
      block: (k) => {
        if (k < 0 || k >= set.count) return null
        const g = blockGeometry(definition, set.structure, k)
        return { row: set.rows ? set.rows[k] : k, centroid: g.centroid, size: g.size, level: g.level }
      },
    })
    invalidate()
    // `sectionOn` is read, not a dependency: the effect below swaps the geometry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, built, set, column, definition, object.id, requestedAt, requestMs, invalidate])
  useEffect(() => () => void view.blocks.delete(object.id), [object.id])

  // ---- display: uniforms only
  const edges = model?.edges ?? true
  const opacity = model?.opacity ?? 1
  const shrink = model?.shrink ?? 1
  const showMissing = (model?.missing ?? 'hidden') === 'neutral'
  useLayoutEffect(() => {
    const u = scene.material.uniforms
    const r = definition.rotation
    ;(u.uRotation.value as Matrix3).set(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
    const sizes = levelSizes(definition)
    const table = u.uSizes.value as Vector3[]
    for (let l = 0; l < MAX_LEVELS; l++) {
      const k = Math.min(l, sizes.length / 3 - 1) * 3
      table[l].set(sizes[k], sizes[k + 1], sizes[k + 2])
    }
  }, [scene, definition])
  useLayoutEffect(() => {
    const u = scene.material.uniforms
    rampTexture(spec, theme, scene.table.image.data as Uint8Array)
    scene.table.needsUpdate = true
    const c = shaderColouring(spec)
    u.uMin.value = c.min
    u.uMax.value = c.max
    u.uSteps.value = c.steps
    u.uCategorical.value = c.categorical ? 1 : 0
    ;(u.uMissingColor.value as Color).setRGB(...hexToRgb(MISSING_COLOR[theme]))
    u.uShowMissing.value = showMissing ? 1 : 0
    u.uEdges.value = edges ? 1 : 0
    u.uOpacity.value = opacity
    u.uShrink.value = shrink
    const material = scene.material
    const transparent = opacity < 1
    // Without a section: three faces per block, turned towards the eye. With one: the
    // whole cube, back faces included, so that cut blocks get their cut face.
    const threeFacesOnly = !sectionOn
    if (built) scene.mesh.geometry = sectionOn ? built.cube : built.faces
    if (material.transparent !== transparent || ('THREE_FACES' in material.defines) !== threeFacesOnly) {
      material.transparent = transparent
      material.depthWrite = !transparent
      if (threeFacesOnly) material.defines.THREE_FACES = 1
      else delete material.defines.THREE_FACES
      material.needsUpdate = true
    }
    scene.mesh.userData.seeThrough = transparent
    scene.outlineMaterial.color.set(INK[theme].select)
    invalidate()
  }, [scene, built, spec, theme, showMissing, edges, opacity, shrink, sectionOn, invalidate])

  // ---- legend
  useEffect(() => {
    const { setLegend } = useDisplay.getState()
    const name = column?.name
    if (!spec || !visible || !name) {
      setLegend(object.id, null)
      return
    }
    const { setNumericSettings, resetNumericSettings } = useBlocks.getState()
    setLegend(object.id, {
      spec,
      subject: object.name,
      onChange: spec.kind === 'numeric' ? (patch) => setNumericSettings(object.id, name, patch) : undefined,
      onReset: spec.kind === 'numeric' ? () => resetNumericSettings(object.id, name) : undefined,
    })
  }, [spec, visible, column?.name, object.id, object.name])
  useEffect(() => () => useDisplay.getState().setLegend(object.id, null), [object.id])

  // ---- picking: pointer ray -> model frame -> grid walk
  useEffect(() => {
    if (!built || !visible) return
    const picker: Picker = (ray) => {
      const ve = useViewer.getState().verticalExaggeration
      const display = useBlocks.getState().models[object.id]
      // Render space -> world -> model frame. The ray parameter t is unchanged by these
      // linear maps, so the hit distance is a render-space distance.
      const o = worldToLocal(definition, [ray.origin.x + origin[0], ray.origin.y + origin[1], ray.origin.z / ve + origin[2]])
      const d = worldToLocal(definition, [ray.direction.x, ray.direction.y, ray.direction.z / ve], true)
      let tMin = 0
      let tMax = Infinity
      if (sectionClipActive() && updateSectionPlanes()) {
        // Only the part of the ray inside the slab is on screen.
        const shift = sectionCutShift()
        for (const plane of SECTION_PLANES) {
          // Blocks end at the section plane, not at the front of the slab.
          const a = plane.normal.dot(ray.origin) + plane.constant - (plane === SECTION_PLANES[0] ? shift : 0)
          const b = plane.normal.dot(ray.direction)
          if (b === 0) {
            if (a < 0) return null
          } else if (b > 0) tMin = Math.max(tMin, -a / b)
          else tMax = Math.min(tMax, -a / b)
        }
      }
      const hideMissing = (display?.missing ?? 'hidden') === 'hidden' && built.missing > 0
      const hit = pickBlock(built.pick, o, d, {
        shrink: display?.shrink ?? 1,
        tMin,
        tMax,
        skip: hideMissing ? (i) => built.values[i] < MISSING_BELOW : undefined,
      })
      if (!hit) return null
      const index = hit.index
      return {
        distance: hit.t,
        point: [ray.origin.x + ray.direction.x * hit.t, ray.origin.y + ray.direction.y * hit.t, ray.origin.z + ray.direction.z * hit.t],
        hover: () => useDisplay.getState().setHover(null),
        select: () => {
          const g = blockGeometry(definition, set.structure, index)
          const row = set.rows ? set.rows[index] : index
          const sub = set.structure.sub.subarray(index * 3, index * 3 + 3)
          useDisplay.getState().setSelection(null)
          useBlocks.getState().setPick({
            objectId: object.id,
            row,
            blockId: blockId(definition, set.structure.parent[index], g.level, sub),
            ijk: g.ijk,
            level: g.level,
            centroid: g.centroid,
            size: g.size,
            detail: null,
            error: null,
          })
          // Every attribute of the block, at full precision, from the server.
          fetchBlockDetail({ href }, row).then(
            (detail) => useBlocks.getState().patchPick(object.id, row, { detail }),
            (error: unknown) => useBlocks.getState().patchPick(object.id, row, { error: error instanceof Error ? error.message : String(error) }),
          )
        },
      }
    }
    view.pickers.add(picker)
    return () => void view.pickers.delete(picker)
  }, [built, visible, set, definition, origin, object.id, href])

  // ---- outline of the picked block
  const pick = useBlocks((s) => (s.pick?.objectId === object.id ? s.pick : null))
  useLayoutEffect(() => {
    const outline = scene.outline
    if (!pick) {
      outline.visible = false
    } else {
      const r = definition.rotation
      const s = pick.size
      const k = shrink * 1.002
      outline.matrix.copy(
        new Matrix4().set(
          r[0] * s[0] * k, r[1] * s[1] * k, r[2] * s[2] * k, pick.centroid[0] - origin[0],
          r[3] * s[0] * k, r[4] * s[1] * k, r[5] * s[2] * k, pick.centroid[1] - origin[1],
          r[6] * s[0] * k, r[7] * s[1] * k, r[8] * s[2] * k, pick.centroid[2] - origin[2],
          0, 0, 0, 1,
        ),
      )
      outline.matrixWorldNeedsUpdate = true
      outline.visible = true
    }
    invalidate()
  }, [scene, pick, definition, origin, shrink, invalidate])

  useEffect(() => invalidate(), [visible, invalidate])

  if (!built) return null
  return <primitive object={scene.group} visible={visible} />
}

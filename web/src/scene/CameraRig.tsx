import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo } from 'react'
import { OrthographicCamera, PerspectiveCamera, Vector3 } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { allBounds, useViewer, visibleBounds, type Bounds, type Projection } from '../state/store'
import { clipPlanes, distanceForOrthoZoom, orthoZoomForDistance, visibleHeight, type DepthSphere } from './cameraMath'
import { getOrigin, renderToWorld, worldToRender, type Vec3 } from './origin'
import { view, type CameraState, type RigApi, type StandardView } from './view'

/** Narrow field of view: less perspective distortion, closer to a CAD/GIS feel. */
const FOV = 35

/** Direction from the orbit centre to the camera for each standard view (Z-up). */
const VIEW_DIRECTIONS: Record<StandardView, [number, number, number]> = {
  // A hair south of vertical so "up" on screen is north and the orbit stays well defined.
  plan: [0, -1e-5, 1],
  north: [0, -1, 0],
  east: [-1, 0, 0],
  iso: [-1, -1, 0.8],
}

/** Area shown by "zoom to extents" when nothing is loaded: 1 km around the origin. */
const EMPTY_RADIUS = 500

const tmp = new Vector3()
const forward = new Vector3()

function renderSphere(bounds: Bounds, exaggeration: number): { centre: Vector3; radius: number } {
  const a = worldToRender(bounds.min, exaggeration)
  const b = worldToRender(bounds.max, exaggeration)
  const centre = new Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
  const radius = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / 2
  return { centre, radius: Math.max(radius, 0.5) }
}

/** Owns both cameras and the orbit controls. Everything here is in render space. */
class Rig implements RigApi {
  readonly persp = new PerspectiveCamera(FOV, 1, 0.1, 1e5)
  readonly ortho = new OrthographicCamera(-1, 1, 1, -1, 0.1, 1e5)
  camera: PerspectiveCamera | OrthographicCamera = this.persp
  controls: OrbitControls | null = null
  readonly target = new Vector3()
  width = 1
  height = 1
  private exaggeration = 1
  onChange: () => void = () => {}

  constructor() {
    for (const camera of [this.persp, this.ortho]) {
      camera.up.set(0, 0, 1)
      // The rig sizes the frustums itself; stop react-three-fiber from doing it too.
      Object.assign(camera, { manual: true })
    }
    this.persp.position.set(...VIEW_DIRECTIONS.iso).normalize().multiplyScalar(EMPTY_RADIUS * 4)
    this.persp.lookAt(this.target)
  }

  get isPerspective(): boolean {
    return this.camera === this.persp
  }

  /** Distance from camera to orbit centre as a perspective camera would need it. */
  private viewDistance(): number {
    return this.isPerspective
      ? this.persp.position.distanceTo(this.target)
      : distanceForOrthoZoom(FOV, this.ortho.zoom, this.height)
  }

  /** Unit vector from the orbit centre towards the camera. */
  private direction(out: Vector3): Vector3 {
    out.copy(this.camera.position).sub(this.target)
    if (out.lengthSq() < 1e-12) out.set(...VIEW_DIRECTIONS.iso)
    return out.normalize()
  }

  /** Switches camera, carrying position, orientation and visible extent across. */
  activate(projection: Projection, dom: HTMLElement): void {
    const next = projection === 'perspective' ? this.persp : this.ortho
    if (next !== this.camera) {
      const from = this.camera
      if (this.controls) this.target.copy(this.controls.target)
      if (next === this.ortho) {
        const distance = from.position.distanceTo(this.target)
        this.ortho.position.copy(from.position)
        this.ortho.zoom = orthoZoomForDistance(FOV, distance, this.height)
      } else {
        const distance = distanceForOrthoZoom(FOV, this.ortho.zoom, this.height)
        const dir = this.direction(tmp)
        this.persp.position.copy(this.target).addScaledVector(dir, distance)
      }
      next.quaternion.copy(from.quaternion)
      this.camera = next
    }
    this.controls?.dispose()
    const controls = new OrbitControls(this.camera, dom)
    controls.target.copy(this.target)
    controls.enableDamping = false
    controls.zoomToCursor = true
    controls.screenSpacePanning = true
    controls.minDistance = 0.02
    controls.maxDistance = 5e6
    controls.minZoom = 1e-5
    controls.maxZoom = 1e5
    controls.addEventListener('change', () => this.onChange())
    controls.update()
    this.controls = controls
    this.frame()
  }

  dispose(): void {
    this.controls?.dispose()
    this.controls = null
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, width)
    this.height = Math.max(1, height)
  }

  /** Puts the camera `distance` from `target` along `dir` (unit, centre -> camera). */
  private place(target: Vector3, dir: Vector3, distance: number): void {
    this.target.copy(target)
    if (this.isPerspective) {
      this.persp.position.copy(target).addScaledVector(dir, distance)
    } else {
      this.ortho.zoom = orthoZoomForDistance(FOV, distance, this.height)
      // Position only sets the viewing direction and the clip range in orthographic.
      this.ortho.position.copy(target).addScaledVector(dir, distance)
    }
    if (this.controls) {
      this.controls.target.copy(target)
      this.controls.update()
    }
    this.frame()
    this.onChange()
  }

  setView = (name: StandardView): void => {
    if (this.controls) this.target.copy(this.controls.target)
    const dir = new Vector3(...VIEW_DIRECTIONS[name]).normalize()
    this.place(this.target.clone(), dir, this.viewDistance())
  }

  zoomToExtents = (): void => {
    if (this.controls) this.target.copy(this.controls.target)
    const bounds = visibleBounds(useViewer.getState())
    const sphere = bounds
      ? renderSphere(bounds, this.exaggeration)
      : { centre: new Vector3(), radius: EMPTY_RADIUS }
    const dir = this.direction(new Vector3())
    // Fit the sphere in the narrower of the two view angles.
    const aspect = this.width / this.height
    const halfV = (FOV * Math.PI) / 360
    const halfH = Math.atan(Math.tan(halfV) * aspect)
    let distance: number
    if (this.isPerspective) {
      distance = (sphere.radius / Math.sin(Math.min(halfV, halfH))) * 1.02
    } else {
      // Orthographic: the sphere's diameter must fit the shorter viewport side.
      const metresHigh = (2 * sphere.radius * 1.06) / Math.min(1, aspect)
      distance = metresHigh / (2 * Math.tan(halfV))
    }
    this.place(sphere.centre, dir, distance)
  }

  lookAt = (world: Vec3, distance: number): void => {
    const p = worldToRender(world, this.exaggeration)
    this.place(new Vector3(p[0], p[1], p[2]), this.direction(new Vector3()), Math.max(distance, 0.02))
  }

  /** Keeps the view on the same true-world point when the exaggeration changes. */
  setExaggeration(value: number): void {
    if (value === this.exaggeration) return
    if (this.controls) this.target.copy(this.controls.target)
    const shift = this.target.z * (value / this.exaggeration) - this.target.z
    this.exaggeration = value
    this.target.z += shift
    this.camera.position.z += shift
    if (this.controls) {
      this.controls.target.copy(this.target)
      this.controls.update()
    }
    this.frame()
  }

  getState = (): CameraState => {
    const o = getOrigin()
    const p = this.camera.position
    return {
      projection: this.isPerspective ? 'perspective' : 'orthographic',
      position: renderToWorld([p.x, p.y, p.z], this.exaggeration, o),
      target: renderToWorld([this.target.x, this.target.y, this.target.z], this.exaggeration, o),
      metresPerPixel: view.metresPerPixel,
      near: this.camera.near,
      far: this.camera.far,
    }
  }

  /** Per-frame: frustum size, clip planes from the scene extents, shared view data. */
  frame(): void {
    const camera = this.camera
    if (this.controls) this.target.copy(this.controls.target)
    const targetDistance = camera.position.distanceTo(this.target)

    let metresPerPixel: number
    if (camera === this.persp) {
      this.persp.aspect = this.width / this.height
      metresPerPixel = visibleHeight(FOV, Math.max(targetDistance, 1e-6)) / this.height
    } else {
      this.ortho.left = -this.width / 2
      this.ortho.right = this.width / 2
      this.ortho.top = this.height / 2
      this.ortho.bottom = -this.height / 2
      metresPerPixel = 1 / this.ortho.zoom
    }

    // Clip planes: enclose every loaded object plus the reference grid around the target.
    camera.getWorldDirection(forward)
    const spheres: DepthSphere[] = []
    const bounds = allBounds(useViewer.getState())
    if (bounds) {
      const s = renderSphere(bounds, this.exaggeration)
      spheres.push({ depth: tmp.copy(s.centre).sub(camera.position).dot(forward), radius: s.radius })
    }
    const viewRadius = Math.max(this.width, this.height) * metresPerPixel
    spheres.push({
      depth: tmp.copy(this.target).sub(camera.position).dot(forward),
      radius: viewRadius * 1.5 + (bounds ? renderSphere(bounds, this.exaggeration).radius : 0),
    })
    const planes = clipPlanes(camera === this.persp, spheres, targetDistance)
    camera.near = planes.near
    camera.far = planes.far
    camera.updateProjectionMatrix()
    camera.updateMatrixWorld()

    view.metresPerPixel = metresPerPixel
    view.target.copy(this.target)
    view.camera = camera
  }
}

/** Cameras, orbit controls and the imperative view commands (`view.rig`). */
export function CameraRig() {
  const gl = useThree((s) => s.gl)
  const set = useThree((s) => s.set)
  const invalidate = useThree((s) => s.invalidate)
  const size = useThree((s) => s.size)
  const projection = useViewer((s) => s.projection)
  const exaggeration = useViewer((s) => s.verticalExaggeration)
  const hasObjects = useViewer((s) => Object.keys(s.objects).length > 0)
  const rig = useMemo(() => new Rig(), [])

  useLayoutEffect(() => {
    rig.onChange = () => invalidate()
    view.rig = rig
    return () => {
      if (view.rig === rig) view.rig = null
    }
  }, [rig, invalidate])

  useLayoutEffect(() => {
    rig.resize(size.width, size.height)
    invalidate()
  }, [rig, size, invalidate])

  useLayoutEffect(() => {
    rig.activate(projection, gl.domElement)
    set({ camera: rig.camera })
    invalidate()
    return () => rig.dispose()
  }, [rig, projection, gl, set, invalidate])

  useLayoutEffect(() => {
    rig.setExaggeration(exaggeration)
    invalidate()
  }, [rig, exaggeration, invalidate])

  // Frame the data when the first object arrives.
  useEffect(() => {
    if (hasObjects) rig.zoomToExtents()
  }, [rig, hasObjects])

  // The canvas renders on demand: any state change (visibility, theme, ...) needs a frame.
  useEffect(() => useViewer.subscribe(() => invalidate()), [invalidate])

  useFrame(() => {
    rig.frame()
    view.frames++
  })
  return null
}

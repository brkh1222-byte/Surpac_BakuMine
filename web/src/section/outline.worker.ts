/**
 * Worker side of the mesh outlines: keeps a copy of each mesh and cuts it on request
 * (see outlineClient.ts for the protocol and outline.ts for the algorithm).
 */
import { cutMesh } from './outline'

export type OutlineRequest =
  | { type: 'mesh'; id: string; vertices: Float64Array; faces: Uint32Array }
  | { type: 'drop'; id: string }
  | { type: 'cut'; id: string; request: number; origin: [number, number, number]; normal: [number, number, number] }

export interface OutlineReply {
  request: number
  points: Float64Array
  offsets: Uint32Array
  closed: Uint8Array
  /** Time spent cutting, ms. */
  ms: number
}

const meshes = new Map<string, { vertices: Float64Array; faces: Uint32Array }>()
const scope = self as unknown as { postMessage: (message: OutlineReply, transfer: Transferable[]) => void }

self.onmessage = (event: MessageEvent<OutlineRequest>) => {
  const message = event.data
  if (message.type === 'mesh') meshes.set(message.id, { vertices: message.vertices, faces: message.faces })
  else if (message.type === 'drop') meshes.delete(message.id)
  else {
    const mesh = meshes.get(message.id)
    const start = performance.now()
    const outline = mesh
      ? cutMesh(mesh.vertices, mesh.faces, message.origin, message.normal)
      : { points: new Float64Array(0), offsets: new Uint32Array(1), closed: new Uint8Array(0) }
    // Copies, so that the shared EMPTY_OUTLINE buffers are never transferred away.
    const points = outline.points.slice()
    const offsets = outline.offsets.slice()
    const closed = outline.closed.slice()
    scope.postMessage({ request: message.request, points, offsets, closed, ms: performance.now() - start }, [
      points.buffer,
      offsets.buffer,
      closed.buffer,
    ])
  }
}

/** Resource hooks of the scene API, built on `useResource`. */
import { MANIFEST_LOAD } from '../state/store'
import {
  fetchDrillholes,
  fetchManifest,
  fetchMesh,
  type DrillholeSet,
  type ManifestObject,
  type MeshData,
  type SceneManifest,
} from './client'
import { useResource, type Resource } from './useResource'

/** The scene manifest; `enabled = false` (mock mode, `?scene=none`) loads nothing. */
export function useManifest(enabled: boolean): Resource<SceneManifest> {
  return useResource(enabled ? MANIFEST_LOAD : null, 'Project', fetchManifest)
}

export function drillholesKey(objectId: string, table: string | null): string {
  return `drillholes:${objectId}:${table ?? ''}`
}

/** Traces of a drillhole object with the intervals of `table` (null = traces only). */
export function useDrillholes(object: ManifestObject, table: string | null): Resource<DrillholeSet> {
  return useResource(
    drillholesKey(object.id, table),
    table ? `${object.name} (${table})` : object.name,
    (signal) => fetchDrillholes(object, table, signal),
    (data) => data.byteLength,
  )
}

export function useMesh(object: ManifestObject): Resource<MeshData> {
  return useResource(
    `mesh:${object.id}`,
    object.name,
    (signal) => fetchMesh(object, signal),
    (data) => data.byteLength,
  )
}

export const APP_TITLE = 'BakuMine Block Modeller'

export type HealthState = 'checking' | 'ok' | 'unreachable'

/** Calls the API health endpoint (proxied to port 8000 in dev). */
export async function fetchHealth(fetchFn: typeof fetch = fetch): Promise<HealthState> {
  try {
    const response = await fetchFn('/api/health')
    if (!response.ok) return 'unreachable'
    const body: unknown = await response.json()
    const status = (body as { status?: unknown } | null)?.status
    return status === 'ok' ? 'ok' : 'unreachable'
  } catch {
    return 'unreachable'
  }
}

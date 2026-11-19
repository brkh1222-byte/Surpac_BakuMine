import { useEffect, useState } from 'react'
import { APP_TITLE, fetchHealth, type HealthState } from './health'

const HEALTH_TEXT: Record<HealthState, string> = {
  checking: 'checking…',
  ok: 'ok',
  unreachable: 'unreachable',
}

function App() {
  const [health, setHealth] = useState<HealthState>('checking')

  useEffect(() => {
    let cancelled = false
    void fetchHealth().then((state) => {
      if (!cancelled) setHealth(state)
    })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main>
      <h1>{APP_TITLE}</h1>
      <p>
        API status: <span data-testid="api-status">{HEALTH_TEXT[health]}</span>
      </p>
    </main>
  )
}

export default App

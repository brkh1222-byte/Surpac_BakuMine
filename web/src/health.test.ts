import { expect, test } from 'vitest'
import { APP_TITLE, fetchHealth } from './health'

test('app title', () => {
  expect(APP_TITLE).toBe('BakuMine Block Modeller')
})

test('fetchHealth reports ok when the API answers {"status":"ok"}', async () => {
  const ok = (async () => new Response('{"status":"ok"}')) as typeof fetch
  expect(await fetchHealth(ok)).toBe('ok')
})

test('fetchHealth reports unreachable when the request fails', async () => {
  const down = (async () => {
    throw new Error('connection refused')
  }) as typeof fetch
  expect(await fetchHealth(down)).toBe('unreachable')
})

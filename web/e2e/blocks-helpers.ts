import { expect, type Page } from '@playwright/test'
import { settle } from './helpers'

/** Shared by the block model specs (blocks.spec.ts, blocks-filter.spec.ts). */
export const ID = 'block-model'
export const HREF = `/api/v1/scene/objects/${ID}/blocks`
/** Dark-theme category colours of rock = AIR, WASTE, ORE (colour/ramps.ts). */
export const ROCK = { AIR: '#3987e5', WASTE: '#d95926', ORE: '#199e70' }

export interface BlockMeta {
  n_total: number
  n_selected: number
  n_blocks: number
  selected_volume: number
  lod: string
}

/** What the server selects for `where` (the BKMA header of the payload). */
export async function serverMeta(page: Page, where: string | null, extra = 'lod=full'): Promise<BlockMeta> {
  const url = `${HREF}?${extra}${where ? `&where=${encodeURIComponent(where)}` : ''}`
  const response = await page.request.get(url)
  expect(response.status(), url).toBe(200)
  const body = await response.body()
  const length = body.readUInt32LE(8)
  return (JSON.parse(body.subarray(12, 12 + length).toString('utf8')) as { meta: BlockMeta }).meta
}

export function summary(page: Page) {
  return page.getByTestId(`block-summary-${ID}`)
}

/** Waits until the blocks of the current filter are on screen; returns the counts shown. */
export async function shown(page: Page): Promise<{ selected: number; shown: number; total: number; lod: string; clipped: boolean }> {
  // A change is debounced before the request goes out.
  await page.waitForTimeout(320)
  await expect(summary(page)).toHaveAttribute('data-state', 'ready')
  await settle(page, 2)
  const s = summary(page)
  return {
    selected: Number(await s.getAttribute('data-selected')),
    shown: Number(await s.getAttribute('data-shown')),
    total: Number(await s.getAttribute('data-total')),
    lod: (await s.getAttribute('data-lod')) ?? '',
    clipped: (await s.getAttribute('data-clipped')) === 'true',
  }
}

export const ORE_WHERE = JSON.stringify({ type: 'attribute', expression: 'rock == "ORE"' })

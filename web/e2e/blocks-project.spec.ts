/**
 * Block display of a PROJECT (not the demo): a model estimated by the synthetic gate
 * (scripts/e2e_display_project.py; own API, see playwright.config.ts). `rock` is
 * AIR / WASTE / ORE and the grade `au` exists in the ORE blocks alone, strictly inside the
 * model; below the topography there are more blocks than are sent in full (lod = shell).
 *
 * Coloured by grade, nothing at all used to be drawn of this model: only the outer shell
 * of the selection was sent, no shell block has a grade, and blocks without a value are
 * hidden.
 */
import { expect, test, type Page } from '@playwright/test'
import { bm, settle } from './helpers'

test.use({ baseURL: 'http://127.0.0.1:5196' })

const ID = 'model'
const HREF = `/api/v1/scene/objects/${ID}/blocks`
const OUT = 'test-results/screens'
/** `bakumine.api.blocks.AUTO_SHELL_BLOCKS`. */
const AUTO_SHELL = 250_000

interface Meta {
  n_total: number
  n_selected: number
  n_visible: number
  n_blocks: number
  lod: string
}

async function serverMeta(page: Page, query: string): Promise<Meta> {
  const response = await page.request.get(`${HREF}?${query}`)
  expect(response.status(), query).toBe(200)
  const body = await response.body()
  return (JSON.parse(body.subarray(12, 12 + body.readUInt32LE(8)).toString('utf8')) as { meta: Meta }).meta
}

const summary = (page: Page) => page.getByTestId(`block-summary-${ID}`)

async function counts(page: Page) {
  // A change is debounced before the request goes out.
  await page.waitForTimeout(320)
  await expect(summary(page)).toHaveAttribute('data-state', 'ready')
  await settle(page, 3)
  const s = summary(page)
  const number = async (name: string) => Number(await s.getAttribute(name))
  return {
    selected: await number('data-selected'),
    valued: await number('data-valued'),
    shown: await number('data-shown'),
    total: await number('data-total'),
    lod: await s.getAttribute('data-lod'),
  }
}

async function open(page: Page): Promise<void> {
  await page.goto('/?theme=dark')
  await page.waitForFunction(() => (window.__bakumine?.frames() ?? 0) > 0)
  await page.locator('[data-testid="load-status"][data-status="ready"]').waitFor({ timeout: 60_000 })
  await expect(summary(page)).toHaveAttribute('data-state', 'ready')
  await settle(page, 3)
}

/**
 * Looks for the blocks on screen: of about `limit` of the blocks drawn, those whose centre
 * is in view, and how many of these pixels show a colour of the legend rather than the
 * background (the pixel at a block's centre is that block or one in front of it).
 */
async function blockPixels(page: Page, limit = 400): Promise<{ inView: number; coloured: number; colours: number }> {
  await settle(page, 2)
  return bm(
    page,
    (h, [id, n]) => {
      const total = h.blocks()[id]?.count ?? 0
      const seen = new Set<string>()
      let inView = 0
      let coloured = 0
      const step = Math.max(1, Math.floor(total / n))
      for (let k = 0; k < total; k += step) {
        const block = h.blockAt(id, k)
        const at = block && h.worldToClient(block.centroid)
        const element = at && document.elementFromPoint(at.x, at.y)
        if (!at || !element || element.tagName !== 'CANVAS') continue
        inView++
        const colour = h.pixelAt(at.x, at.y)
        const [r, g, b] = [1, 3, 5].map((i) => parseInt(colour.slice(i, i + 2), 16))
        // Saturated: a ramp colour, not the grey of the background, the grid or "no value".
        if (Math.max(r, g, b) - Math.min(r, g, b) > 40) {
          coloured++
          seen.add(colour)
        }
      }
      return { inView, coloured, colours: seen.size }
    },
    [ID, limit] as const,
  )
}

test('the whole model coloured by grade draws the blocks that have one', async ({ page }) => {
  await open(page)
  // Look at the blocks alone.
  await page.getByRole('checkbox', { name: 'Show Drillholes', exact: true }).uncheck()
  await page.getByRole('checkbox', { name: 'Show Topography', exact: true }).uncheck()
  // Coloured by grade, no filter at all: every block is selected, far more than are sent in full.
  await page.getByTestId(`colour-by-${ID}`).selectOption('au')
  const remove = page.getByTestId('filter-a-remove')
  if ((await remove.count()) > 0) await remove.click()
  await expect(remove).toHaveCount(0)
  const c = await counts(page)
  const all = await serverMeta(page, 'lod=auto&attributes=au')
  const valued = await serverMeta(page, 'lod=auto&attributes=au&present=au')
  expect(all.n_selected).toBe(all.n_total)
  expect(all.n_total).toBeGreaterThan(AUTO_SHELL)
  // What used to be sent: the outer shell of the box.
  expect(all.lod).toBe('shell')
  expect(valued.n_visible).toBeGreaterThan(10_000)
  expect(valued.n_visible).toBeLessThan(all.n_total)
  expect(c).toMatchObject({ selected: all.n_total, total: all.n_total, valued: valued.n_visible, shown: valued.n_blocks })
  await expect(page.getByTestId(`colour-by-${ID}`)).toHaveValue('au')
  await expect(page.getByTestId(`missing-${ID}`)).toHaveValue('hidden')
  await expect(page.getByTestId(`filter-status-${ID}`)).toContainText('with au')

  const pixels = await blockPixels(page)
  expect(pixels.inView).toBeGreaterThan(100)
  expect(pixels.coloured).toBeGreaterThan(0.6 * pixels.inView)
  expect(pixels.colours).toBeGreaterThan(10)
  await page.screenshot({ path: `${OUT}/project-whole-model-by-grade.png` })

  // "No value: Grey" shows the others too: now the shell of the whole box, all of it grey.
  await page.getByTestId(`missing-${ID}`).selectOption('neutral')
  const grey = await counts(page)
  expect(grey).toMatchObject({ selected: all.n_total, valued: all.n_total, shown: all.n_blocks, lod: 'shell' })
  await expect(page.getByTestId(`filter-status-${ID}`)).not.toContainText('with au')
  const box = await blockPixels(page)
  expect(box.inView).toBeGreaterThan(100)
  expect(box.coloured).toBeLessThan(0.1 * box.inView)
  await page.screenshot({ path: `${OUT}/project-whole-model-grey.png` })

  // Hidden again: the blocks with a grade are back.
  await page.getByTestId(`missing-${ID}`).selectOption('hidden')
  const again = await counts(page)
  expect(again.shown).toBe(valued.n_blocks)
  expect((await blockPixels(page)).coloured).toBeGreaterThan(0.6 * pixels.inView)
})

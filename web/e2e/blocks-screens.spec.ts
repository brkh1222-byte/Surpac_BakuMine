import { test, type Page } from '@playwright/test'
import { bm, openDemo, settle } from './helpers'

/**
 * Review screenshots of the block model in test-results/screens/ (nothing is asserted;
 * the pictures are for a person to look at).
 */
const OUT = 'test-results/screens'
const ID = 'block-model'

async function ready(page: Page): Promise<void> {
  // Past the request debounce, then until the answer is on screen.
  await page.waitForTimeout(350)
  await page.locator(`[data-testid="block-summary-${ID}"][data-state="ready"]`).waitFor()
  await page.waitForFunction(() => !window.__bakumine?.outlineStats().busy)
  await settle(page, 3)
}

async function shot(page: Page, name: string): Promise<void> {
  await ready(page)
  await page.mouse.move(700, 795)
  await settle(page, 2)
  await page.waitForTimeout(150)
  await page.screenshot({ path: `${OUT}/${name}.png` })
}

for (const theme of ['dark', 'light'] as const) {
  test(`block model review screenshots, ${theme} theme`, async ({ page }) => {
    await openDemo(page, `theme=${theme}`, true)
    // 1. The default view: ORE blocks by grade, with the drillholes.
    await shot(page, `blocks-${theme}-iso-ore-au`)

    // 2. Grade filter.
    await page.getByTestId(`filter-add-${ID}`).selectOption('numeric')
    await page.getByTestId('filter-b-value').fill('1')
    await shot(page, `blocks-${theme}-iso-au-gt-1`)
    await page.getByTestId('legend-stepped').first().click()
    await shot(page, `blocks-${theme}-iso-au-gt-1-classes`)
    await page.getByTestId('legend-continuous').first().click()

    // The filter panel with one condition of most kinds, and an invalid expression.
    await page.getByTestId('filter-b-op').selectOption('between')
    await page.getByTestId('filter-b-min').fill('1')
    await page.getByTestId('filter-b-slider-max').fill('3.45')
    await page.getByTestId(`filter-add-${ID}`).selectOption('surface')
    await page.getByTestId(`filter-add-${ID}`).selectOption('elevation')
    await page.getByTestId('filter-d-min').fill('950')
    await page.getByTestId('filter-d-max').fill('1050')
    await page.getByTestId(`filter-add-${ID}`).selectOption('index')
    await page.getByTestId(`filter-combine-${ID}`).fill('a and b and (c or not d) and e')
    await page.getByTestId(`filter-combine-${ID}`).press('Enter')
    await shot(page, `blocks-${theme}-filter-panel`)
    await page.getByTestId(`filter-add-${ID}`).selectOption('expression')
    await page.getByTestId('filter-f-text').fill('gold > 1')
    await page.getByTestId('filter-f-text').press('Enter')
    await page.getByTestId(`filter-combine-${ID}`).fill('a and b and (c or not d) and e and f')
    await page.getByTestId(`filter-combine-${ID}`).press('Enter')
    await page.locator(`[data-testid="filter-status-${ID}"][data-state="error"]`).waitFor()
    await page.mouse.move(700, 795)
    await settle(page, 2)
    await page.screenshot({ path: `${OUT}/blocks-${theme}-filter-error.png` })
    await page.getByTestId(`filter-combine-${ID}`).fill('')
    await page.getByTestId(`filter-combine-${ID}`).press('Enter')
    for (const name of ['f', 'e', 'd', 'c', 'b']) await page.getByTestId(`filter-${name}-remove`).click()

    // 3. Close-up of the ore contact: sub-blocks at their own size; then with gaps.
    await ready(page)
    const block = await bm(page, (h, id) => {
      const n = h.blocks()[id].count
      for (let k = Math.floor(n / 2); k < n; k++) {
        const b = h.blockAt(id, k)
        if (b && b.level === 2) return b
      }
      return null
    }, ID)
    if (block) await bm(page, (h, p) => h.lookAt(p, 70), block.centroid)
    await shot(page, `blocks-${theme}-closeup-sub-blocks`)
    await page.getByTestId(`shrink-${ID}`).fill('0.9')
    await shot(page, `blocks-${theme}-closeup-shrink-0.9`)
    if (block) {
      const at = await bm(page, (h, p) => h.worldToClient(p), block.centroid)
      if (at) await page.mouse.click(at.x, at.y)
      await page.locator('[data-testid="block-panel"][data-state="ready"]').waitFor()
    }
    await shot(page, `blocks-${theme}-closeup-picked`)
    await page.keyboard.press('Escape')
    await page.getByTestId(`shrink-${ID}`).fill('1')

    // 4. Every rock type, W-E section view, with the ore outline and the topography profile.
    await page.getByTestId('filter-a-cat-AIR').check()
    await page.getByTestId('filter-a-cat-WASTE').check()
    await page.getByTestId(`colour-by-${ID}`).selectOption('rock')
    await page.getByRole('checkbox', { name: 'Show Ore solid', exact: true }).check()
    await page.getByTestId('view-iso').click()
    await bm(page, (h) => h.zoomToExtents())
    await shot(page, `blocks-${theme}-iso-all-rock`)
    await page.getByTestId('section-preset-we').click()
    await shot(page, `blocks-${theme}-3d-section-rock`)
    await page.getByTestId('section-look').click()
    await shot(page, `blocks-${theme}-we-section-rock`)
    await page.getByTestId(`colour-by-${ID}`).selectOption('au')
    await page.getByTestId(`missing-${ID}`).selectOption('neutral')
    await shot(page, `blocks-${theme}-we-section-au-missing-grey`)
    await page.getByTestId('filter-a-cat-AIR').uncheck()
    await page.getByTestId('filter-a-cat-WASTE').uncheck()
    await page.getByTestId('ve-input').fill('3')
    await page.getByTestId('ve-input').press('Enter')
    await shot(page, `blocks-${theme}-we-section-ore-ve3`)
  })
}

test('2M-block performance scene screenshot', async ({ page }) => {
  await page.goto('/?mock=perf&theme=dark&blocks=2000000&tris=0&holes=0')
  await page.locator('[data-testid="block-summary-perf-2000000"][data-state="ready"]').waitFor({ timeout: 120_000 })
  await settle(page, 3)
  await page.screenshot({ path: `${OUT}/blocks-perf-2m-shell.png` })
  await page.getByTestId('filter-add-perf-2000000').selectOption('numeric')
  await page.getByTestId('filter-a-attribute').selectOption('au')
  await page.getByTestId('filter-a-value').fill('1')
  await page.waitForTimeout(400)
  await page.locator('[data-testid="block-summary-perf-2000000"][data-state="ready"]').waitFor()
  await settle(page, 3)
  await page.screenshot({ path: `${OUT}/blocks-perf-2m-au-gt-1.png` })
})

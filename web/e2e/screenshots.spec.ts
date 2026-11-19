import { test, type Page } from '@playwright/test'
import type {} from '../src/scene/debug'

/**
 * Writes review screenshots of the mock scene to test-results/screens/. Nothing is
 * asserted against stored images (GPU and font rendering differ between machines);
 * the pictures are for a person to look at.
 */
const OUT = 'test-results/screens'

async function open(page: Page, query: string): Promise<void> {
  await page.goto(`/?${query}`)
  await page.waitForFunction(() => (window.__bakumine?.frames() ?? 0) > 0)
  await page.waitForTimeout(300)
}

async function shot(page: Page, name: string): Promise<void> {
  await page.evaluate(() => window.__bakumine?.render())
  await page.waitForTimeout(150)
  await page.screenshot({ path: `${OUT}/${name}.png` })
}

for (const theme of ['dark', 'light'] as const) {
  test(`review screenshots, ${theme} theme`, async ({ page }) => {
    await open(page, `mock=1&theme=${theme}`)
    await shot(page, `${theme}-perspective-iso`)

    await page.getByTestId('proj-orthographic').click()
    await shot(page, `${theme}-orthographic-iso`)

    await page.getByTestId('view-plan').click()
    await shot(page, `${theme}-orthographic-plan`)

    await page.getByTestId('view-north').click()
    await shot(page, `${theme}-orthographic-north`)

    await page.getByTestId('proj-perspective').click()
    await page.getByTestId('view-east').click()
    await shot(page, `${theme}-perspective-east`)

    await page.getByTestId('view-iso').click()
    await page.getByTestId('ve-input').fill('3')
    await page.getByTestId('ve-input').press('Enter')
    await page.getByTestId('zoom-extents').click()
    await shot(page, `${theme}-perspective-iso-ve3`)

    // 1.5 m from a box corner: checks the clip planes and labels at close range.
    await page.getByTestId('ve-input').fill('1')
    await page.getByTestId('ve-input').press('Enter')
    await page.evaluate(() => {
      const handle = window.__bakumine
      if (!handle) return
      const box = handle.mockBoxes()[0]
      handle.lookAt(
        [box.centre[0] + box.size[0] / 2, box.centre[1] + box.size[1] / 2, box.centre[2] + box.size[2] / 2],
        1.5,
      )
    })
    await shot(page, `${theme}-perspective-closeup`)
  })
}

test('review screenshots, empty scene', async ({ page }) => {
  await open(page, 'scene=none&theme=dark')
  await shot(page, 'empty-dark')
  await open(page, 'scene=none&theme=light')
  await shot(page, 'empty-light')
})

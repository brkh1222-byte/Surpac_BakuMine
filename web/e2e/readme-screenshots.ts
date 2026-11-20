/**
 * README screenshots (`make screenshots`, config playwright.screenshots.config.ts).
 *
 * Follows walkthrough A of the README on a fresh project: imports the synthetic CSVs,
 * composites, creates and codes the block model, estimates Au by ordinary kriging, reports,
 * then looks at the result in 3D and on a section. PNGs go to docs/screenshots/.
 * Only the steps are checked (each must finish); the pictures are for people to look at.
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'
import { bm, settle } from './helpers'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO = path.resolve(HERE, '../..')
const SYNTH = path.join(REPO, 'data/synthetic/default')
const OUT = path.join(REPO, 'docs/screenshots')
const truth = JSON.parse(readFileSync(path.join(SYNTH, 'truth.json'), 'utf8'))

const tid = (page: Page, id: string) => page.getByTestId(id)

async function openPanel(page: Page, id: string) {
  const toggle = tid(page, `${id}-toggle`)
  if ((await toggle.getAttribute('aria-expanded')) !== 'true') await toggle.click()
  await expect(toggle).toHaveAttribute('aria-expanded', 'true')
}

async function closePanel(page: Page, id: string) {
  const toggle = tid(page, `${id}-toggle`)
  if ((await toggle.getAttribute('aria-expanded')) === 'true') await toggle.click()
}

async function shot(page: Page, name: string) {
  await page.waitForFunction(() => !window.__bakumine?.outlineStats().busy)
  await page.mouse.move(1000, 980)
  await settle(page, 3)
  await page.waitForTimeout(300)
  await page.screenshot({ path: path.join(OUT, `${name}.png`) })
}

test('README walkthrough screenshots', async ({ page }) => {
  await page.goto('/?theme=dark')
  await expect(tid(page, 'api-status')).toHaveText('API: connected')

  // ---- 1. import -------------------------------------------------------------------------
  await openPanel(page, 'import')
  for (const table of ['collar', 'survey', 'assay', 'geology']) {
    await tid(page, `import-file-${table}`).setInputFiles(path.join(SYNTH, `${table}.csv`))
    await expect(tid(page, `import-rows-${table}`)).toContainText('rows')
  }
  await tid(page, 'import-run').click()
  await expect(tid(page, 'import-summary')).toContainText('0 errors', { timeout: 60_000 })
  await closePanel(page, 'import')

  // ---- 2. composite ----------------------------------------------------------------------
  await openPanel(page, 'composite')
  await tid(page, 'composite-table').selectOption('assay')
  await tid(page, 'composite-mode').selectOption('domain')
  await tid(page, 'composite-length').fill('2')
  await tid(page, 'composite-domain-table').selectOption('geology')
  await tid(page, 'composite-domain-column').selectOption('rock')
  await tid(page, 'composite-run').click()
  await expect(tid(page, 'composite-result')).toContainText('composites', { timeout: 120_000 })
  await closePanel(page, 'composite')

  // ---- 3. block model: create and code -----------------------------------------------------
  await openPanel(page, 'model')
  const grid = truth.blocks
  await tid(page, 'model-origin').fill(grid.origin.join(', '))
  await tid(page, 'model-size').fill(grid.size.join(', '))
  await tid(page, 'model-shape').fill(grid.shape.join(', '))
  await tid(page, 'model-depth').fill('2')
  await tid(page, 'model-create').click()
  await expect(tid(page, 'model-result')).toContainText('parent blocks', { timeout: 60_000 })

  await tid(page, 'code-attr-0-name').fill('rock')
  await tid(page, 'code-attr-0-background').fill('WASTE')
  await tid(page, 'code-add-attribute').click()
  await tid(page, 'code-attr-1-name').fill('density')
  await tid(page, 'code-attr-1-unit').fill('t/m3')
  await tid(page, 'code-add-step').click()
  await tid(page, 'code-add-step').click()
  const density = truth.density_t_m3
  await tid(page, 'code-step-0-attribute').fill('rock')
  await tid(page, 'code-step-0-value').fill('AIR')
  await tid(page, 'code-step-0-first-type').selectOption('dtm')
  await tid(page, 'code-step-0-first-ref').selectOption('topo')
  await tid(page, 'code-step-0-first-side').selectOption('above')
  await tid(page, 'code-step-0-first-outside').check()
  await tid(page, 'code-step-1-attribute').fill('rock')
  await tid(page, 'code-step-1-value').fill('ORE')
  await tid(page, 'code-step-1-first-type').selectOption('solid')
  await tid(page, 'code-step-1-first-ref').selectOption('ore-solid')
  await tid(page, 'code-step-1-first-side').selectOption('inside')
  await tid(page, 'code-step-2-attribute').fill('density')
  await tid(page, 'code-step-2-value').fill(`iif(rock == "ORE", ${density.ORE}, ${density.WASTE})`)
  await tid(page, 'code-step-2-first-type').selectOption('attribute')
  await tid(page, 'code-step-2-first-expression').fill('rock != "AIR"')
  await tid(page, 'code-run').click()
  await expect(tid(page, 'code-result')).toContainText('blocks now', { timeout: 120_000 })
  await closePanel(page, 'model')

  // ---- 4. estimate -------------------------------------------------------------------------
  await openPanel(page, 'estimate')
  await expect(tid(page, 'estimate-attribute')).toHaveValue('au')
  await tid(page, 'estimate-domain').fill('ORE')
  await tid(page, 'estimate-block-where').fill('rock == "ORE"')
  await tid(page, 'estimate-sample-column').selectOption('domain')
  await tid(page, 'estimate-sample-values').fill('ORE')
  const v = truth.variogram
  await tid(page, 'estimate-radii').fill(v.ranges_m.join(', '))
  // truth.json's plunge is +10 in its own convention, -10 in the angle fields (see workflow.spec.ts)
  await tid(page, 'estimate-angles').fill(`${v.angles_deg.bearing}, ${-v.angles_deg.plunge}, ${v.angles_deg.dip}`)
  await tid(page, 'estimate-nugget').fill(String(v.nugget))
  await tid(page, 'estimate-structure-0-sill').fill(String(v.sill))
  await tid(page, 'estimate-add-pass').click()
  await tid(page, 'estimate-add-pass').click()
  const passes = [
    ['tight', '0.25', '6', '16', '4'],
    ['medium', '0.5', '3', '16', '4'],
    ['wide', '1.5', '1', '16', ''],
  ]
  for (const [i, [name, factor, min, max, perHole]] of passes.entries()) {
    await tid(page, `estimate-pass-${i}-name`).fill(name)
    await tid(page, `estimate-pass-${i}-factor`).fill(factor)
    await tid(page, `estimate-pass-${i}-min`).fill(min)
    await tid(page, `estimate-pass-${i}-max`).fill(max)
    await tid(page, `estimate-pass-${i}-perhole`).fill(perHole)
  }
  await tid(page, 'estimate-run').click()
  await expect(tid(page, 'estimate-job')).toHaveAttribute('data-status', 'done', { timeout: 8 * 60_000 })
  await expect(tid(page, 'estimate-summary')).toContainText('blocks estimated')
  await tid(page, 'estimate-toggle').evaluate((el) => el.scrollIntoView({ block: 'start' }))
  await shot(page, '04-estimation-panel')
  await closePanel(page, 'estimate')

  // ---- 5. report ---------------------------------------------------------------------------
  await openPanel(page, 'report')
  await tid(page, 'report-grade-au').check({ timeout: 60_000 })
  await tid(page, 'report-density').fill('density')
  await tid(page, 'report-group-rock').check()
  await tid(page, 'report-run').click()
  await expect(tid(page, 'report-table')).toBeVisible({ timeout: 60_000 })
  // The sidebar is 228 px wide and the table scrolls sideways inside it; for the picture the
  // sidebar is widened so that every column shows.
  const wide = await page.addStyleTag({ content: '.app { grid-template-columns: 760px minmax(0, 1fr) !important; } .report-table-wrap { max-height: none !important; }' })
  await tid(page, 'report-table').scrollIntoViewIfNeeded()
  await shot(page, '05-report')
  await wide.evaluate((el) => (el as HTMLElement).remove())
  await closePanel(page, 'report')

  // ---- 6. the model in 3D: ORE blocks by Au grade, drillholes, ore solid --------------------
  // The model the panels made is the project object "model" (name given in the Block model panel).
  const ID = 'model'
  await tid(page, `colour-by-${ID}`).selectOption('au')
  await tid(page, `filter-add-${ID}`).selectOption('category')
  // (the attribute list shows only when the model has more than one category attribute)
  if (await tid(page, 'filter-a-attribute').count()) await tid(page, 'filter-a-attribute').selectOption('rock')
  // the condition starts with every category ticked: keep ORE only
  await tid(page, 'filter-a-cat-AIR').uncheck()
  await tid(page, 'filter-a-cat-WASTE').uncheck()
  await page.getByRole('checkbox', { name: 'Show Ore solid', exact: true }).check()
  await tid(page, 'mode-ore-solid').selectOption('transparent')
  await page.getByRole('checkbox', { name: 'Show Topography', exact: true }).uncheck()
  await tid(page, 'view-iso').click()
  await bm(page, (h) => h.zoomToExtents())
  await blocksReady(page, ID)
  await shot(page, '01-scene-3d')

  // ---- 7. filter: ORE and Au above 2 g/t ------------------------------------------------------
  await tid(page, `filter-add-${ID}`).selectOption('numeric')
  await tid(page, 'filter-b-attribute').selectOption('au')
  await tid(page, 'filter-b-value').fill('2')
  await tid(page, `filter-${ID}`).scrollIntoViewIfNeeded()
  await blocksReady(page, ID)
  await shot(page, '02-filter-panel')
  await tid(page, 'filter-b-remove').click()

  // ---- 8. W-E section through the model, looked at square-on ---------------------------------
  await tid(page, 'filter-a-cat-WASTE').check()
  await tid(page, `missing-${ID}`).selectOption('neutral')
  await tid(page, 'mode-ore-solid').selectOption('wireframe')
  await tid(page, 'section-preset-we').click()
  await tid(page, 'section-look').click()
  // closer in, on the ore body where the W-E preset cuts it
  await bm(page, (h) => h.lookAt([500620, 5300563, 1000], 560))
  await blocksReady(page, ID)
  await shot(page, '03-section')
})

async function blocksReady(page: Page, id: string) {
  await page.waitForTimeout(400)
  await page.locator(`[data-testid="block-summary-${id}"][data-state="ready"]`).waitFor()
}

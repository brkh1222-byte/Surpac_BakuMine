import { expect, test, type Page } from '@playwright/test'
import { HREF, ID, ROCK, serverMeta, shown, summary } from './blocks-helpers'
import { bm, openDemo, settle } from './helpers'

test('the filter panel opens on the ORE condition and shows the count', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  const counts = await shown(page)
  // One condition, "rock: ORE"; the number under it is the server's mask count.
  await expect(page.getByTestId(`filter-status-${ID}`)).toHaveAttribute('data-selected', String(counts.selected))
  await expect(page.getByTestId(`filter-status-${ID}`)).toHaveAttribute('data-state', 'ready')
  await expect(page.getByTestId('filter-a-cat-ORE')).toBeChecked()
  await expect(page.getByTestId('filter-a-cat-WASTE')).not.toBeChecked()
  await expect(page.getByTestId('filter-a-cat-AIR')).not.toBeChecked()
  // No category at all selects nothing; unticking the condition selects everything.
  await page.getByTestId('filter-a-cat-ORE').uncheck()
  expect((await shown(page)).selected).toBe(0)
  await page.getByTestId('filter-a-enabled').uncheck()
  const everything = await shown(page)
  expect(everything.selected).toBe(everything.total)
  expect(everything.lod).toBe('shell')
  expect(everything.shown).toBeLessThan(0.2 * everything.total)
})

test('a filter changes the count to the number the Python engine gives', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  const all = await shown(page)

  await page.getByTestId(`filter-add-${ID}`).selectOption('numeric')
  await expect(page.getByTestId('filter-b-attribute')).toHaveValue('au')
  await page.getByTestId('filter-b-value').fill('1')
  const filtered = await shown(page)
  // Exactly the JSON the panel sends; tests/api proves the server's rows for it equal
  // constraint.mask(model) (84 517 blocks on the committed demo data).
  const where = JSON.stringify({
    type: 'and',
    children: [
      { type: 'attribute', expression: 'rock == "ORE"' },
      { type: 'attribute', expression: 'au > 1' },
    ],
  })
  expect(await bm(page, (h, id) => h.blockStore.getState().models[id].where, ID)).toBe(where)
  const server = await serverMeta(page, where)
  expect(filtered.selected).toBe(server.n_selected)
  expect(filtered.selected).toBe(84_517)
  expect(filtered.selected).toBeLessThan(all.selected)
  expect(await bm(page, (h, id) => h.blocks()[id].count, ID)).toBe(server.n_selected)
  await expect(page.getByTestId(`filter-status-${ID}`)).toContainText('84 517')

  // A range with the dual slider, tied to the legend range (0 .. 4.6).
  await page.getByTestId('filter-b-op').selectOption('between')
  const slider = page.getByTestId('filter-b-slider')
  await expect(slider).toHaveAttribute('data-low', '0')
  await expect(slider).toHaveAttribute('data-high', '4.6')
  await page.getByTestId('filter-b-slider-min').fill('2.3')
  await expect(page.getByTestId('filter-b-min')).toHaveValue('2.3')
  await page.getByTestId('filter-b-max').fill('3')
  const ranged = await shown(page)
  const rangeWhere = JSON.stringify({
    type: 'and',
    children: [
      { type: 'attribute', expression: 'rock == "ORE"' },
      { type: 'attribute', expression: '2.3 <= au <= 3' },
    ],
  })
  expect(ranged.selected).toBe((await serverMeta(page, rangeWhere)).n_selected)
  expect(ranged.selected).toBeGreaterThan(0)
  expect(ranged.selected).toBeLessThan(filtered.selected)

  // Spatial conditions and the combine expression.
  await page.getByTestId('filter-b-remove').click()
  await page.getByTestId(`filter-add-${ID}`).selectOption('index')
  await page.getByTestId('filter-b-min').fill('20')
  await page.getByTestId('filter-b-max').fill('30')
  await page.getByTestId(`filter-add-${ID}`).selectOption('elevation')
  await page.getByTestId('filter-c-max').fill('1000')
  await page.getByTestId(`filter-combine-${ID}`).fill('a and (b or not c)')
  await page.getByTestId(`filter-combine-${ID}`).press('Enter')
  const combined = await shown(page)
  const combinedWhere = JSON.stringify({
    type: 'expression',
    expression: 'a and (b or not c)',
    constraints: {
      a: { type: 'attribute', expression: 'rock == "ORE"' },
      b: { type: 'range', j: [20, 30] },
      c: { type: 'range', z: [null, 1000] },
    },
  })
  expect(await bm(page, (h, id) => h.blockStore.getState().models[id].where, ID)).toBe(combinedWhere)
  expect(combined.selected).toBe((await serverMeta(page, combinedWhere)).n_selected)
  expect(combined.selected).toBeGreaterThan(0)
  expect(combined.selected).toBeLessThan(all.selected)
})

test('filter requests are debounced and cancelled; errors show the engine message', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  const all = await shown(page)
  const sent: string[] = []
  const aborted: string[] = []
  page.on('request', (r) => {
    if (r.url().includes(`${HREF}?`)) sent.push(r.url())
  })
  page.on('requestfailed', (r) => {
    if (r.url().includes(`${HREF}?`)) aborted.push(r.failure()?.errorText ?? '')
  })
  // Slow the endpoint down so that the states in between can be seen.
  await page.route(`**${HREF}?*`, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 600))
    await route.continue().catch(() => {})
  })

  await page.getByTestId(`filter-add-${ID}`).selectOption('numeric')
  const value = page.getByTestId('filter-b-value')
  // Typing "1.25" is four changes and one request.
  await value.pressSequentially('1.25', { delay: 30 })
  const status = page.getByTestId(`filter-status-${ID}`)
  await expect(status).toHaveAttribute('data-state', 'loading')
  await expect(status).toHaveText('filtering…')
  await expect(summary(page)).toContainText('filtering…')
  // The previous blocks stay on screen meanwhile.
  expect(await bm(page, (h, id) => h.blocks()[id].count, ID)).toBe(all.selected)
  await page.waitForTimeout(350)
  // A newer change while that request is out: it is aborted, the new one wins.
  await value.fill('2')
  await expect(status).toHaveAttribute('data-state', 'ready', { timeout: 10_000 })
  const final = await shown(page)
  const expected = await serverMeta(
    page,
    JSON.stringify({ type: 'and', children: [{ type: 'attribute', expression: 'rock == "ORE"' }, { type: 'attribute', expression: 'au > 2' }] }),
  )
  expect(final.selected).toBe(expected.n_selected)
  expect(sent.length).toBeLessThanOrEqual(3) // "0" default, "1.25", "2" - not one per key
  expect(sent.length).toBeGreaterThanOrEqual(2)
  expect(aborted.length).toBeGreaterThanOrEqual(1)
  await page.unroute(`**${HREF}?*`)

  // An invalid expression: the server's ExpressionError text, and the blocks stay.
  await page.getByTestId(`filter-add-${ID}`).selectOption('expression')
  await page.getByTestId('filter-c-text').fill('gold > 1')
  await page.getByTestId('filter-c-text').press('Enter')
  await expect(status).toHaveAttribute('data-state', 'error')
  const error = page.getByTestId(`filter-error-${ID}`)
  await expect(error).toContainText("unknown name 'gold'")
  await expect(error).toContainText('available: rock, au')
  await expect(summary(page)).toHaveAttribute('data-state', 'error')
  expect(await bm(page, (h, id) => h.blocks()[id].count, ID)).toBe(final.selected)
  // Nothing but the expression language is accepted.
  await page.getByTestId('filter-c-text').fill("__import__('os')")
  await page.getByTestId('filter-c-text').press('Enter')
  await expect(error).toContainText("unknown function '__import__'")
  // Fixed: back to counts.
  await page.getByTestId('filter-c-text').fill('density > 2.8')
  await page.getByTestId('filter-c-text').press('Enter')
  await expect(status).toHaveAttribute('data-state', 'ready')
  expect((await shown(page)).selected).toBeLessThanOrEqual(final.selected)
})

/** Screen extent of the ORE-coloured pixels against the ore outline's, in client pixels. */
async function oreAgainstOutline(page: Page) {
  await page.waitForFunction(() => !window.__bakumine?.outlineStats().busy)
  await settle(page, 3)
  return bm(
    page,
    (h, hex) => {
      const pixels = h.pixelBounds(hex, 8)
      const outline = h.sectionOutlines()['ore-solid']
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity
      for (const line of outline?.polylines ?? []) {
        for (let i = 0; i < line.points.length; i += 3) {
          const p = h.worldToClient([line.points[i], line.points[i + 1], line.points[i + 2]])
          if (!p) continue
          minX = Math.min(minX, p.x)
          maxX = Math.max(maxX, p.x)
          minY = Math.min(minY, p.y)
          maxY = Math.max(maxY, p.y)
        }
      }
      const camera = h.camera()
      return { pixels, outline: { minX, minY, maxX, maxY }, metresPerPixel: camera?.metresPerPixel ?? 1, ve: h.store.getState().verticalExaggeration }
    },
    ROCK.ORE,
  )
}

function expectInsideOutline(r: Awaited<ReturnType<typeof oreAgainstOutline>>): void {
  expect(r.pixels).not.toBeNull()
  // One parent block (10 m across, 5 m high) plus the line width.
  const dx = 10 / r.metresPerPixel + 3
  const dy = (5 * r.ve) / r.metresPerPixel + 3
  const p = r.pixels!
  expect(p.minX).toBeGreaterThan(r.outline.minX - dx)
  expect(p.maxX).toBeLessThan(r.outline.maxX + dx)
  expect(p.minY).toBeGreaterThan(r.outline.minY - dy)
  expect(p.maxY).toBeLessThan(r.outline.maxY + dy)
  // And they fill it: at least most of the outline's extent is covered.
  expect(p.maxX - p.minX).toBeGreaterThan(0.8 * (r.outline.maxX - r.outline.minX) - dx)
  expect(p.maxY - p.minY).toBeGreaterThan(0.8 * (r.outline.maxY - r.outline.minY) - dy)
}

test('a W-E section shows the interior blocks inside the ore outline, also at vertical x 3', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  await shown(page)
  await page.getByRole('checkbox', { name: 'Show Drillholes', exact: true }).uncheck()
  await page.getByRole('checkbox', { name: 'Show Topography', exact: true }).uncheck()
  // The solid as its outline only: its fill would tint the blocks.
  await page.getByRole('checkbox', { name: 'Show Ore solid', exact: true }).check()
  await page.getByTestId('mode-ore-solid').selectOption('wireframe')
  await page.getByTestId(`colour-by-${ID}`).selectOption('rock')
  await page.getByTestId(`edges-${ID}`).uncheck()
  await shown(page)

  // The default slab (12.5 m either side). Blocks are cut at the plane itself, where
  // the outline is drawn.
  await page.getByTestId('section-preset-we').click()
  await page.getByTestId('section-look').click()
  await shown(page)
  const clip = await bm(page, (h) => h.sectionClip())
  expect(clip.materials.find((m) => m.name === `blocks:${ID}`)?.planes).toBe(2)

  // ORE only (all of it is loaded: the GPU clips): the cut blocks fill the ore outline.
  const ore = await oreAgainstOutline(page)
  expectInsideOutline(ore)
  expect(ore.pixels!.count).toBeGreaterThan(300)

  // Every rock type: 400 183 blocks arrive as a shell, so the slab is requested from the
  // server - and the section is solid rock, not a hollow skin.
  await page.getByTestId('filter-a-cat-WASTE').check()
  await page.getByTestId('filter-a-cat-AIR').check()
  await expect.poll(async () => (await shown(page)).clipped, { timeout: 15_000 }).toBe(true)
  const state = await shown(page)
  expect(state.selected).toBe(state.total)
  expect(state.shown).toBeLessThan(state.selected)
  expect(state.shown).toBeGreaterThan(3_000)
  const clipConstraint = await bm(page, (h) => {
    const s = h.section.getState()
    return { centre: s.plane.centre, front: s.front, back: s.back }
  })
  // The server's count for the same slab: from the plane to its back (12.5 m north),
  // widened by half a block (5 m) either side.
  const y = clipConstraint.centre[1]
  expect(clipConstraint.back).toBe(12.5)
  const slab = JSON.stringify({
    type: 'and',
    children: [
      { type: 'plane', point: [clipConstraint.centre[0], y - 5, clipConstraint.centre[2]], normal: [0, -1, 0], side: 'below' },
      { type: 'plane', point: [clipConstraint.centre[0], y + 17.5, clipConstraint.centre[2]], normal: [0, -1, 0], side: 'above' },
    ],
  })
  const viaApi = await serverMeta(page, null, `lod=auto&clip=${encodeURIComponent(slab)}`)
  expect(state.shown).toBe(viaApi.n_blocks)

  const all = await oreAgainstOutline(page)
  expectInsideOutline(all)
  // Waste fills the section around the ore: far more waste pixels than a skin would give,
  // in the exact legend colour (faces square-on to the section view).
  const waste = await bm(page, (h, hex) => h.pixelBounds(hex, 8), ROCK.WASTE)
  const air = await bm(page, (h, hex) => h.countPixels(hex, 8), ROCK.AIR)
  expect(waste).not.toBeNull()
  const canvas = await page.getByTestId('scene-canvas').boundingBox()
  expect(waste!.count).toBeGreaterThan(0.2 * canvas!.width * canvas!.height)
  expect(waste!.count).toBeGreaterThan(0.85 * (waste!.maxX - waste!.minX) * (waste!.maxY - waste!.minY) - all.pixels!.count * 3)
  expect(air).toBeGreaterThan(1_000)
  // The middle of the model, well inside the rock mass, is drawn.
  const inside = await bm(page, (h, yy) => h.worldToClient([500480, yy, 900]), y)
  expect(inside).not.toBeNull()
  expect(await bm(page, (h, p) => h.pixelAt(p.x, p.y), inside!)).toBe(ROCK.WASTE)

  // Vertical exaggeration 3: blocks and outline stretch together.
  await page.getByTestId('ve-input').fill('3')
  await page.getByTestId('ve-input').press('Enter')
  // Frame the ore (three times as tall now) so that its blocks are several pixels each.
  await bm(page, (h) => {
    const points = h.sectionOutlines()['ore-solid'].polylines.flatMap((l) => l.points)
    const middle = (axis: number) => {
      const values = points.filter((_, i) => i % 3 === axis)
      return (Math.min(...values) + Math.max(...values)) / 2
    }
    h.lookAt([middle(0), middle(1), middle(2)], 800)
  })
  await shown(page)
  const stretched = await oreAgainstOutline(page)
  expect(stretched.ve).toBe(3)
  expect(stretched.metresPerPixel).toBeLessThan(1.5)
  expectInsideOutline(stretched)
  // About three times as tall for its width as the unexaggerated section (the view may
  // cut the ends of the outline off).
  const aspect = (r: typeof all) => (r.outline.maxY - r.outline.minY) / (r.outline.maxX - r.outline.minX)
  expect(aspect(stretched) / aspect(all)).toBeGreaterThan(2.5)
  expect(aspect(stretched) / aspect(all)).toBeLessThan(3.1)

  // Leaving the section: the shell again, without a clip.
  await page.getByTestId('section-enabled').uncheck()
  await expect.poll(async () => (await shown(page)).clipped, { timeout: 15_000 }).toBe(false)
  const back = await shown(page)
  expect(back.lod).toBe('shell')
  expect(back.shown).toBeLessThan(0.2 * back.selected)
})


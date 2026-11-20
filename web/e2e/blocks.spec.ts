import { expect, test } from '@playwright/test'
import { HREF, ID, ORE_WHERE, ROCK, serverMeta, shown, summary } from './blocks-helpers'
import { bm, isSoftwareRenderer, openDemo, settle } from './helpers'

test('the demo block model opens on its ORE blocks, coloured by grade', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  // Object tree: Block models -> Demo block model; the solid it is cut by starts unticked.
  const tree = page.getByTestId('tree-blockModels')
  await expect(tree.getByText('Demo block model')).toBeVisible()
  await expect(page.getByRole('checkbox', { name: 'Show Demo block model', exact: true })).toBeChecked()
  await expect(page.getByRole('checkbox', { name: 'Show Ore solid', exact: true })).not.toBeChecked()

  // The count on screen is the server's mask count for rock == "ORE".
  const counts = await shown(page)
  const server = await serverMeta(page, ORE_WHERE)
  expect(server.n_selected).toBeGreaterThan(50_000)
  expect(counts.selected).toBe(server.n_selected)
  expect(counts.total).toBe(server.n_total)
  expect(counts.shown).toBe(server.n_selected) // small enough: every selected block is drawn
  expect(counts.lod).toBe('full')
  await expect(summary(page)).toContainText('Mm³')
  expect(Number(await summary(page).getAttribute('data-volume'))).toBeCloseTo(server.selected_volume, 3)
  // Coloured by au with the shared legend.
  await expect(page.getByTestId(`colour-by-${ID}`)).toHaveValue('au')
  const legend = page.getByTestId('legend').filter({ hasText: 'Demo block model' })
  await expect(legend.getByTestId('legend-title')).toHaveText('au (g/t)')

  // What is drawn: one instance per block, sub-blocks at their own size, no mine
  // coordinates in any buffer.
  const drawn = await bm(page, (h, id) => h.blocks()[id], ID)
  expect(drawn.count).toBe(server.n_selected)
  expect(drawn.attribute).toBe('au')
  expect(drawn.sizes).toEqual([[10, 10, 5], [5, 5, 2.5], [2.5, 2.5, 1.25]])
  expect(drawn.maxAbsCentre).toBeLessThan(1e3)
  const buffers = await bm(page, (h) => h.buffers())
  const mine = buffers.find((b) => b.name === `blocks:${ID}`)
  expect(mine?.vertices).toBe(server.n_selected)
  for (const buffer of buffers) expect(buffer.maxAbs, buffer.name).toBeLessThan(1e5)
  expect(await bm(page, (h) => h.maxAbsBufferCoord())).toBeLessThan(1e5)

  // One draw call for all the blocks.
  const withBlocks = await bm(page, (h) => h.renderInfo().calls)
  await page.getByRole('checkbox', { name: 'Show Demo block model', exact: true }).uncheck()
  await settle(page)
  const without = await bm(page, (h) => h.renderInfo().calls)
  expect(withBlocks - without).toBe(1)
  await expect(summary(page)).toHaveCount(0)
})

test('changing the colour attribute changes the legend and the pixels', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  await shown(page)
  // Look at the blocks alone, from above: top faces carry the legend colour itself.
  await page.getByRole('checkbox', { name: 'Show Drillholes', exact: true }).uncheck()
  await page.getByRole('checkbox', { name: 'Show Topography', exact: true }).uncheck()
  await page.getByTestId('view-plan').click()
  await page.getByTestId('proj-orthographic').click()
  await settle(page)
  const legend = page.getByTestId('legend').filter({ hasText: 'Demo block model' })
  const byGrade = await page.getByTestId('viewport').screenshot()
  const oreBefore = await bm(page, (h, hex) => h.countPixels(hex, 6), ROCK.ORE)

  await page.getByTestId(`colour-by-${ID}`).selectOption('rock')
  await expect(legend).toHaveAttribute('data-kind', 'categorical')
  await expect(legend.getByTestId('legend-title')).toHaveText('rock')
  await expect(legend.getByTestId('legend-rows')).toContainText('ORE')
  await shown(page)
  await expect.poll(() => bm(page, (h, id) => h.blocks()[id].attribute, ID)).toBe('rock')
  const byRock = await page.getByTestId('viewport').screenshot()
  expect(byRock.equals(byGrade)).toBe(false)
  // Every block is ORE: the faces are the legend's ORE colour, exactly.
  const ore = await bm(page, (h, hex) => h.countPixels(hex, 6), ROCK.ORE)
  expect(ore).toBeGreaterThan(oreBefore + 5_000)
  expect(await bm(page, (h, hex) => h.countPixels(hex, 6), ROCK.WASTE)).toBe(0)

  // A numeric attribute again: range editing recolours without a new request.
  await page.getByTestId(`colour-by-${ID}`).selectOption('au')
  await expect(legend).toHaveAttribute('data-kind', 'continuous')
  await shown(page)
  await expect.poll(() => bm(page, (h, id) => h.blocks()[id].attribute, ID)).toBe('au')
  let requests = 0
  page.on('request', (r) => {
    if (r.url().includes('/blocks')) requests++
  })
  const before = await page.getByTestId('viewport').screenshot()
  await legend.getByTestId('legend-max').fill('1')
  await legend.getByTestId('legend-max').press('Enter')
  await settle(page)
  const narrow = await page.getByTestId('viewport').screenshot()
  expect(narrow.equals(before)).toBe(false)
  await legend.getByTestId('legend-stepped').click()
  await settle(page)
  expect((await page.getByTestId('viewport').screenshot()).equals(narrow)).toBe(false)
  // Edges, opacity and missing-value display are uniforms too.
  await page.getByTestId(`edges-${ID}`).uncheck()
  await settle(page)
  expect(requests).toBe(0)
})

test('picking a block shows all its attributes, equal to the server values for that block', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  await shown(page)
  await page.getByRole('checkbox', { name: 'Show Drillholes', exact: true }).uncheck()
  await page.getByRole('checkbox', { name: 'Show Topography', exact: true }).uncheck()
  // Aim at a sub-block in the middle of the table, close up.
  const target = await bm(page, (h, id) => {
    const n = h.blocks()[id].count
    for (let k = Math.floor(n / 2); k < n; k++) {
      const b = h.blockAt(id, k)
      if (b && b.level === 2) return b
    }
    return null
  }, ID)
  expect(target).not.toBeNull()
  await bm(page, (h, p) => h.lookAt(p, 60), target!.centroid)
  await settle(page)
  const at = await bm(page, (h, p) => h.worldToClient(p), target!.centroid)
  expect(at).not.toBeNull()
  await page.mouse.move(at!.x, at!.y)
  await page.mouse.click(at!.x, at!.y)

  const panel = page.getByTestId('block-panel')
  await expect(panel).toHaveAttribute('data-state', 'ready')
  const blockId = await panel.getAttribute('data-block-id')
  const row = Number(await panel.getAttribute('data-row'))
  // The server's record of that block id.
  const response = await page.request.get(`${HREF}/detail?block_id=${blockId}`)
  expect(response.status()).toBe(200)
  const detail = (await response.json()) as {
    row: number
    ijk: number[]
    level: number
    centroid: number[]
    size: number[]
    attributes: Record<string, number | string | null>
  }
  expect(detail.row).toBe(row)
  await expect(page.getByTestId('block-ijk')).toHaveText(detail.ijk.join(', '))
  await expect(page.getByTestId('block-level')).toContainText(String(detail.level))
  expect(await page.getByTestId('block-size').getAttribute('data-value')).toBe(detail.size.join(','))
  for (const [axis, testId] of [[0, 'block-e'], [1, 'block-n'], [2, 'block-z']] as const) {
    expect(Number(await page.getByTestId(testId).getAttribute('data-value'))).toBe(detail.centroid[axis])
  }
  // ALL attributes, not just the one the scene is coloured by.
  const names = Object.keys(detail.attributes)
  expect(names).toEqual(['rock', 'au', 'au_ore', 'ore_fraction', 'density'])
  for (const name of names) {
    const value = detail.attributes[name]
    expect(await page.getByTestId(`block-attr-${name}`).getAttribute('data-value'), name).toBe(value === null ? '' : String(value))
  }
  expect(detail.attributes.rock).toBe('ORE') // only ORE blocks are on screen
  // The block hit is the one in front along the ray: no farther than the one aimed at.
  const camera = await bm(page, (h) => h.camera())
  const distance = (p: number[]) => Math.hypot(p[0] - camera!.position[0], p[1] - camera!.position[1], p[2] - camera!.position[2])
  expect(distance(detail.centroid)).toBeLessThanOrEqual(distance(target!.centroid) + 1e-6)
  // It is outlined, and the cursor read-out is on it.
  const buffers = await bm(page, (h) => h.buffers())
  expect(buffers.some((b) => b.name === 'block-highlight')).toBe(true)
  await expect(page.getByTestId('status-coords')).toHaveAttribute('data-on', 'object')

  // A click on empty space clears it.
  await page.getByTestId('view-plan').click()
  await bm(page, (h) => h.zoomToExtents())
  await settle(page)
  const box = await page.getByTestId('scene-canvas').boundingBox()
  await page.mouse.click(box!.x + 30, box!.y + box!.height - 60)
  await expect(panel).toHaveCount(0)
})

test('a section clips the blocks on the GPU and closes the cut blocks', async ({ page }) => {
  await openDemo(page, 'theme=dark', true)
  const before = await shown(page)
  await page.getByRole('checkbox', { name: 'Show Drillholes', exact: true }).uncheck()
  await page.getByRole('checkbox', { name: 'Show Topography', exact: true }).uncheck()
  await page.getByTestId(`colour-by-${ID}`).selectOption('rock')
  await page.getByTestId(`edges-${ID}`).uncheck()
  await shown(page)
  await page.getByTestId('section-preset-we').click()
  await page.getByTestId('section-look').click()
  await settle(page, 3)
  // The same blocks (every ORE block was loaded): clipping is two planes in the shader.
  const during = await shown(page)
  expect(during.selected).toBe(before.selected)
  expect(during.shown).toBe(before.selected)
  expect(during.clipped).toBe(false)
  const clip = await bm(page, (h) => h.sectionClip())
  expect(clip.active).toBe(true)
  expect(clip.materials.find((m) => m.name === `blocks:${ID}`)?.planes).toBe(2)
  // Cut blocks are closed with their own colour: the section of the ore body is solid,
  // square-on faces in the exact legend colour, and it is a thin band of the whole body.
  const cut = await bm(page, (h, hex) => h.pixelBounds(hex, 6), ROCK.ORE)
  expect(cut).not.toBeNull()
  expect(cut!.count).toBeGreaterThan(1_500)
  // The cut is at the section plane: its extent on screen is the ore outline's there
  // (the solid's own outline needs the solid shown; its bounds along the section do).
  const height = (cut!.maxY - cut!.minY) * (await bm(page, (h) => h.camera()?.metresPerPixel ?? 1))
  expect(height).toBeGreaterThan(100)
  expect(height).toBeLessThan(260)
  // Ghost mode keeps the whole body (dimmed) and still draws the slab; off: no planes.
  await bm(page, (h) => h.section.getState().setMode('ghost'))
  await settle(page, 3)
  expect((await bm(page, (h, hex) => h.pixelBounds(hex, 6), ROCK.ORE))?.count ?? 0).toBeGreaterThan(1_500)
  await bm(page, (h) => h.section.getState().setEnabled(false))
  await settle(page, 3)
  expect((await bm(page, (h) => h.sectionClip())).materials.find((m) => m.name === `blocks:${ID}`)?.planes).toBe(0)
})

test('2M blocks: transferred and displayed, filtered and orbited within the plan criteria', async ({ page }) => {
  test.setTimeout(240_000)
  const N = 2_000_000
  const PERF = `perf-${N}`
  const perfSummary = page.getByTestId(`block-summary-${PERF}`)
  const drawnState = () => bm(page, (h, id) => h.blocks()[id], PERF)
  const waitDrawn = async (previousCount: number | null) => {
    await expect.poll(async () => {
      const b = await drawnState().catch(() => undefined)
      return b !== undefined && b.timing.toFrameMs > 0 && b.count !== previousCount
    }, { timeout: 120_000, intervals: [50] }).toBe(true)
    return drawnState()
  }

  const opened = Date.now()
  await page.goto(`/?mock=perf&theme=dark&blocks=${N}&tris=0&holes=0`)
  await expect(perfSummary).toHaveAttribute('data-state', 'ready', { timeout: 120_000 })
  const first = await waitDrawn(null)
  const pageLoadMs = Date.now() - opened
  const info = await bm(page, (h) => h.renderInfo())
  const software = isSoftwareRenderer(info.renderer)
  const note = software ? ' (SOFTWARE rendering: not indicative)' : ' (headless: indicative only)'
  expect(Number(await perfSummary.getAttribute('data-total'))).toBe(N)
  expect(Number(await perfSummary.getAttribute('data-selected'))).toBe(N)
  expect(first.lod).toBe('shell')
  expect(first.count).toBeLessThan(150_000)
  expect(first.maxAbsCentre).toBeLessThan(1e5)
  const lines: string[] = []
  const line = (what: string, b: Awaited<ReturnType<typeof drawnState>>) =>
    `[perf] ${what}: ${b.count.toLocaleString('en-US')} blocks drawn of ${b.selected.toLocaleString('en-US')} selected (${b.lod}, ${b.bytesPerBlock.toFixed(1)} B/block); ` +
    `request+decode ${b.timing.requestMs.toFixed(0)} ms, buffers ${b.timing.buildMs.toFixed(0)} ms, request -> draw issued ${b.timing.toDrawMs.toFixed(0)} ms, request -> GPU done ${b.timing.toFrameMs.toFixed(0)} ms`
  lines.push(line(`2M model, first view (exposed shell)`, first) + `; page open -> first frame ${pageLoadMs} ms`)

  let orbit = await bm(page, (h) => h.measureOrbit(90, 2))
  lines.push(`[perf] orbit, shell of 2M: frame interval median ${orbit.intervalMedianMs.toFixed(1)} ms, p95 ${orbit.intervalP95Ms.toFixed(1)} ms; unthrottled frame cost mean ${orbit.renderMeanMs.toFixed(2)} ms`)
  if (!software) expect(orbit.renderMeanMs).toBeLessThan(33)

  // Filter change: au > 1 (about 258 000 blocks selected). Timed from the store change to
  // the GPU having drawn the answer; the request debounce (220 ms) is part of it.
  const filterMs = await bm(
    page,
    async (h, id) => {
      const before = h.blocks()[id]
      const start = performance.now()
      h.blockStore.getState().setFilter(id, {
        combine: '',
        conditions: [{ id: 9001, enabled: true, kind: 'numeric', attribute: 'au', op: '>', value: 1, min: null, max: null }],
      })
      for (;;) {
        await new Promise((resolve) => requestAnimationFrame(resolve))
        const now = h.blocks()[id]
        if (now.timing !== before.timing && now.count !== before.count && now.timing.toFrameMs > 0) break
        if (performance.now() - start > 60_000) throw new Error('filter did not finish')
      }
      h.finishFrame()
      return performance.now() - start
    },
    PERF,
  )
  const filtered = await drawnState()
  expect(filtered.selected).toBeGreaterThan(200_000)
  expect(filtered.selected).toBeLessThan(N)
  lines.push(line(`filter au > 1`, filtered) + `; store change -> on screen ${filterMs.toFixed(0)} ms (220 ms of it debounce)`)
  if (!software) expect(filterMs).toBeLessThan(1_000)

  // The brute-force case: every one of the 2M blocks as an instance (lod = full).
  await bm(page, (h, id) => h.blockStore.getState().setFilter(id, { combine: '', conditions: [] }), PERF)
  await waitDrawn(filtered.count)
  const shell = await drawnState()
  await bm(page, (h, id) => h.blockStore.getState().setDisplay(id, { shrink: 0.95 }), PERF)
  const full = await waitDrawn(shell.count)
  expect(full.count).toBe(N)
  expect(full.lod).toBe('full')
  lines.push(line(`2M model, every block (lod=full, shrink 0.95)`, full))
  if (!software) expect(full.timing.toFrameMs).toBeLessThan(5_000)
  orbit = await bm(page, (h) => h.measureOrbit(60, 2))
  const fullInfo = await bm(page, (h) => h.renderInfo())
  lines.push(
    `[perf] orbit, all 2M blocks as instances: frame interval median ${orbit.intervalMedianMs.toFixed(1)} ms, p95 ${orbit.intervalP95Ms.toFixed(1)} ms; ` +
      `unthrottled frame cost mean ${orbit.renderMeanMs.toFixed(2)} ms; ${fullInfo.triangles.toLocaleString('en-US')} triangles in ${fullInfo.calls} draw calls`,
  )
  lines.push(`[perf] renderer: ${info.renderer}${note}`)
  for (const text of lines) console.log(text)
  if (!software) {
    expect(first.timing.toFrameMs).toBeLessThan(5_000)
    expect(pageLoadMs).toBeLessThan(15_000)
    // 30 fps while orbiting the full 2M-instance set (loose: headless GPU).
    expect(orbit.renderMeanMs).toBeLessThan(50)
  }
})

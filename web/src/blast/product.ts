/** Which product the page shows: the block modeller (default) or Blast BakuMine (`?app=blast`). */
export type Product = 'modeller' | 'blast'

export function productFromSearch(search: string): Product {
  return new URLSearchParams(search).get('app') === 'blast' ? 'blast' : 'modeller'
}

/** The query string that opens `product`, every other parameter kept. */
export function searchForProduct(search: string, product: Product): string {
  const params = new URLSearchParams(search)
  if (product === 'blast') params.set('app', 'blast')
  else {
    params.delete('app')
    params.delete('blast')
  }
  const text = params.toString()
  return text ? `?${text}` : ''
}

/** `?blast=demo`: do not ask the API, show the browser-made demo bundle. */
export function demoRequested(search: string): boolean {
  return new URLSearchParams(search).get('blast') === 'demo'
}

/**
 * Switches product with a page load: each product owns the whole page (its own WebGL
 * canvas and stores), and the choice lives in the URL so that it survives a reload.
 */
export function openProduct(product: Product): void {
  if (productFromSearch(window.location.search) === product) return
  window.location.assign(window.location.pathname + searchForProduct(window.location.search, product) + window.location.hash)
}

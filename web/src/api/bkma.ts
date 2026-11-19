/**
 * Decoder of the BKMA container: the one binary format of every bulk-data endpoint.
 * The layout is documented in bakumine/api/binary.py:
 *
 *   "BKMA" | u32 version | u32 H | H bytes of JSON header (space padded) | array data
 *
 * All little-endian; the data section and every array in it start at a multiple of
 * 8 bytes, so each array is a typed-array VIEW on the response buffer (no copy).
 */

export type BkmaDtype = 'f8' | 'f4' | 'i4' | 'u4' | 'i2' | 'u2' | 'i1' | 'u1'

export type BkmaArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array

interface TypedArrayConstructor<T extends BkmaArray> {
  new (buffer: ArrayBuffer, byteOffset: number, length: number): T
  readonly BYTES_PER_ELEMENT: number
}

const CONSTRUCTORS = {
  f8: Float64Array,
  f4: Float32Array,
  i4: Int32Array,
  u4: Uint32Array,
  i2: Int16Array,
  u2: Uint16Array,
  i1: Int8Array,
  u1: Uint8Array,
} as const satisfies Record<BkmaDtype, TypedArrayConstructor<BkmaArray>>

type ArrayOf<D extends BkmaDtype> = InstanceType<(typeof CONSTRUCTORS)[D]>

export interface BkmaEntry {
  name: string
  dtype: BkmaDtype
  shape: number[]
  /** Flat data, C order. */
  data: BkmaArray
}

export class BkmaError extends Error {}

const MAGIC = [0x42, 0x4b, 0x4d, 0x41] // "BKMA"
const VERSION = 1

/** True on little-endian machines (every browser platform in practice). */
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1

export class BkmaPayload {
  readonly kind: string
  readonly meta: Record<string, unknown>
  readonly byteLength: number
  private readonly entries: Map<string, BkmaEntry>

  constructor(kind: string, meta: Record<string, unknown>, entries: BkmaEntry[], byteLength: number) {
    this.kind = kind
    this.meta = meta
    this.entries = new Map(entries.map((e) => [e.name, e]))
    this.byteLength = byteLength
  }

  names(): string[] {
    return [...this.entries.keys()]
  }

  has(name: string): boolean {
    return this.entries.has(name)
  }

  entry(name: string): BkmaEntry {
    const entry = this.entries.get(name)
    if (!entry) throw new BkmaError(`payload has no array "${name}"`)
    return entry
  }

  /**
   * The array `name`, checked to have dtype `dtype` and, when `columns` is given,
   * shape (n, columns). Returns the flat typed array.
   */
  get<D extends BkmaDtype>(name: string, dtype: D, columns?: number): ArrayOf<D> {
    const entry = this.entry(name)
    if (entry.dtype !== dtype) {
      throw new BkmaError(`array "${name}" is ${entry.dtype}, expected ${dtype}`)
    }
    const ok =
      columns === undefined
        ? entry.shape.length === 1
        : entry.shape.length === 2 && entry.shape[1] === columns
    if (!ok) {
      const want = columns === undefined ? '(n)' : `(n, ${columns})`
      throw new BkmaError(`array "${name}" has shape [${entry.shape.join(', ')}], expected ${want}`)
    }
    return entry.data as ArrayOf<D>
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Parses one BKMA payload. Throws `BkmaError` on anything malformed. */
export function decodeBkma(buffer: ArrayBuffer): BkmaPayload {
  if (!LITTLE_ENDIAN) throw new BkmaError('BKMA needs a little-endian machine')
  if (buffer.byteLength < 12) throw new BkmaError('not a BKMA payload: too short')
  const bytes = new Uint8Array(buffer)
  if (MAGIC.some((b, i) => bytes[i] !== b)) throw new BkmaError('not a BKMA payload: bad magic')
  const head = new DataView(buffer)
  const version = head.getUint32(4, true)
  if (version !== VERSION) throw new BkmaError(`unsupported BKMA version ${version}`)
  const dataStart = 12 + head.getUint32(8, true)
  if (dataStart > buffer.byteLength) throw new BkmaError('BKMA header is truncated')

  let header: unknown
  try {
    header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, dataStart)))
  } catch {
    throw new BkmaError('BKMA header is not valid JSON')
  }
  if (!isRecord(header) || typeof header.kind !== 'string' || !Array.isArray(header.arrays)) {
    throw new BkmaError('BKMA header lacks "kind" or "arrays"')
  }

  const entries: BkmaEntry[] = []
  for (const raw of header.arrays as unknown[]) {
    if (!isRecord(raw)) throw new BkmaError('BKMA array entry is not an object')
    const { name, dtype, shape, offset, nbytes } = raw
    if (typeof name !== 'string' || typeof dtype !== 'string' || !(dtype in CONSTRUCTORS)) {
      throw new BkmaError(`BKMA array entry has an unknown dtype: ${String(dtype)}`)
    }
    if (
      !Array.isArray(shape) ||
      !shape.every((n) => Number.isInteger(n) && n >= 0) ||
      typeof offset !== 'number' ||
      typeof nbytes !== 'number'
    ) {
      throw new BkmaError(`BKMA array "${name}" has a malformed entry`)
    }
    const ctor = CONSTRUCTORS[dtype as BkmaDtype]
    const count = (shape as number[]).reduce((a, b) => a * b, 1)
    const start = dataStart + offset
    if (count * ctor.BYTES_PER_ELEMENT !== nbytes || start + nbytes > buffer.byteLength) {
      throw new BkmaError(`BKMA array "${name}" does not fit the payload`)
    }
    // The writer aligns to 8 bytes; copy only if a foreign writer did not.
    const data =
      start % ctor.BYTES_PER_ELEMENT === 0
        ? new ctor(buffer, start, count)
        : new ctor(buffer.slice(start, start + nbytes), 0, count)
    entries.push({ name, dtype: dtype as BkmaDtype, shape: shape as number[], data })
  }
  const meta = isRecord(header.meta) ? header.meta : {}
  return new BkmaPayload(header.kind, meta, entries, buffer.byteLength)
}

/** Writer, for tests and client-generated data. Same layout as the Python side. */
export function encodeBkma(
  kind: string,
  meta: Record<string, unknown>,
  arrays: { name: string; dtype: BkmaDtype; shape: number[]; data: BkmaArray }[],
): ArrayBuffer {
  const pad = (n: number) => (8 - (n % 8)) % 8
  let offset = 0
  const entries = arrays.map((a) => {
    const entry = { name: a.name, dtype: a.dtype, shape: a.shape, offset, nbytes: a.data.byteLength }
    offset += a.data.byteLength + pad(a.data.byteLength)
    return entry
  })
  let json = JSON.stringify({ kind, meta, arrays: entries })
  let header = new TextEncoder().encode(json)
  json += ' '.repeat(pad(12 + header.length))
  header = new TextEncoder().encode(json)
  const out = new Uint8Array(12 + header.length + offset)
  out.set(MAGIC, 0)
  const view = new DataView(out.buffer)
  view.setUint32(4, VERSION, true)
  view.setUint32(8, header.length, true)
  out.set(header, 12)
  arrays.forEach((a, i) => {
    out.set(new Uint8Array(a.data.buffer, a.data.byteOffset, a.data.byteLength), 12 + header.length + entries[i].offset)
  })
  return out.buffer
}

"""Block table with octree sub-blocks (research.md sections 6.2 and 6.3).

A :class:`BlockModel` is a :class:`BlockModelDefinition` plus a flat, columnar
table with one row per *leaf* block. The leaves tile the model box exactly:
every point of the model is in exactly one leaf.

Structure columns (8 bytes per block)
-------------------------------------
``parent`` : int32
    Flat index of the parent block, ``(i * ny + j) * nz + k``.
``level`` : uint8
    Number of splits below the parent; 0 = the parent block itself.
``sub`` : uint8, shape (n, 3)
    Position of the block's **minimum corner** inside its parent, along
    (i, j, k), counted in the finest cells of each axis
    (``block_size[a] / 2**max_depth[a]``). Always a multiple of the block's
    own size in those cells.

Why integer sub-indices rather than a Morton / locational code: centroid,
size and bounds follow from them with one integer expression and a single
multiplication per axis, without decoding, and the three bytes are as compact
as a 24-bit code. The Morton code is still what orders the table; it is
computed from the sub-indices when needed (:meth:`BlockModel.keys`).

Exact geometry
--------------
Nothing geometric is stored. With ``D = max_depth[a]``, ``L = level``,
``l = min(L, D)`` and the integer ``I = (parent index along a) * 2**D + sub``::

    size      = block_size / 2**l                       exact (power of two)
    min face  = I * (block_size / 2**D)
    centroid  = (2 I + 2**(D - l)) * (block_size / 2**(D + 1))

``2 I + 2**(D - l)`` is an integer and ``block_size / 2**(D + 1)`` is exact,
so each local coordinate is the correctly rounded value of the true rational
number: one rounding, no accumulation, the same value however the block was
reached. World coordinates are ``origin + R @ local``
(:meth:`BlockModelDefinition.local_to_world`).

Splitting
---------
:meth:`BlockModel.split` replaces a block by its children, which inherit all
attribute values. A split from level ``L`` to ``L + 1`` halves every axis
whose ``max_depth`` is greater than ``L`` and leaves the other axes whole. So

* with equal depths on all axes this is a plain octree (8 children);
* with unequal depths the first splits are octree splits and the later ones
  halve only the deeper axes (4 or 2 children). ``max_depth = (3, 3, 0)``
  never splits Z: every split is a quadtree split in plan.

That is the only anisotropy supported: which axes a split halves is fixed by
the level, never chosen per block. A block at ``level == max(max_depth)``
cannot be split.

:meth:`BlockModel.merge` is the inverse: a complete set of sibling leaves with
identical attribute values becomes their parent cell again.

Ordering
--------
Rows are sorted by ``(parent, Morton code)``: parents in flat order, and
inside a parent the Z-order curve with ``i`` as the most significant bit, then
``j``, then ``k`` -- the same significance as the flat index. Equivalently,
rows are sorted by :meth:`BlockModel.keys`. Consequences: the order depends
only on the set of leaves, not on the history of splits and merges; the
children of a split take the place of their parent, in child order
``(i, j, k) = (0,0,0), (0,0,1), (0,1,0), ... (1,1,1)``; siblings are always
contiguous; a model without sub-blocks is in flat order, so its columns can be
viewed as ``(nx, ny, nz)`` arrays without copying (:meth:`BlockModel.dense_view`).

Row numbers change when blocks are split or merged. :meth:`BlockModel.block_ids`
gives identifiers that do not.

Half-open rule
--------------
As for parents (:mod:`bakumine.core.blockmodel.definition`): a block contains
``min face <= local < max face`` on every axis, with the faces computed as
above. :meth:`BlockModel.locate` implements exactly this.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np

from bakumine.core.blockmodel.attributes import BUILTIN_VARIABLES, Attribute, check_name
from bakumine.core.blockmodel.definition import BlockModelDefinition, cell_index
from bakumine.core.blockmodel.expressions import (
    CategoryValues,
    Expression,
    ExpressionError,
    ExpressionWarning,
    parse,
)

__all__ = ["BlockModel", "SplitResult"]

_LEVEL_BITS = 4  # block id = key * 16 + level


class SplitResult(NamedTuple):
    """Result of :meth:`BlockModel.split`.

    Attributes
    ----------
    source : (n_new,) int64
        For every row of the table after the split, the row it had (or its
        parent block had) before the split.
    is_child : (n_new,) bool
        True for the rows created by this split.
    n_split : int
        Number of blocks that were split.
    """

    source: np.ndarray
    is_child: np.ndarray
    n_split: int


class BlockModel:
    """Block table: a definition, structure columns and attribute columns.

    ``BlockModel(definition)`` creates the full model of parent blocks, in
    flat order, with no attributes.

    Attributes
    ----------
    definition : BlockModelDefinition
    parent, level, sub : structure columns (read-only views; see the module
        docstring). Change them only through :meth:`split` and :meth:`merge`.
    """

    def __init__(self, definition: BlockModelDefinition):
        self.definition = definition
        n = definition.n_parents
        self._parent = np.arange(n, dtype=np.int32)
        self._level = np.zeros(n, dtype=np.uint8)
        self._sub = np.zeros((n, 3), dtype=np.uint8)
        self._columns: dict[str, np.ndarray] = {}
        self._attributes: dict[str, Attribute] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._init_tables()

    def _init_tables(self) -> None:
        depth = np.array(self.definition.max_depth, dtype=np.int64)
        self._depth = depth
        self._max_level = int(depth.max())
        levels = np.arange(self._max_level + 1)[:, None]
        clipped = np.minimum(levels, depth[None, :])
        #: size of a block of each level along each axis, in finest cells
        self._size_units = (1 << (depth[None, :] - clipped)).astype(np.int64)
        size = np.array(self.definition.block_size)
        self._fine = size / (1 << depth)  # finest cell size, exact
        self._half_fine = size / (1 << (depth + 1))
        self._size_table = self._size_units * self._fine  # exact
        #: children created by splitting a block of each level (0 at the last level)
        active = depth[None, :] > levels
        self._n_children = np.where(active.any(axis=1), 1 << active.sum(axis=1), 0).astype(np.int64)
        self._active = active

    @classmethod
    def from_arrays(
        cls, definition: BlockModelDefinition, parent: Any, level: Any, sub: Any
    ) -> BlockModel:
        """Model from existing structure columns (e.g. read from disk).

        The rows must be in the documented order and tile the model; this is
        verified (:meth:`check_invariants`). Attributes are added afterwards.
        """
        model = cls.__new__(cls)
        model.definition = definition
        model._parent = np.array(parent, dtype=np.int32).reshape(-1)
        model._level = np.array(level, dtype=np.uint8).reshape(-1)
        model._sub = np.array(sub, dtype=np.uint8).reshape(-1, 3)
        if not len(model._parent) == len(model._level) == len(model._sub):
            raise ValueError("parent, level and sub must have the same length")
        model._columns = {}
        model._attributes = {}
        model._cache = {}
        model._init_tables()
        model.check_invariants()
        return model

    # ------------------------------------------------------------------ basics

    @property
    def parent(self) -> np.ndarray:
        return _readonly(self._parent)

    @property
    def level(self) -> np.ndarray:
        return _readonly(self._level)

    @property
    def sub(self) -> np.ndarray:
        return _readonly(self._sub)

    @property
    def n_blocks(self) -> int:
        """Number of leaf blocks (rows)."""
        return len(self._parent)

    def __len__(self) -> int:
        return len(self._parent)

    @property
    def is_dense(self) -> bool:
        """True when there are no sub-blocks: row number == flat parent index."""
        return len(self._parent) == self.definition.n_parents

    @property
    def max_level(self) -> int:
        """Deepest level a block can have, ``max(max_depth)``."""
        return self._max_level

    def __repr__(self) -> str:
        return (
            f"BlockModel(shape={self.definition.shape}, n_blocks={self.n_blocks}, "
            f"attributes={list(self._columns)})"
        )

    @property
    def nbytes(self) -> int:
        """Bytes held by structure and attribute columns (caches excluded)."""
        total = self._parent.nbytes + self._level.nbytes + self._sub.nbytes
        return total + sum(c.nbytes for c in self._columns.values())

    @property
    def cache_nbytes(self) -> int:
        """Bytes held by cached derived arrays (keys, centroids); see :meth:`clear_cache`."""
        return sum(c.nbytes for c in self._cache.values())

    def clear_cache(self) -> None:
        """Drop cached derived arrays (they are rebuilt on demand)."""
        self._cache.clear()

    def _rows(self, where: Any) -> Any:
        """Normalise a row selection: None (all), a bool mask or an int64 index array."""
        if where is None:
            return None
        sel = np.asarray(where)
        if sel.size == 0 and sel.dtype != bool:
            return np.zeros(0, dtype=np.int64)
        if sel.dtype == bool:
            if sel.shape != (self.n_blocks,):
                raise ValueError(f"mask must have shape ({self.n_blocks},), got {sel.shape}")
            return sel
        if sel.dtype.kind not in "iu":
            raise TypeError("select blocks with a boolean mask or integer row numbers")
        sel = sel.astype(np.int64).reshape(-1)
        if sel.size and (sel.min() < 0 or sel.max() >= self.n_blocks):
            raise IndexError("block row number out of range")
        return sel

    # ---------------------------------------------------------------- geometry

    def _take(self, where: Any):
        rows = self._rows(where)
        if rows is None:
            return self._parent, self._level, self._sub
        return self._parent[rows], self._level[rows], self._sub[rows]

    def ijk(self, where: Any = None) -> np.ndarray:
        """(n, 3) int64 parent ``ijk`` of each block."""
        return self.definition.unflat_index(self._take(where)[0])

    def _fine_index(self, where: Any):
        parent, level, sub = self._take(where)
        ijk = self.definition.unflat_index(parent)
        return (ijk << self._depth) + sub, self._size_units[level]

    def sizes(self, where: Any = None) -> np.ndarray:
        """(n, 3) block sizes along the model axes, ``block_size / 2**min(level, max_depth)``."""
        return self._size_table[self._take(where)[1]]

    def volumes(self, where: Any = None) -> np.ndarray:
        """(n,) block volumes."""
        s = self.sizes(where)
        return s[:, 0] * s[:, 1] * s[:, 2]

    def volume_units(self, where: Any = None) -> np.ndarray:
        """(n,) int64 block volumes counted in finest cells (exact).

        The model holds ``n_parents * 2**sum(max_depth)`` finest cells.
        """
        u = self._size_units[self._take(where)[1]]
        return u[:, 0] * u[:, 1] * u[:, 2]

    def bounds_local(self, where: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)``, each (n, 3): minimum and maximum corners in the model frame."""
        index, units = self._fine_index(where)
        return index * self._fine, (index + units) * self._fine

    def centroids_local(self, where: Any = None) -> np.ndarray:
        """(n, 3) centroids in the model frame; exact (see the module docstring)."""
        index, units = self._fine_index(where)
        return (2 * index + units) * self._half_fine

    def centroids_grid(self, where: Any = None) -> np.ndarray:
        """(n, 3) centroids relative to the centroid of parent (0, 0, 0), in the model frame.

        For a parent block this is exactly ``(i * dx, j * dy, k * dz)``, the
        coordinates of the regular grid used by
        :func:`bakumine.core.geometry.inside.grid_in_solid`.
        """
        index, units = self._fine_index(where)
        return (2 * index + units - (1 << self._depth)) * self._half_fine

    def centroids(self, where: Any = None) -> np.ndarray:
        """(n, 3) world centroids (east, north, up).

        The full array (``where=None``) is cached and returned read-only.
        """
        if where is not None:
            return self.definition.local_to_world(self.centroids_local(where))
        if "centroids" not in self._cache:
            c = self.definition.local_to_world(self.centroids_local())
            c.setflags(write=False)
            self._cache["centroids"] = c
        return self._cache["centroids"]

    def corners(self, where: Any = None) -> np.ndarray:
        """(n, 8, 3) world corners; order as :meth:`BlockModelDefinition.box_corners`."""
        return self.definition.box_corners(*self.bounds_local(where))

    # ---------------------------------------------------------------- ordering

    def _make_keys(self, parent: np.ndarray, sub: np.ndarray) -> np.ndarray:
        """Sort key ``parent * 8**max_level + Morton(sub)`` (int64)."""
        morton = np.zeros(len(parent), dtype=np.int64)
        cols = [sub[:, a].astype(np.int64) for a in range(3)]
        for lev in range(1, self._max_level + 1):
            morton <<= 3
            for a in range(3):
                shift = int(self._depth[a]) - lev
                if shift >= 0:
                    morton |= ((cols[a] >> shift) & 1) << (2 - a)
        return (parent.astype(np.int64) << (3 * self._max_level)) | morton

    def keys(self) -> np.ndarray:
        """(n,) int64 sort keys, strictly increasing down the table (cached, read-only).

        ``key = parent * 8**L + m`` with ``L = max(max_depth)`` and ``m`` the
        Morton code of ``sub``: for each level from 1 to ``L`` three bits
        ``(i, j, k)``, the bit of an axis being that level's bit of its
        sub-index (0 for levels beyond the axis' ``max_depth``). A block of
        level ``l`` owns the key range ``[key, key + 8**(L - l))``.
        """
        if "keys" not in self._cache:
            k = self._make_keys(self._parent, self._sub)
            k.setflags(write=False)
            self._cache["keys"] = k
        return self._cache["keys"]

    def block_ids(self, where: Any = None) -> np.ndarray:
        """(n,) int64 identifiers, ``key * 16 + level``.

        The id of a block depends only on the block itself (its parent, level
        and position), so it survives splits and merges of other blocks and
        saving / loading. Ids increase down the table.
        """
        ids = (self.keys() << _LEVEL_BITS) | self._level
        rows = self._rows(where)
        return ids if rows is None else ids[rows]

    def rows_of(self, block_ids: Any) -> np.ndarray:
        """Row numbers of blocks given by id; -1 for ids that are not (or no longer) leaves."""
        wanted = np.asarray(block_ids, dtype=np.int64)
        ids = self.block_ids()
        pos = np.searchsorted(ids, wanted)
        pos_ok = np.minimum(pos, len(ids) - 1)
        return np.where(ids[pos_ok] == wanted, pos_ok, -1)

    # ------------------------------------------------------------------ locate

    def locate(self, points: Any) -> np.ndarray:
        """Row of the leaf block containing each world point ``(..., 3)``; -1 outside the model.

        Half-open rule: see the module docstring.
        """
        d = self.definition
        local = d.world_to_local(points)
        inside = np.ones(local.shape[:-1], dtype=bool)
        fine = []
        for a in range(3):
            idx, ok = cell_index(local[..., a], float(self._fine[a]), d.shape[a] << self._depth[a])
            inside &= ok
            fine.append(np.where(ok, idx, 0))
        ijk = np.stack([f >> self._depth[a] for a, f in enumerate(fine)], axis=-1)
        flat = d.flat_index(ijk)
        if self.is_dense:
            rows = flat
        else:
            sub = np.stack(
                [f & ((1 << self._depth[a]) - 1) for a, f in enumerate(fine)], axis=-1
            ).reshape(-1, 3)
            query = self._make_keys(flat.reshape(-1), sub)
            rows = (np.searchsorted(self.keys(), query, side="right") - 1).reshape(flat.shape)
        return np.where(inside, rows, -1)

    # ------------------------------------------------------------- split/merge

    def split(self, where: Any, *, strict: bool = True) -> SplitResult:
        """Split the selected blocks into their children.

        Parameters
        ----------
        where : bool mask over the rows, or integer row numbers.
        strict : bool
            Blocks already at the deepest level cannot be split. With
            ``strict=True`` (default) selecting one raises ``ValueError``;
            with ``strict=False`` they are left as they are.

        Each selected block is replaced, in place in the table order, by its
        8 children (4 or 2 where only some axes are halved; see the module
        docstring), which inherit the values of **all** attributes. Total
        volume and every attribute's volume-weighted sum are unchanged.

        Returns
        -------
        SplitResult ``(source, is_child, n_split)``.
        """
        n = self.n_blocks
        rows = self._rows(where)
        selected = np.zeros(n, dtype=bool)
        if rows is not None:
            selected[rows] = True
        else:
            selected[:] = True
        children = self._n_children[self._level]
        if strict and (selected & (children == 0)).any():
            count = int((selected & (children == 0)).sum())
            raise ValueError(
                f"{count} selected block(s) are at the deepest level "
                f"(max_depth={self.definition.max_depth}) and cannot be split"
            )
        selected &= children > 0
        n_split = int(selected.sum())
        if n_split == 0:
            return SplitResult(np.arange(n), np.zeros(n, dtype=bool), 0)
        counts = np.where(selected, children, 1)
        if n + int((counts - 1).sum()) >= 2**31:
            raise ValueError("more than 2**31 - 1 blocks are not supported")
        source = np.repeat(np.arange(n), counts)
        first = np.cumsum(counts) - counts
        child_no = np.arange(len(source)) - first[source]
        is_child = selected[source]
        old_level = self._level[source]
        sub = self._sub[source]
        # bit of each axis in the child number: i is the most significant of the halved axes
        active = self._active[old_level] & is_child[:, None]
        shift = np.zeros(len(source), dtype=np.int64)
        for a in (2, 1, 0):
            bit = (child_no >> shift) & 1
            # the child is half the block: offset by the child's size in finest cells
            half = self._size_units[np.minimum(old_level + 1, self._max_level), a]
            sub[:, a] += (np.where(active[:, a], bit * half, 0)).astype(np.uint8)
            shift += active[:, a]
        self._parent = self._parent[source]
        self._level = (old_level + is_child).astype(np.uint8)
        self._sub = sub
        for name, column in self._columns.items():
            self._columns[name] = column[source]
        self._cache.clear()
        return SplitResult(source, is_child, n_split)

    def merge(
        self,
        attributes: Iterable[str] | None = None,
        where: Any = None,
        *,
        recursive: bool = True,
    ) -> int:
        """Merge complete sets of sibling leaves whose attribute values are identical.

        Parameters
        ----------
        attributes : names of the attributes that must be identical across
            the siblings. Default: all attributes. NaN equals NaN and missing
            equals missing here. With an explicit subset, the merged block
            takes every attribute's value from its first child (the one at
            the minimum corner), so differing values of the other attributes
            are lost; pass ``[]`` to merge on structure alone.
        where : optional bool mask or row numbers; only sets whose siblings
            are all selected are merged.
        recursive : bool
            Keep merging the merged blocks with their own siblings (default),
            up to parent blocks. With a ``where`` selection, merged blocks
            count as selected.

        A set is merged only when all its siblings exist as leaves of the same
        level (8, or 4 / 2 for a partial split). Returns the number of merges
        (sibling sets replaced by one block).
        """
        names = list(self._columns) if attributes is None else list(attributes)
        for name in names:
            self._require(name)
        rows = self._rows(where)
        allowed = None
        if rows is not None:
            allowed = np.zeros(self.n_blocks, dtype=bool)
            allowed[rows] = True
        total = 0
        while True:
            merged, keep = self._merge_pass(names, allowed)
            total += merged
            if merged == 0:
                break
            self._parent = self._parent[keep]
            self._level = self._level[keep]
            self._sub = self._sub[keep]
            for name, column in self._columns.items():
                self._columns[name] = column[keep]
            if allowed is not None:
                allowed = allowed[keep]
            self._cache.clear()
            if not recursive:
                break
        return total

    def _merge_pass(self, names: list[str], allowed: np.ndarray | None):
        n = self.n_blocks
        level = self._level
        if self._max_level == 0 or not (level > 0).any():
            return 0, None
        keys = self.keys()
        drop = np.zeros(n, dtype=bool)
        merged = 0
        firsts = []
        for lev in range(1, self._max_level + 1):
            n_sib = int(self._n_children[lev - 1])
            span = 1 << (3 * (self._max_level - lev + 1))  # key range of the cell one level up
            cand = np.flatnonzero(level == lev)
            # first child: its key is the start of the enclosing cell's key range
            cand = cand[(keys[cand] & (span - 1)) == 0]
            cand = cand[cand + n_sib <= n]
            if len(cand) == 0:
                continue
            ok = np.ones(len(cand), dtype=bool) if allowed is None else allowed[cand].copy()
            for m in range(1, n_sib):
                other = cand + m
                ok &= (level[other] == lev) & (keys[other] - keys[cand] < span)
                if allowed is not None:
                    ok &= allowed[other]
            for name in names:
                column = self._columns[name]
                base = column[cand]
                for m in range(1, n_sib):
                    value = column[cand + m]
                    same = value == base
                    if column.dtype.kind == "f":
                        same |= (value != value) & (base != base)
                    ok &= same
            cand = cand[ok]
            if len(cand) == 0:
                continue
            for m in range(1, n_sib):
                drop[cand + m] = True
            firsts.append(cand)
            merged += len(cand)
        if merged == 0:
            return 0, None
        first = np.concatenate(firsts)
        self._level = level.copy()
        self._level[first] -= 1
        return merged, ~drop

    # -------------------------------------------------------------- invariants

    def check_invariants(self) -> None:
        """Verify that the table is a valid, ordered, exact tiling; ``ValueError`` if not.

        Checked with integer arithmetic only: structure values are in range
        and aligned; keys are strictly increasing and no block starts inside
        the key range of the previous one (blocks are disjoint); and the block
        volumes, counted in finest cells, add up to the model volume. Disjoint
        blocks inside the model whose volumes sum to the model's leave no gap.
        """
        d = self.definition
        n = self.n_blocks
        parent, level, sub = self._parent, self._level, self._sub
        if not (parent.dtype == np.int32 and level.dtype == np.uint8 and sub.dtype == np.uint8):
            raise ValueError("structure columns have the wrong dtype")
        if n == 0:
            raise ValueError("the block table is empty")
        if parent.min() < 0 or parent.max() >= d.n_parents:
            raise ValueError("parent index out of range")
        if level.max() > self._max_level:
            raise ValueError("block level exceeds max(max_depth)")
        units = self._size_units[level]
        s = sub.astype(np.int64)
        if (s % units != 0).any() or (s + units > (1 << self._depth)).any():
            raise ValueError("sub-index not aligned to the block size or outside the parent")
        keys = self._make_keys(parent, sub)
        span = np.int64(1) << (3 * (self._max_level - level.astype(np.int64)))
        if n > 1 and (keys[1:] < keys[:-1] + span[:-1]).any():
            raise ValueError("blocks overlap or the table is not in (parent, Morton) order")
        total = int((units[:, 0] * units[:, 1] * units[:, 2]).sum())
        expected = d.n_parents << int(self._depth.sum())
        if total != expected:
            raise ValueError(
                f"blocks do not fill the model: {total} of {expected} finest cells covered"
            )
        for name, column in self._columns.items():
            if len(column) != n:
                raise ValueError(f"attribute {name!r} has {len(column)} values for {n} blocks")

    # -------------------------------------------------------------- attributes

    @property
    def attribute_names(self) -> list[str]:
        """Attribute names in creation order."""
        return list(self._columns)

    @property
    def attributes(self) -> dict[str, Attribute]:
        """Name -> :class:`Attribute` description (a copy of the mapping)."""
        return dict(self._attributes)

    def __contains__(self, name: object) -> bool:
        return name in self._columns

    def _require(self, name: str) -> Attribute:
        try:
            return self._attributes[name]
        except KeyError:
            raise KeyError(
                f"no attribute {name!r}; attributes: {', '.join(self._columns) or '(none)'}"
            ) from None

    def attribute(self, name: str) -> Attribute:
        """Description of one attribute (kind, dtype, background, metadata, categories)."""
        return self._require(name)

    def add_attribute(
        self,
        name: str,
        kind: str = "float",
        *,
        background: Any = None,
        dtype: Any = None,
        description: str = "",
        unit: str = "",
        decimals: int | None = None,
        categories: Iterable[str] = (),
    ) -> Attribute:
        """Add an attribute; every block gets the background value.

        Parameters
        ----------
        name : identifier, not reserved (see :mod:`bakumine.core.blockmodel.attributes`).
        kind : ``"float"`` (NaN = missing), ``"int"`` or ``"category"``
            (dictionary-encoded strings, ``None`` = missing).
        background : default value; NaN / 0 / missing when omitted.
        dtype : storage dtype; float64 / int32 / int32 codes when omitted
            (float32, int8..int64 and int8/int16 codes are allowed).
        description, unit, decimals : optional metadata.
        categories : initial dictionary of a category attribute, in code order.
        """
        if name in self._columns:
            raise ValueError(f"attribute {name!r} already exists")
        spec = Attribute(name, kind, dtype, background, description, unit, decimals,
                         list(categories))  # fmt: skip
        self._columns[name] = np.full(self.n_blocks, spec.background_stored, dtype=spec.dtype)
        self._attributes[name] = spec
        return spec

    def delete_attribute(self, name: str) -> None:
        """Remove an attribute and its values."""
        self._require(name)
        del self._columns[name]
        del self._attributes[name]

    def rename_attribute(self, old: str, new: str) -> None:
        """Rename an attribute; values, metadata and position in the list are kept."""
        spec = self._require(old)
        if new == old:
            return
        check_name(new)
        if new in self._columns:
            raise ValueError(f"attribute {new!r} already exists")
        spec.name = new
        self._columns = {(new if k == old else k): v for k, v in self._columns.items()}
        self._attributes = {(new if k == old else k): v for k, v in self._attributes.items()}

    def __getitem__(self, name: str) -> np.ndarray:
        """The stored column itself (codes for a category attribute), writable in place."""
        self._require(name)
        return self._columns[name]

    def get(self, name: str, where: Any = None, *, decode: bool = False) -> np.ndarray:
        """Values of an attribute.

        Parameters
        ----------
        where : optional bool mask or row numbers; default all blocks.
        decode : for a category attribute, return strings (object array,
            ``None`` where missing) instead of codes.

        Without ``where`` and ``decode`` this is the stored column, not a copy.
        """
        spec = self._require(name)
        rows = self._rows(where)
        values = self._columns[name] if rows is None else self._columns[name][rows]
        if decode:
            return spec.decode(values)
        return values

    def set(self, name: str, values: Any, where: Any = None) -> None:
        """Assign values to an attribute.

        Parameters
        ----------
        values : a scalar, or one value per selected block. For a category
            attribute: strings / ``None`` (new strings are added to the
            dictionary) or integer codes. For a float attribute NaN marks
            missing.
        where : optional bool mask or row numbers; default all blocks.
        """
        spec = self._require(name)
        rows = self._rows(where)
        stored = spec.to_stored(values)
        column = self._columns[name]
        if rows is None:
            column[:] = stored
        else:
            column[rows] = stored

    def reset(self, name: str, where: Any = None) -> None:
        """Set the selected blocks (default all) back to the background value."""
        spec = self._require(name)
        rows = self._rows(where)
        self._columns[name][slice(None) if rows is None else rows] = spec.background_stored

    def is_missing(self, name: str, where: Any = None) -> np.ndarray:
        """True where a float attribute is NaN or a category is missing; never for an int."""
        spec = self._require(name)
        values = self.get(name, where)
        if spec.kind == "float":
            return np.isnan(values)
        if spec.kind == "category":
            return values < 0
        return np.zeros(len(values), dtype=bool)

    # ------------------------------------------------------------- expressions

    def variables(self, names: Iterable[str], where: Any = None) -> dict[str, Any]:
        """Values of expression variables for the selected blocks.

        ``names`` may contain attribute names and the built-in variables
        ``x``, ``y``, ``z`` (world centroid), ``dx``, ``dy``, ``dz`` (block
        size along the model axes) and ``volume``. Category attributes are
        returned as :class:`~bakumine.core.blockmodel.expressions.CategoryValues`.
        Unknown names raise ``ExpressionError``.
        """
        rows = self._rows(where)
        out: dict[str, Any] = {}
        names = list(names)
        centroid = sizes = None
        for name in names:
            if name in self._columns:
                spec = self._attributes[name]
                values = self._columns[name] if rows is None else self._columns[name][rows]
                if spec.kind == "category":
                    values = CategoryValues(values, tuple(spec.categories))
                out[name] = values
            elif name in ("x", "y", "z"):
                if centroid is None:
                    centroid = self.centroids(rows)
                out[name] = centroid[:, "xyz".index(name)]
            elif name in ("dx", "dy", "dz", "volume"):
                if sizes is None:
                    sizes = self.sizes(rows)
                if name == "volume":
                    out[name] = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]
                else:
                    out[name] = sizes[:, ("dx", "dy", "dz").index(name)]
            else:
                known = ", ".join([*self._columns, *BUILTIN_VARIABLES])
                raise ExpressionError(f"unknown name {name!r}; available: {known}")
        return out

    def evaluate(
        self, expression: str | Expression, where: Any = None, *, return_invalid: bool = False
    ) -> Any:
        """Evaluate an expression for the selected blocks (default all).

        The language is described in :mod:`bakumine.core.blockmodel.expressions`;
        it is parsed with a strict whitelist and never passed to ``eval``.

        Returns one value per selected block: a bool array for a condition,
        float64 otherwise. Elements where an operation gave inf / NaN from
        finite operands (division by zero, log of a non-positive number...)
        are counted: with ``return_invalid=True`` the result is ``(values,
        count)``; otherwise a non-zero count is reported as an
        :class:`~bakumine.core.blockmodel.expressions.ExpressionWarning`.
        """
        expr = parse(expression)
        rows = self._rows(where)
        n = self.n_blocks if rows is None else (
            int(rows.sum()) if rows.dtype == bool else len(rows)
        )
        result = expr.evaluate(self.variables(expr.names, rows))
        values = np.asarray(result.values)
        n_invalid = result.n_invalid
        if values.shape != (n,):
            n_invalid *= n if values.ndim == 0 else 1
            values = np.broadcast_to(values, (n,)).copy()
        if return_invalid:
            return values, n_invalid
        if n_invalid:
            warnings.warn(
                f"expression {expr.source!r}: {n_invalid} value(s) became inf or NaN "
                "(division by zero, log or sqrt outside the domain, or overflow)",
                ExpressionWarning,
                stacklevel=2,
            )
        return values

    def calculate(
        self, name: str, expression: str | Expression, where: Any = None, **attribute: Any
    ) -> int:
        """Fill an attribute from an expression (a calculated attribute).

        If ``name`` does not exist it is created first (``**attribute`` are
        the arguments of :meth:`add_attribute`; kind ``"float"`` by default,
        and the description defaults to ``"= <expression>"``). The expression
        is evaluated for the selected blocks (default all) and stored there;
        other blocks keep their value (the background for a new attribute).
        The values are computed once: they do not follow later changes of
        the attributes they were computed from.

        A condition stores 1 / 0. An integer attribute needs whole, finite
        results. A category attribute cannot be calculated.

        Returns the number of values that became inf / NaN from finite
        operands (see :meth:`evaluate`); nothing is raised or warned for them.
        """
        expr = parse(expression)
        created = name not in self._columns
        if not created and attribute:
            raise ValueError(f"attribute {name!r} exists; its settings cannot be given again")
        if (self._attributes[name].kind if not created else attribute.get("kind")) == "category":
            raise ValueError("a category attribute cannot be calculated from an expression")
        values, n_invalid = self.evaluate(expr, where, return_invalid=True)
        if created:
            attribute.setdefault("description", f"= {expr.source}")
            self.add_attribute(name, **attribute)
        try:
            self.set(name, values.astype(np.float64) if values.dtype == bool else values, where)
        except Exception:
            if created:
                self.delete_attribute(name)
            raise
        return n_invalid

    def dense_view(self, name: str) -> np.ndarray:
        """An attribute as an ``(nx, ny, nz)`` array ``a[i, j, k]`` sharing memory with the column.

        Only for a model without sub-blocks (:attr:`is_dense`); ``ValueError``
        otherwise. Writing to the view writes to the model.
        """
        self._require(name)
        if not self.is_dense:
            raise ValueError("dense_view needs a model without sub-blocks")
        return self._columns[name].reshape(self.definition.shape)


def _readonly(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.setflags(write=False)
    return v

"""Exact plan-view predicates shared by point-in-solid and DTM queries.

Everything that decides *which triangles a vertical line passes through* is
done here, exactly, so that a line through a shared edge or a shared vertex
is assigned to exactly one of the triangles around it.

Exact orientation
-----------------
:func:`orient2d_sign` returns the exact sign of

    (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)

for float64 inputs: a floating-point evaluation with a rigorous error bound,
and, when the bound cannot decide, an exact evaluation with floating-point
expansions (Shewchuk 1997, "Adaptive precision floating-point arithmetic and
fast robust geometric predicates"; Dekker 1971 for the exact product). The
result does not depend on how far the coordinates are from the origin.

The tie-breaking rule (symbolic perturbation)
---------------------------------------------
A query point ``p`` is treated as ``p + (e, e**2)`` with ``e`` an
infinitesimal positive number (Edelsbrunner & Muecke 1990, "Simulation of
simplicity"). The displaced point never lies on an edge or a vertex, so it is
strictly inside or strictly outside every projected triangle:

* ``p`` on an edge shared by two triangles belongs to the triangle on the
  +x side of the edge (on the +y side if the edge is parallel to x);
* ``p`` on a vertex belongs to the single triangle of the fan that contains
  the direction ``(+1, +0)``;
* a triangle that is vertical (zero area in plan) contains no point.

Because the rule is evaluated exactly, a vertical line through any point is
counted against every surface sheet exactly once.

The exact part needs products of coordinates not to overflow or underflow:
coordinate magnitudes between about 1e-140 and 1e140 are safe.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["orient2d_sign", "orient2d_sos", "tri_covers", "tri_covers_closed", "tri_z"]

_EPS = 2.0**-53
_ERRBOUND = (3.0 + 16.0 * _EPS) * _EPS
_SPLITTER = 134217729.0  # 2**27 + 1


@njit(cache=True)
def _orient2d_exact(ax, ay, bx, by, cx, cy):
    """Exact sign of ax*by - ax*cy - cx*by - ay*bx + ay*cx + cy*bx."""
    # terms: +ax*by  -cx*by  +ay*cx  +cy*bx  -ax*cy  -ay*bx
    left = (ax, -cx, ay, cy, -ax, -ay)
    right = (by, by, cx, bx, cy, bx)
    exp = np.zeros(13)
    n = 0
    for i in range(6):
        a = left[i]
        b = right[i]
        # two_product (Dekker): a * b = x + y exactly
        x = a * b
        c = _SPLITTER * a
        abig = c - a
        ahi = c - abig
        alo = a - ahi
        c = _SPLITTER * b
        abig = c - b
        bhi = c - abig
        blo = b - bhi
        err = x - ahi * bhi
        err = err - alo * bhi
        err = err - ahi * blo
        y = alo * blo - err
        for part in (y, x):
            # grow_expansion: add one float to a non-overlapping expansion
            q = part
            m = 0
            for j in range(n):
                e = exp[j]
                s = q + e
                bv = s - q
                av = s - bv
                h = (q - av) + (e - bv)
                q = s
                if h != 0.0:
                    exp[m] = h
                    m += 1
            if q != 0.0:
                exp[m] = q
                m += 1
            n = m
    if n == 0:
        return 0
    return 1 if exp[n - 1] > 0.0 else -1


@njit(cache=True, inline="always")
def orient2d_sign(ax, ay, bx, by, cx, cy):
    """Exact sign (+1, 0, -1) of the orientation of ``a, b, c`` in plan.

    +1 when ``c`` is to the left of the directed line ``a -> b``
    (counter-clockwise), -1 to the right, 0 when exactly collinear.
    """
    detleft = (ax - cx) * (by - cy)
    detright = (ay - cy) * (bx - cx)
    det = detleft - detright
    if detleft > 0.0:
        if detright <= 0.0:
            return 1 if det > 0.0 else (-1 if det < 0.0 else 0)
        detsum = detleft + detright
    elif detleft < 0.0:
        if detright >= 0.0:
            return 1 if det > 0.0 else (-1 if det < 0.0 else 0)
        detsum = -detleft - detright
    else:
        return 1 if det > 0.0 else (-1 if det < 0.0 else 0)
    errbound = _ERRBOUND * detsum
    if det >= errbound:
        return 1
    if -det >= errbound:
        return -1
    return _orient2d_exact(ax, ay, bx, by, cx, cy)


@njit(cache=True, inline="always")
def orient2d_sos(ax, ay, bx, by, px, py):
    """Sign of the orientation of ``a, b, p + (e, e**2)``; never 0 unless a == b in plan."""
    s = orient2d_sign(ax, ay, bx, by, px, py)
    if s != 0:
        return s
    # d(orient)/d(px) = -(by - ay), d(orient)/d(py) = bx - ax
    if by > ay:
        return -1
    if by < ay:
        return 1
    if bx > ax:
        return 1
    if bx < ax:
        return -1
    return 0


@njit(cache=True, inline="always")
def tri_covers(ax, ay, bx, by, cx, cy, px, py):
    """True if the displaced point ``p + (e, e**2)`` is inside triangle ``a, b, c`` in plan.

    Either winding is accepted. Exact; see the module docstring for the rule.
    """
    s1 = orient2d_sos(ax, ay, bx, by, px, py)
    if s1 == 0:
        return False
    if orient2d_sos(bx, by, cx, cy, px, py) != s1:
        return False
    return orient2d_sos(cx, cy, ax, ay, px, py) == s1


@njit(cache=True, inline="always")
def tri_covers_closed(ax, ay, bx, by, cx, cy, px, py):
    """True if ``p`` is inside or on the boundary of triangle ``a, b, c`` in plan (exact).

    False for a triangle with zero area in plan.
    """
    s1 = orient2d_sign(ax, ay, bx, by, px, py)
    s2 = orient2d_sign(bx, by, cx, cy, px, py)
    s3 = orient2d_sign(cx, cy, ax, ay, px, py)
    if s1 == 0 and s2 == 0 and s3 == 0:
        return False
    return (s1 >= 0 and s2 >= 0 and s3 >= 0) or (s1 <= 0 and s2 <= 0 and s3 <= 0)


@njit(cache=True, inline="always")
def tri_z(ax, ay, az, bx, by, bz, cx, cy, cz, px, py):
    """Z of the plane of triangle ``a, b, c`` at ``(px, py)``, for a point inside it in plan.

    Barycentric interpolation with weights computed from differences to
    ``p`` (accurate at any distance from the coordinate origin). The result
    is exactly the vertex Z when ``p`` is a vertex, exactly the common Z for
    a horizontal triangle, and always within the Z range of the triangle.
    """
    dax = ax - px
    day = ay - py
    dbx = bx - px
    dby = by - py
    dcx = cx - px
    dcy = cy - py
    wa = abs(dbx * dcy - dby * dcx)
    wb = abs(dcx * day - dcy * dax)
    wc = abs(dax * dby - day * dbx)
    s = wa + wb + wc
    if not (s > 0.0):
        z = (az + bz + cz) / 3.0
    elif wb == 0.0 and wc == 0.0:
        return az
    elif wa == 0.0 and wc == 0.0:
        return bz
    elif wa == 0.0 and wb == 0.0:
        return cz
    else:
        z = az + (wb * (bz - az) + wc * (cz - az)) / s
    zlo = min(az, bz, cz)
    zhi = max(az, bz, cz)
    if z < zlo:
        return zlo
    if z > zhi:
        return zhi
    return z

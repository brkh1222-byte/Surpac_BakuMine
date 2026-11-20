"""Numba parallel kernels called from several Python threads at once (the API's thread
pool plus an estimation thread) must not abort the process.

Numba's workqueue threading layer (the only one on macOS arm64: no TBB / OpenMP wheels)
terminates the process when two threads launch parallel kernels concurrently. The run is
in a subprocess so that an abort fails this test instead of killing pytest.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from bakumine.core.geometry.inside import points_in_solid
    from bakumine.core.geometry.primitives import icosphere
    from bakumine.core.geometry.validation import validate

    mesh = icosphere(3)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.2, 1.2, (200_000, 3))

    def work(i):
        if i % 2 == 0:
            return int(points_in_solid(mesh, pts).sum())
        return int(validate(mesh).is_closed)

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(work, range(48)))
    first = out[0::2]
    assert all(v == first[0] for v in first), first
    print("ok")
    """
)


def test_parallel_kernels_from_many_threads_do_not_abort():
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().endswith("ok")

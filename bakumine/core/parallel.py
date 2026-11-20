"""Numba ``parallel=True`` kernels that are safe to call from several Python threads.

Numba picks a threading layer on the first parallel launch: TBB, else OpenMP, else its
own *workqueue*. Where there are no TBB / OpenMP builds (macOS arm64 wheels), workqueue
is used, and it terminates the whole process when two Python threads launch parallel
kernels at the same time. The API serves requests on a thread pool and runs estimation
on a thread of its own, so that happens in normal use.

:func:`parallel_njit` compiles like ``njit(parallel=True, ...)`` and launches the kernel
under one process-wide lock. Each launch still uses every core; launches from different
threads take turns. Kernels must be called from Python (not from other jitted code); the
compiled dispatcher is the wrapper's ``kernel`` attribute.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any

from numba import njit

#: held while a parallel kernel runs
launch_lock = threading.RLock()


def parallel_njit(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        kernel = njit(parallel=True, **options)(func)

        @functools.wraps(func)
        def launch(*args: Any) -> Any:
            with launch_lock:
                return kernel(*args)

        launch.kernel = kernel  # type: ignore[attr-defined]
        return launch

    return decorate

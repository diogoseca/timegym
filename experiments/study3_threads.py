"""One numerical thread per process (protocol 3.0, environment.threads), for processes that load libraries late.

threadpoolctl limits only the thread pools of the libraries that are loaded when it is called. A process that
sets the limit first and imports NumPy, SciPy or scikit-learn afterwards (a unit of Parts 1 and 2, the parent
that computes their statistics, a zoo model that imports its library when it is built) therefore ran those
libraries on every core. start_with_one_thread makes libraries that are loaded later start with one thread, and
require_one_thread reads the loaded pools back and refuses a pool with more than one thread, so that a record is
written only by a process that kept the rule.
"""
from __future__ import annotations

import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def start_with_one_thread() -> None:
    """Make numerical libraries loaded from now on, in this process and in its children, start with one thread."""
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def wide_pools(pools: list[dict]) -> list[str]:
    """The pools of a threadpoolctl.threadpool_info() list that have more than one thread."""
    return [f"{pool.get('prefix')} ({pool.get('num_threads')} threads)" for pool in pools
            if pool.get("num_threads") != 1]


def require_one_thread() -> list[dict]:
    """The thread pools loaded in this process; raises if one of them has more than one thread."""
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    if wide_pools(pools):
        raise RuntimeError("More than one numerical thread in this process: " + ", ".join(wide_pools(pools)))
    return pools

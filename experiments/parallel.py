"""Ordered process execution; random seeds belong to tasks, not workers."""
import atexit
from concurrent.futures import ProcessPoolExecutor
import os

_POOL = None


def ordered_map(function, tasks):
    global _POOL
    workers = int(os.environ.get('METHANE_WORKERS','1'))
    if workers <= 1:
        return map(function,tasks)
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=workers)
        atexit.register(_POOL.shutdown)
    return _POOL.map(function,tasks,chunksize=10)

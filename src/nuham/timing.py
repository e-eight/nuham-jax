"""Wall-clock stage timers for the drivers; they only observe and never change results."""

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext


class StageTimer:
    """Accumulate wall-clock seconds per named stage.

    Stages may be entered many times (once per channel, say); the totals add up. Drivers take an
    optional timer and use :func:`stage`, so untimed runs pay nothing. Kernels never see it.
    """

    def __init__(self) -> None:
        """Start with no stages recorded."""
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the body of a ``with`` block under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start
            self.calls[name] = self.calls.get(name, 0) + 1

    def wrap(self, name: str, items: Iterator) -> Iterator:
        """Yield from ``items``, charging only the time spent producing each item to ``name``."""
        it = iter(items)
        while True:
            with self.stage(name):
                item = next(it, _END)
            if item is _END:
                return
            yield item


_END = object()


def stage(timer: StageTimer | None, name: str) -> AbstractContextManager[None]:
    """Return ``timer.stage(name)``, or a do-nothing context manager when ``timer`` is None."""
    return nullcontext() if timer is None else timer.stage(name)

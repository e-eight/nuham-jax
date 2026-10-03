"""Device-to-host transfer of results."""

import os

import jax
import numpy as np
from jax import Array


def pinned_host_available(device: jax.Device) -> bool:
    """Return whether ``device`` can address a ``pinned_host`` memory space."""
    try:
        return "pinned_host" in {m.kind for m in device.addressable_memories()}
    except (AttributeError, RuntimeError):
        return False


def use_pinned_by_default() -> bool:
    """Decide whether results are copied through pinned host memory.

    ``NUHAM_PINNED_COPY=0`` turns it off and ``=1`` forces it on (where available). Otherwise it is
    used on accelerator backends only: on a GH200 it copies a 152 MB result at ~146 GB/s against
    ~9 GB/s for a pageable copy, which is limited by first-touch page faults on the new buffer.
    """
    setting = os.environ.get("NUHAM_PINNED_COPY")
    if setting in ("0", "1"):
        return setting == "1"
    return jax.default_backend() != "cpu"


def to_host(x: Array, pinned: bool | None = None) -> np.ndarray:
    """Return ``x`` as a host array, going through pinned host memory when that helps.

    The result is read-only when it is a view of pinned memory; the underlying buffer stays alive
    as long as the array does.

    Parameters
    ----------
    x
        Array on a single device.
    pinned
        ``True`` or ``False`` to force the path; ``None`` chooses with
        :func:`use_pinned_by_default`. Falls back to a normal copy if the device has no pinned
        host memory.

    Returns
    -------
    np.ndarray
        The values of ``x``.
    """
    if pinned is None:
        pinned = use_pinned_by_default()
    device = next(iter(x.devices()))
    if pinned and pinned_host_available(device):
        sharding = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
        return np.asarray(jax.device_put(x, sharding))
    return np.asarray(x)

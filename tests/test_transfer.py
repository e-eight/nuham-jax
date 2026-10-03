import jax.numpy as jnp
import numpy as np
import pytest

import nuham  # noqa: F401
from nuham.kernels import transfer
from nuham.kernels.transfer import pinned_host_available, to_host, use_pinned_by_default


def make() -> jnp.ndarray:
    return jnp.arange(24.0).reshape(4, 6) * 3.0 - 7.0


@pytest.mark.parametrize("pinned", [True, False, None])
def test_values_are_identical_on_every_path(pinned: bool | None) -> None:
    np.testing.assert_array_equal(to_host(make(), pinned=pinned), np.asarray(make()))


def test_pinned_view_stays_valid_after_the_device_array_is_gone() -> None:
    import gc

    host = to_host(make(), pinned=True)
    expected = np.asarray(make())
    gc.collect()
    np.testing.assert_array_equal(host, expected)


def test_falls_back_when_pinned_memory_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer, "pinned_host_available", lambda device: False)
    np.testing.assert_array_equal(to_host(make(), pinned=True), np.asarray(make()))


def test_default_follows_the_backend_and_the_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax

    monkeypatch.delenv("NUHAM_PINNED_COPY", raising=False)
    assert use_pinned_by_default() is (jax.default_backend() != "cpu")
    monkeypatch.setenv("NUHAM_PINNED_COPY", "1")
    assert use_pinned_by_default() is True
    monkeypatch.setenv("NUHAM_PINNED_COPY", "0")
    assert use_pinned_by_default() is False


def test_cpu_backend_reports_pinned_memory_support() -> None:
    import jax

    assert isinstance(pinned_host_available(jax.devices()[0]), bool)

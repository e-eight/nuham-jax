import jax.numpy as jnp
import numpy as np
import pytest

import nuham  # noqa: F401
from nuham.kernels.srg import _commutator_rhs, srg_alpha, srg_evolve_batch, srg_flow


def random_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = 8.0 * (2.0 * np.arange(n) + 1.5)
    e = 8.0 * np.sqrt(np.arange(1, n) * (np.arange(1, n) + 1.5))
    t = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    v = rng.normal(size=(n, n))
    v = -20.0 * (v + v.T) / n
    return t + v, t, d, e


def test_alpha_matches_nuhamil() -> None:
    assert srg_alpha(2.0) == pytest.approx(3.634037e-5, rel=1e-6)


def test_banded_rhs_matches_dense_commutator() -> None:
    h, t, d, e = random_problem(12, 0)
    zero = np.zeros(1)
    got = _commutator_rhs(
        jnp.asarray(h), jnp.asarray(d), jnp.asarray(np.r_[e, zero]), jnp.asarray(np.r_[zero, e])
    )
    eta = t @ h - h @ t
    np.testing.assert_allclose(got, eta @ h - h @ eta, rtol=1e-12, atol=1e-8)


def test_flow_preserves_spectrum() -> None:
    h, t, d, e = random_problem(20, 1)
    out = np.asarray(
        srg_flow(jnp.asarray(h / 100), jnp.asarray(d / 100), jnp.asarray(e / 100), 0.2)
    )
    np.testing.assert_allclose(np.linalg.eigvalsh(out), np.linalg.eigvalsh(h / 100), atol=1e-8)
    np.testing.assert_allclose(out, out.T, atol=1e-12)


def test_flow_decouples_toward_diagonal() -> None:
    h, _, d, e = random_problem(20, 2)
    small = srg_flow(jnp.asarray(h / 100), jnp.asarray(d / 100), jnp.asarray(e / 100), 0.01)
    large = srg_flow(jnp.asarray(h / 100), jnp.asarray(d / 100), jnp.asarray(e / 100), 1.0)
    far = np.abs(np.arange(20)[:, None] - np.arange(20)[None, :]) > 3
    assert np.abs(np.asarray(large))[far].max() < np.abs(np.asarray(small))[far].max()


def test_padding_does_not_change_the_result() -> None:
    h, _, d, e = random_problem(15, 3)
    pad = 24
    hp = np.zeros((pad, pad))
    hp[:15, :15] = h
    dp = np.zeros(pad)
    dp[:15] = d
    ep = np.zeros(pad - 1)
    ep[:14] = e
    args = (100.0, 5e-5)
    a = srg_evolve_batch(jnp.asarray(h[None]), jnp.asarray(d[None]), jnp.asarray(e[None]), *args)
    b = srg_evolve_batch(jnp.asarray(hp[None]), jnp.asarray(dp[None]), jnp.asarray(ep[None]), *args)
    np.testing.assert_allclose(np.asarray(b)[0, :15, :15], np.asarray(a)[0], atol=1e-8)
    assert np.all(np.asarray(b)[0, 15:] == 0.0)

"""Coulomb interaction between protons in the HO relative basis."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import block_diag

from nuham.constants import ALPHA_EM, HBARC, M_PROTON
from nuham.kernels.ho import radial_functions

_R_MAX = 40.0
_N_QUAD = 500


def _radial_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre mesh on ``[0, 40]`` fm with 500 points, as NuHamil uses."""
    x, w = np.polynomial.legendre.leggauss(_N_QUAD)
    half = _R_MAX / 2.0
    return half * (x + 1.0), half * w


@partial(jax.jit, static_argnames=("ls", "n2max"))
def coulomb_matrix(ls: tuple[int, ...], n2max: int, hw: float) -> Array:
    """Point-Coulomb ``(hbar c / alpha) / r`` in the relative HO basis of one pp channel.

    The oscillator parameter is ``nu^2 = m_p hw / (2 (hbar c)^2)`` in fm^-2, which differs from
    the one used for the momentum-space projection. The matrix is block-diagonal in ``l``.

    Parameters
    ----------
    ls
        Orbital angular momenta of the channel (static).
    n2max
        Truncation on ``2n + l`` (static).
    hw
        Oscillator energy in MeV.

    Returns
    -------
    Array
        Matrix in MeV, with the same l-major state ordering as the projected potential.
    """
    r_np, w_np = _radial_mesh()
    r = jnp.asarray(r_np)
    w = jnp.asarray(w_np)
    nu = jnp.sqrt(0.5 * M_PROTON * hw) / HBARC
    blocks = []
    for l in ls:
        rad = jnp.sqrt(nu) * radial_functions(nu * r, l, (n2max - l) // 2)
        blocks.append((rad * (w * HBARC / ALPHA_EM / r)[None, :]) @ rad.T)
    return block_diag(*blocks)

"""JAX generator of nucleon-nucleon matrix elements, validated against NuHamil."""

import os

# JAX reserves ~75% of the GPU's memory at start-up by default, which starves other users of a
# shared GPU (measured on a shared H200: a tiny process raised used memory from 12 GB to 138 GB).
# Allocate on demand instead; an explicit setting by the user still wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

"""JAX generator of nucleon-nucleon matrix elements, validated against NuHamil."""

import jax

jax.config.update("jax_enable_x64", True)

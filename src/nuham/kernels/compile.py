"""Compiler options shared by kernels whose products are large and compiled for many shapes."""

NO_AUTOTUNE = {"xla_gpu_autotune_level": 0}
"""Compile a kernel without GEMM autotuning (a GPU option; other backends ignore it).

XLA tunes every distinct matrix product by running a few dozen algorithms on it. For the products
of the 3N chain (up to 3e13 flops each, one new shape per channel) that took about 20 s for the
antisymmetrisation of one channel and 32 s for the lab contraction of the largest one, in every
process and also with a warm compile cache; without it the first call is 4 s and 14 s, for
products that run 1.0x to 1.3x slower (DeltaAI job 3317317, ``docs/study-me3j.md``). It is set per
kernel and not as the global ``XLA_FLAGS`` because the SRG flow gains 2x from its tuned products.
"""

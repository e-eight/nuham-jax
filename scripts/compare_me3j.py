"""Element-wise comparison of two ``me3j.gz`` files, streamed in lockstep so memory stays small.

usage: python scripts/compare_me3j.py A.me3j.gz B.me3j.gz [--tol 1e-5]

Prints the number of values, the largest value, the largest absolute difference with its index and
the number of values above ``tol`` (and above 1e-3). The exit status is 1 if any value is above
``tol`` or the lengths differ. Used to compare a GPU-made file with a CPU-made or NuHamil file
(``docs/imsrg-check.md``).
"""

import argparse
import gzip
import sys
from collections.abc import Iterator

import numpy as np

BLOCK = 5_000_000


def blocks(path: str, size: int = BLOCK) -> Iterator[np.ndarray]:
    """Yield the values of a me3j file (after its header line) in arrays of ``size`` values."""
    buffer = np.empty(0)
    carry = ""
    with gzip.open(path, "rt") as handle:
        handle.readline()
        while True:
            chunk = handle.read(1 << 26)
            if not chunk:
                break
            chunk = carry + chunk
            cut = chunk.rfind(" ")
            if cut < 0:
                carry = chunk
                continue
            carry, chunk = chunk[cut:], chunk[:cut]
            buffer = np.concatenate([buffer, np.array(chunk.split(), dtype=float)])
            while len(buffer) >= size:
                yield buffer[:size]
                buffer = buffer[size:]
    if carry.strip():
        buffer = np.concatenate([buffer, np.array(carry.split(), dtype=float)])
    while len(buffer) >= size:
        yield buffer[:size]
        buffer = buffer[size:]
    if len(buffer):
        yield buffer


def compare(path_a: str, path_b: str, tol: float) -> int:
    """Compare two files, print the summary and return the exit status (0 if all within ``tol``)."""
    count, vmax, worst, above, above_1e3 = 0, 0.0, (0.0, -1, 0.0, 0.0), 0, 0
    iter_a, iter_b = blocks(path_a), blocks(path_b)
    for x, y in zip(iter_a, iter_b, strict=False):
        if len(x) != len(y):
            print(
                f"length mismatch in the block starting at value {count}: {len(x)} against {len(y)}"
            )
            return 1
        diff = np.abs(x - y)
        k = int(diff.argmax())
        if diff[k] > worst[0]:
            worst = (float(diff[k]), count + k, float(x[k]), float(y[k]))
        above += int((diff > tol).sum())
        above_1e3 += int((diff > 1e-3).sum())
        vmax = max(vmax, float(np.abs(x).max()))
        count += len(x)
    leftover = sum(len(z) for z in iter_a) + sum(len(z) for z in iter_b)
    print(f"compared {count} values; max |value| {vmax:.3f}; leftover values {leftover}")
    print(f"max |diff| {worst[0]:.3e} at index {worst[1]}: {worst[2]:.8f} against {worst[3]:.8f}")
    print(f"values with |diff| > {tol:g}: {above}; > 1e-3: {above_1e3}")
    return 0 if above == 0 and leftover == 0 else 1


def main() -> None:
    """Parse the arguments and compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()
    sys.exit(compare(args.a, args.b, args.tol))


if __name__ == "__main__":
    main()

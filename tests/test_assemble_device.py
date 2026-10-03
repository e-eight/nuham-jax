"""The JAX assembly of me2j values must equal the numpy one; no NuHamil data needed for most."""

import numpy as np
import pytest
from synthetic import synthetic_nn

import nuham  # noqa: F401
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits
from nuham.lab_transform import lab_channel_matrices
from nuham.me2j import LabChannel, assemble, assemble_device
from nuham.relative_nn import evolve_relative_nn
from nuham.runtime import WriterOptions

EMAX, E2MAX = 3, 6


@pytest.fixture(scope="module")
def setup():
    relative = {
        (c.j, c.parity, c.s, c.tz): c
        for c in evolve_relative_nn(synthetic_nn(), 16.0, 2.0, n2max=24)
    }
    orb = make_orbits(EMAX)
    mosh = MoshinskyTable()
    channels = {
        jp: LabChannel(lab_channel_matrices(orb, E2MAX, *jp, relative, mosh), len(orb))
        for jp in lab_channels(orb, E2MAX)
    }
    return orb, channels


@pytest.mark.parametrize("batch_tuples", [37, 256, 1 << 20])
def test_device_assembly_matches_numpy(setup, batch_tuples: int) -> None:
    orb, channels = setup
    want = np.concatenate(list(assemble(orb, E2MAX, channels)))
    got = np.concatenate(list(assemble_device(orb, E2MAX, channels, batch_tuples)))
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)


def test_windows_have_the_requested_size(setup) -> None:
    orb, channels = setup
    sizes = [len(c) for c in assemble_device(orb, E2MAX, channels, batch_tuples=100)]
    assert all(s == 100 for s in sizes[:-1]) and 0 < sizes[-1] <= 100


def test_generate_with_device_assembler_writes_the_same_file(tmp_path) -> None:
    from nuham.generate import generate_me2j
    from nuham.me2j import read_me2j

    nn, mosh = synthetic_nn(), MoshinskyTable()
    generate_me2j(nn, tmp_path / "a.me2j", 16.0, 2.0, EMAX, E2MAX, mosh=mosh, n2max=24)
    generate_me2j(
        nn,
        tmp_path / "b.me2j",
        16.0,
        2.0,
        EMAX,
        E2MAX,
        mosh=mosh,
        n2max=24,
        options=WriterOptions(assembler="device"),
    )
    np.testing.assert_array_equal(read_me2j(tmp_path / "a.me2j"), read_me2j(tmp_path / "b.me2j"))


def test_scan_with_device_assembler_matches(tmp_path) -> None:
    from nuham.me2j import read_me2j
    from nuham.scan import ScanPoint, generate_scan

    nn, mosh = synthetic_nn(), MoshinskyTable()
    pts = [ScanPoint(16.0, 2.0, tmp_path / "n.me2j"), ScanPoint(20.0, 1.8, tmp_path / "m.me2j")]
    generate_scan(
        nn, pts, EMAX, E2MAX, mosh=mosh, n2max=24, options=WriterOptions(assembler="numpy")
    )
    pts2 = [ScanPoint(16.0, 2.0, tmp_path / "dn.me2j"), ScanPoint(20.0, 1.8, tmp_path / "dm.me2j")]
    generate_scan(
        nn, pts2, EMAX, E2MAX, mosh=mosh, n2max=24, options=WriterOptions(assembler="device")
    )
    for a, b in zip(("n", "m"), ("dn", "dm"), strict=True):
        np.testing.assert_array_equal(
            read_me2j(tmp_path / f"{a}.me2j"), read_me2j(tmp_path / f"{b}.me2j")
        )


def test_device_resident_matrices_give_the_same_values(setup) -> None:
    import jax

    orb, channels = setup
    relative = {
        (c.j, c.parity, c.s, c.tz): c
        for c in evolve_relative_nn(synthetic_nn(), 16.0, 2.0, n2max=24)
    }
    mosh = MoshinskyTable()
    on_device = {
        jp: LabChannel(lab_channel_matrices(orb, E2MAX, *jp, relative, mosh, None, True), len(orb))
        for jp in lab_channels(orb, E2MAX)
    }
    some = next(m for ch in on_device.values() for m in ch.matrices.values() if m.v.size)
    assert isinstance(some.v, jax.Array)
    want = np.concatenate(list(assemble(orb, E2MAX, channels)))
    got = np.concatenate(list(assemble_device(orb, E2MAX, on_device, 4096)))
    np.testing.assert_array_equal(got, want)

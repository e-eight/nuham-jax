"""Physical constants, fixed to the values NuHamil uses so results can match it."""

HBARC: float = 197.32705
"""hbar*c in MeV fm."""

M_PROTON: float = 938.27231
"""Proton mass in MeV."""

M_NEUTRON: float = 939.56563
"""Neutron mass in MeV."""

M_REDUCED_PN: float = M_PROTON * M_NEUTRON / (M_PROTON + M_NEUTRON)
"""Reduced mass of the proton-neutron pair in MeV, used for pp, nn and pn alike."""

ALPHA_EM: float = 137.035999
"""Inverse fine-structure constant."""


def oscillator_length_sq(hw: float, mass: float = M_REDUCED_PN) -> float:
    """Return b^2 = (hbar c)^2 / (mass * hw) in fm^2 for an oscillator of energy ``hw``."""
    return HBARC**2 / (mass * hw)

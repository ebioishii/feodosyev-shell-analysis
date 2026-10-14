"""Small-strain shell kinematics; no equilibrium or material-history solution."""

from __future__ import annotations

import math


def mechanical_hoop_strain(
    radial_displacement_m: float, reference_radius_m: float, free_thermal_strain: float
) -> float:
    """Return delta_R/R - free thermal strain for a thin circular shell.

    The radius and displacement use m and refer to the same reference state.
    This is a kinematic relation, not a Feodosyev shell solver.
    """
    try:
        displacement = float(radial_displacement_m)
        radius = float(reference_radius_m)
        thermal = float(free_thermal_strain)
    except (TypeError, ValueError) as exc:
        raise ValueError("Shell kinematics inputs must be finite numbers") from exc
    if not all(math.isfinite(value) for value in (displacement, radius, thermal)):
        raise ValueError("Shell kinematics inputs must be finite numbers")
    if radius <= 0:
        raise ValueError("reference_radius_m must be positive")
    return displacement / radius - thermal

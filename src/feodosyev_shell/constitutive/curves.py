"""Monotonic, uniaxial stress-strain curves only.

No function here calculates unloading, residual strain or load history.
"""

from __future__ import annotations

import numpy as np

from feodosyev_shell.materials.dataset import CURVE_KINDS, Material, _finite_number


def _bracket(material: Material, temperature_k: float) -> tuple[float, float]:
    if not material.curves:
        raise ValueError("No stress-strain curves available")
    temperature = _finite_number(temperature_k, "temperature_k")
    knots = sorted(material.curves)
    if not knots[0] <= temperature <= knots[-1]:
        raise ValueError("Temperature outside available curve range")
    high = int(np.searchsorted(knots, temperature))
    low = high if knots[high] == temperature else high - 1
    return knots[low], knots[high]


def curve_strain_intervals(material: Material, temperature_k: float) -> list[tuple[float, float]]:
    """Supported strain spans at T: intersection of the two bracketing curves."""
    low, high = _bracket(material, temperature_k)
    return [(max(a, c), min(b, d))
            for a, b in material.strain_intervals(low)
            for c, d in material.strain_intervals(high)
            if max(a, c) <= min(b, d)]


def _allow_curve_use(material: Material, allow_provisional: bool) -> None:
    if material.is_provisional and not allow_provisional:
        raise ValueError("Provisional dataset: pass allow_provisional=True for exploratory use")


def evaluate_curve(
    material: Material, temperature_k: float, strain: float | np.ndarray, *, kind: str,
    allow_provisional: bool = False,
) -> float | np.ndarray:
    """Linear interpolation in strain and then between the bracketing temperatures.

    Curves must share the same stress/strain definition and material condition.
    The valid strain interval between temperatures is their intersection.
    """
    if kind not in CURVE_KINDS or material.curve_kind != kind:
        raise ValueError("Requested curve definition does not match the dataset")
    _allow_curve_use(material, allow_provisional)
    low, high = _bracket(material, temperature_k)
    temperature = _finite_number(temperature_k, "temperature_k")
    values = np.asarray(strain, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Strain must be finite")
    x0, y0 = material.curves[low]
    x1, y1 = material.curves[high]
    minimum, maximum = max(x0[0], x1[0]), min(x0[-1], x1[-1])
    if minimum > maximum or np.any(values < minimum) or np.any(values > maximum):
        raise ValueError(f"Strain outside common range [{minimum}, {maximum}]")
    supported = np.zeros(values.shape, dtype=bool)
    for left, right in curve_strain_intervals(material, temperature):
        supported |= (values >= left) & (values <= right)
    if not supported.all():
        raise ValueError("Strain falls in an excluded data gap; interpolation is unavailable")
    stress0 = np.interp(values, x0, y0)
    if low == high:
        result = stress0
    else:
        stress1 = np.interp(values, x1, y1)
        fraction = (temperature - low) / (high - low)
        result = stress0 + fraction * (stress1 - stress0)
    return float(result) if values.ndim == 0 else result


def convert_curve(
    strain: np.ndarray | list[float],
    stress_pa: np.ndarray | list[float],
    *,
    from_kind: str,
    to_kind: str,
    youngs_modulus_pa: float | None = None,
    uniform_deformation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tensile engineering data to true total or true plastic data.

    This conventional conversion requires uniform deformation (before necking).
    It is not a general reversible mapping or a cyclic plasticity model.
    """
    if from_kind != "engineering" or to_kind not in {"true_total", "true_plastic"}:
        raise ValueError("Only engineering to true total/plastic conversion is supported")
    if not uniform_deformation:
        raise ValueError("Uniform deformation must be confirmed before conversion")
    eps = np.asarray(strain, dtype=float)
    sigma = np.asarray(stress_pa, dtype=float)
    if eps.ndim != 1 or sigma.shape != eps.shape or len(eps) < 2:
        raise ValueError("Strain and stress must be equal-length 1D arrays")
    if (not np.isfinite(eps).all() or not np.isfinite(sigma).all()
            or np.any(eps < 0) or np.any(sigma < 0) or np.any(np.diff(eps) <= 0)):
        raise ValueError("Input must be finite, nonnegative tensile data with increasing strain")
    true_strain = np.log1p(eps)
    true_stress = sigma * (1 + eps)
    if to_kind == "true_total":
        return true_strain, true_stress
    modulus = _finite_number(youngs_modulus_pa, "youngs_modulus_pa")
    if modulus <= 0:
        raise ValueError("Young's modulus must be positive")
    plastic_strain = true_strain - true_stress / modulus
    if np.any(plastic_strain < -1e-12):
        raise ValueError("Negative plastic strain: data, modulus or assumptions inconsistent")
    plastic_strain = np.maximum(plastic_strain, 0.0)
    if np.any(np.diff(plastic_strain) <= 0):
        raise ValueError("Plastic strain is not strictly increasing; trim elastic points")
    return plastic_strain, true_stress

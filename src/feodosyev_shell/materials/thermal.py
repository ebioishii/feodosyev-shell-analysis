"""Free thermal strain from a mean (secant) expansion coefficient table.

All temperatures are K. The coefficient's source reference temperature and
the stress-free temperature of a structural case are separate quantities.
"""

from __future__ import annotations

import math

import numpy as np

from feodosyev_shell.materials.dataset import Material


def _temperature(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite temperature in K") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite temperature in K")
    return result


def _mean_expansion_table(material: Material) -> tuple[np.ndarray, np.ndarray, float]:
    if "thermal_expansion" not in material.properties:
        raise ValueError("Material property unavailable: thermal_expansion")
    entry = next(
        item for item in material.metadata["properties"] if item["name"] == "thermal_expansion"
    )
    if entry["coefficient_type"] != "mean":
        raise ValueError("Accumulated thermal strain requires a mean expansion coefficient")
    temperature_k, mean_alpha_per_k = material.properties["thermal_expansion"]
    return temperature_k, mean_alpha_per_k, float(entry["reference_temperature_k"])


def accumulated_thermal_strain(material: Material, temperature_k: float) -> float:
    """Return beta(T) = mean_alpha(T0 -> T) * (T - T0), dimensionless.

    Interpolation is applied to beta at the measured temperature nodes, rather
    than to alpha before multiplication. The source reference T0 is an exact
    zero; a gap between T0 and the first measured node is not interpolated.
    """
    temperature = _temperature(temperature_k, "temperature_k")
    nodes_k, mean_alpha_per_k, source_reference_k = _mean_expansion_table(material)
    if temperature == source_reference_k:
        return 0.0
    if not nodes_k[0] <= temperature <= nodes_k[-1]:
        raise ValueError(
            "Accumulated thermal strain: temperature outside measured "
            f"[{nodes_k[0]}, {nodes_k[-1]}] K (except source reference "
            f"{source_reference_k} K)"
        )
    beta_nodes = mean_alpha_per_k * (nodes_k - source_reference_k)
    if nodes_k[0] < source_reference_k < nodes_k[-1] and source_reference_k not in nodes_k:
        insertion = int(np.searchsorted(nodes_k, source_reference_k))
        nodes_k = np.insert(nodes_k, insertion, source_reference_k)
        beta_nodes = np.insert(beta_nodes, insertion, 0.0)
    return float(np.interp(temperature, nodes_k, beta_nodes))


def thermal_strain(
    material: Material, temperature_k: float, stress_free_temperature_k: float
) -> float:
    """Return free strain relative to a stress-free reference: beta(T)-beta(Tref)."""
    current = accumulated_thermal_strain(material, temperature_k)
    reference = accumulated_thermal_strain(material, stress_free_temperature_k)
    return current - reference

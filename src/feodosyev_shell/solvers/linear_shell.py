"""Small-strain, hoop-only, two-layer cylindrical membrane model.

This is a linear-elastic reduction of Feodosyev's circumferential equilibrium,
not his complete axial/elastic-plastic load-capacity method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from feodosyev_shell.materials import Material, thermal_strain


def _positive(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


@dataclass(frozen=True)
class LinearShellCase:
    radius_m: float
    inner_thickness_m: float
    outer_thickness_m: float
    inner_temperature_k: float
    outer_temperature_k: float
    stress_free_temperature_k: float
    gas_pressure_pa: float

    def __post_init__(self) -> None:
        for name in (
            "radius_m", "inner_thickness_m", "outer_thickness_m",
            "inner_temperature_k", "outer_temperature_k", "stress_free_temperature_k",
        ):
            _positive(getattr(self, name), name)
        try:
            pressure = float(self.gas_pressure_pa)
        except (TypeError, ValueError) as exc:
            raise ValueError("gas_pressure_pa must be finite and nonnegative") from exc
        if not math.isfinite(pressure) or pressure < 0:
            raise ValueError("gas_pressure_pa must be finite and nonnegative")
        if (self.inner_thickness_m + self.outer_thickness_m) / self.radius_m >= 0.1:
            raise ValueError("Thin-shell assumption requires (t_i+t_o)/R < 0.1")


@dataclass(frozen=True)
class ShellResponse:
    total_hoop_strain: float
    thermal_hoop_strain: float
    mechanical_hoop_strain: float
    hoop_stress_pa: float
    youngs_modulus_pa: float
    elastic_limit_pa: float | None
    elastic_status: str


@dataclass(frozen=True)
class LinearShellResult:
    radial_displacement_m: float
    inner: ShellResponse
    outer: ShellResponse
    elastic_prediction_credible: bool | None
    pressure_balance_residual_n_per_m: float


def solve_linear_shell(
    case: LinearShellCase, inner_material: Material, outer_material: Material | None = None
) -> LinearShellResult:
    """Solve common hoop strain from membrane equilibrium; no extrapolation.

    A tabulated yield strength, when present, is used only as an indicative
    elastic-applicability threshold, not as a failure criterion.
    """
    outer_material = inner_material if outer_material is None else outer_material
    materials = (inner_material, outer_material)
    temperatures = (case.inner_temperature_k, case.outer_temperature_k)
    thicknesses = (case.inner_thickness_m, case.outer_thickness_m)
    moduli = tuple(
        m.property_at("youngs_modulus", t)
        for m, t in zip(materials, temperatures, strict=True)
    )
    thermal = tuple(
        thermal_strain(m, t, case.stress_free_temperature_k)
        for m, t in zip(materials, temperatures, strict=True)
    )
    stiffnesses = tuple(e * h for e, h in zip(moduli, thicknesses, strict=True))
    total = (case.gas_pressure_pa * case.radius_m + sum(
        k * strain for k, strain in zip(stiffnesses, thermal, strict=True)
    )) / sum(stiffnesses)

    responses = []
    for material, temperature, modulus, free_strain in zip(
        materials, temperatures, moduli, thermal, strict=True
    ):
        mechanical = total - free_strain
        stress = modulus * mechanical
        if "yield_strength" in material.properties:
            limit = material.property_at("yield_strength", temperature)
            status = (
                "outside_selected_elastic_limit"
                if abs(stress) > limit else "within_selected_elastic_limit"
            )
        else:
            limit = None
            status = "elastic_limit_unknown"
        responses.append(
            ShellResponse(total, free_strain, mechanical, stress, modulus, limit, status)
        )
    inner, outer = responses
    if any(response.elastic_status == "outside_selected_elastic_limit" for response in responses):
        credible = False
    elif any(response.elastic_status == "elastic_limit_unknown" for response in responses):
        credible = None
    else:
        credible = True
    residual = (case.inner_thickness_m * inner.hoop_stress_pa
                + case.outer_thickness_m * outer.hoop_stress_pa
                - case.gas_pressure_pa * case.radius_m)
    return LinearShellResult(case.radius_m * total, inner, outer, credible, residual)

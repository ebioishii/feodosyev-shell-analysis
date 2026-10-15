"""Two-shell, plane-stress linear elasticity with axial and hoop equilibrium.

This adds axial strain compatibility and a prescribed axial force resultant
to the Phase 3A membrane model. It is not an elastic-plastic capacity model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from feodosyev_shell.materials import Material, thermal_strain
from feodosyev_shell.solvers.linear_shell import LinearShellCase


@dataclass(frozen=True)
class BiaxialLinearCase(LinearShellCase):
    # Sum(t_j sigma_x,j) = axial_force_ratio * p_g * R.
    axial_force_ratio: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.axial_force_ratio):
            raise ValueError("axial_force_ratio must be finite")


@dataclass(frozen=True)
class BiaxialShellResponse:
    total_axial_strain: float
    total_hoop_strain: float
    thermal_strain: float
    mechanical_axial_strain: float
    mechanical_hoop_strain: float
    axial_stress_pa: float
    hoop_stress_pa: float
    von_mises_stress_pa: float
    youngs_modulus_pa: float
    poisson_ratio: float
    elastic_limit_pa: float | None
    elastic_status: str


@dataclass(frozen=True)
class BiaxialLinearResult:
    radial_displacement_m: float
    common_axial_strain: float
    inner: BiaxialShellResponse
    outer: BiaxialShellResponse
    hoop_balance_residual_n_per_m: float
    axial_balance_residual_n_per_m: float
    elastic_prediction_credible: bool | None


def solve_biaxial_linear_shell(
    case: BiaxialLinearCase, inner_material: Material, outer_material: Material | None = None
) -> BiaxialLinearResult:
    """Solve the two common strains analytically; positive stresses are tensile.

    The elastic check uses plane-stress von Mises magnitude and an optional
    tabulated yield strength as an indicative limit only.
    """
    outer_material = inner_material if outer_material is None else outer_material
    materials = (inner_material, outer_material)
    temperatures = (case.inner_temperature_k, case.outer_temperature_k)
    thicknesses = (case.inner_thickness_m, case.outer_thickness_m)
    properties = [
        (material.property_at("youngs_modulus", temperature),
         material.property_at("poisson_ratio", temperature),
         thermal_strain(material, temperature, case.stress_free_temperature_k))
        for material, temperature in zip(materials, temperatures, strict=True)
    ]
    # [A B; B A] [eps_x, eps_theta] = [k*p*R + C, p*R + C].
    a = sum(h * e / (1 - nu**2) for h, (e, nu, _) in zip(
        thicknesses, properties, strict=True
    ))
    b = sum(h * e * nu / (1 - nu**2) for h, (e, nu, _) in zip(
        thicknesses, properties, strict=True
    ))
    thermal_force = sum(h * e * free / (1 - nu) for h, (e, nu, free) in zip(
        thicknesses, properties, strict=True
    ))
    axial_force = case.axial_force_ratio * case.gas_pressure_pa * case.radius_m
    hoop_force = case.gas_pressure_pa * case.radius_m
    determinant = a * a - b * b
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError("Invalid biaxial stiffness matrix")
    total_axial = (a * (axial_force + thermal_force)
                   - b * (hoop_force + thermal_force)) / determinant
    total_hoop = (a * (hoop_force + thermal_force)
                  - b * (axial_force + thermal_force)) / determinant
    responses = []
    for material, temperature, (e, nu, free) in zip(
        materials, temperatures, properties, strict=True
    ):
        mechanical_axial = total_axial - free
        mechanical_hoop = total_hoop - free
        factor = e / (1 - nu**2)
        sigma_axial = factor * (mechanical_axial + nu * mechanical_hoop)
        sigma_hoop = factor * (mechanical_hoop + nu * mechanical_axial)
        mises = math.sqrt(max(0.0, sigma_axial**2 - sigma_axial * sigma_hoop
                              + sigma_hoop**2))
        if "yield_strength" in material.properties:
            limit = material.property_at("yield_strength", temperature)
            status = ("outside_selected_elastic_limit" if mises > limit
                      else "within_selected_elastic_limit")
        else:
            limit = None
            status = "elastic_limit_unknown"
        responses.append(BiaxialShellResponse(
            total_axial, total_hoop, free, mechanical_axial, mechanical_hoop,
            sigma_axial, sigma_hoop, mises, e, nu, limit, status,
        ))
    inner, outer = responses
    axial_residual = sum(h * response.axial_stress_pa for h, response in zip(
        thicknesses, responses, strict=True
    )) - axial_force
    hoop_residual = sum(h * response.hoop_stress_pa for h, response in zip(
        thicknesses, responses, strict=True
    )) - hoop_force
    if any(response.elastic_status == "outside_selected_elastic_limit" for response in responses):
        credible = False
    elif any(response.elastic_status == "elastic_limit_unknown" for response in responses):
        credible = None
    else:
        credible = True
    return BiaxialLinearResult(
        case.radius_m * total_hoop, total_axial, inner, outer,
        hoop_residual, axial_residual, credible,
    )

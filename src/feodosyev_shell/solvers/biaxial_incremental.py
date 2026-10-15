"""Incremental two-shell equilibrium using synthetic plane-stress J2 material.

Research scaffold only: use the synthetic fixture, never provisional SS316
engineering curves. Each accepted step supplies the next step's plastic state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from feodosyev_shell.constitutive.bilinear_j2 import J2Response, J2State, SyntheticBilinearJ2
from feodosyev_shell.materials import Material, thermal_strain
from feodosyev_shell.solvers.biaxial_linear import BiaxialLinearCase, solve_biaxial_linear_shell


@dataclass(frozen=True)
class SyntheticShellLayer:
    material: Material
    law: SyntheticBilinearJ2

    def validate(self, temperature_k: float) -> None:
        if self.material.metadata.get("source_status") != "synthetic":
            raise ValueError("Incremental solver accepts synthetic material datasets only")
        e = self.material.property_at("youngs_modulus", temperature_k)
        nu = self.material.property_at("poisson_ratio", temperature_k)
        if not math.isclose(e, self.law.youngs_modulus_pa, rel_tol=1e-10):
            raise ValueError("Synthetic law E does not match material table")
        if not math.isclose(nu, self.law.poisson_ratio, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Synthetic law Poisson ratio does not match material table")
        if "yield_strength" in self.material.properties:
            limit = self.material.property_at("yield_strength", temperature_k)
            if not math.isclose(limit, self.law.initial_yield_pa, rel_tol=1e-10):
                raise ValueError("Synthetic law initial yield does not match material table")


@dataclass(frozen=True)
class IncrementalShellState:
    common_axial_strain: float = 0.0
    common_hoop_strain: float = 0.0
    inner: J2State = field(default_factory=J2State)
    outer: J2State = field(default_factory=J2State)
    geometry_m: tuple[float, float, float] | None = None
    reference_temperature_k: float | None = None
    law_signature: tuple[tuple[float, float, float, float], ...] | None = None


@dataclass(frozen=True)
class IncrementalShellResult:
    radial_displacement_m: float
    common_axial_strain: float
    inner: J2Response
    outer: J2Response
    state: IncrementalShellState
    hoop_balance_residual_n_per_m: float
    axial_balance_residual_n_per_m: float
    global_evaluations: int


def solve_incremental_shell(
    case: BiaxialLinearCase,
    inner_layer: SyntheticShellLayer,
    previous_state: IncrementalShellState,
    outer_layer: SyntheticShellLayer | None = None,
) -> IncrementalShellResult:
    """Solve one accepted load step; previous_state is never changed in place."""
    outer_layer = inner_layer if outer_layer is None else outer_layer
    layers = (inner_layer, outer_layer)
    geometry = (case.radius_m, case.inner_thickness_m, case.outer_thickness_m)
    signature = tuple((layer.law.youngs_modulus_pa, layer.law.poisson_ratio,
                       layer.law.initial_yield_pa, layer.law.plastic_tangent_pa)
                      for layer in layers)
    if previous_state.geometry_m is not None and previous_state.geometry_m != geometry:
        raise ValueError("Incremental load path cannot change shell geometry")
    if (previous_state.reference_temperature_k is not None
            and previous_state.reference_temperature_k != case.stress_free_temperature_k):
        raise ValueError("Incremental load path cannot change stress-free reference temperature")
    if previous_state.law_signature is not None and previous_state.law_signature != signature:
        raise ValueError("Incremental load path cannot change material law")
    if not all(math.isfinite(value) for value in (
        previous_state.common_axial_strain, previous_state.common_hoop_strain
    )):
        raise ValueError("Previous common strains must be finite")
    temperatures = (case.inner_temperature_k, case.outer_temperature_k)
    thicknesses = (case.inner_thickness_m, case.outer_thickness_m)
    old_states = (previous_state.inner, previous_state.outer)
    for layer, temperature in zip(layers, temperatures, strict=True):
        layer.validate(temperature)
    free = tuple(
        thermal_strain(layer.material, temperature, case.stress_free_temperature_k)
        for layer, temperature in zip(layers, temperatures, strict=True)
    )
    target_hoop = case.gas_pressure_pa * case.radius_m
    target_axial = case.axial_force_ratio * target_hoop
    scale = sum(h * layer.law.youngs_modulus_pa for h, layer in zip(
        thicknesses, layers, strict=True
    ))

    def evaluate(strains: np.ndarray) -> tuple[tuple[J2Response, J2Response], float, float]:
        responses = tuple(
            layer.law.update(
                mechanical_axial_strain=float(strains[0] - thermal),
                mechanical_hoop_strain=float(strains[1] - thermal),
                previous_state=old,
            )
            for layer, thermal, old in zip(layers, free, old_states, strict=True)
        )
        axial = sum(h * response.axial_stress_pa for h, response in zip(
            thicknesses, responses, strict=True
        )) - target_axial
        hoop = sum(h * response.hoop_stress_pa for h, response in zip(
            thicknesses, responses, strict=True
        )) - target_hoop
        return responses, axial, hoop

    def equations(strains: np.ndarray) -> np.ndarray:
        _, axial, hoop = evaluate(strains)
        return np.array([axial / scale, hoop / scale])

    # Elastic solution is a useful first load-step guess. Subsequent guesses
    # use the previously accepted total strains, preserving path direction.
    if previous_state == IncrementalShellState():
        elastic = solve_biaxial_linear_shell(case, inner_layer.material, outer_layer.material)
        guess = [elastic.common_axial_strain, elastic.inner.total_hoop_strain]
    else:
        guess = [previous_state.common_axial_strain, previous_state.common_hoop_strain]
    solution = root(equations, np.asarray(guess, dtype=float), tol=1e-9)
    responses, axial_residual, hoop_residual = evaluate(solution.x)
    force_tolerance = max(1e-5, 1e-9 * max(abs(target_axial), abs(target_hoop)))
    if max(abs(axial_residual), abs(hoop_residual)) > force_tolerance:
        raise RuntimeError("Incremental shell equilibrium did not converge")
    inner, outer = responses
    state = IncrementalShellState(
        float(solution.x[0]), float(solution.x[1]), inner.state, outer.state, geometry,
        case.stress_free_temperature_k, signature,
    )
    return IncrementalShellResult(
        case.radius_m * state.common_hoop_strain, state.common_axial_strain,
        inner, outer, state, hoop_residual, axial_residual, int(solution.nfev),
    )

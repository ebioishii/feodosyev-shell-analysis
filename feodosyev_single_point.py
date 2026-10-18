"""Single-point implementation of Feodosyev Chapter 9 section 9.3.

This module implements equations (9.1)-(9.6) exactly in their published
deformation-theory form.  It deliberately does not reuse the incremental 1D
plasticity algorithm in ``heat_then_pressurize.py`` and does not replace
equation (9.4) with a three-dimensional incremental J2 model.

All internal values use SI units: metre, pascal, and dimensionless strain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Protocol

import numpy as np
from scipy.optimize import brentq


SQRT_THREE = math.sqrt(3.0)
ZERO_STRAIN_TOLERANCE = 1e-14


@dataclass(frozen=True)
class UniaxialResponse:
    """Point on the uniaxial total-mechanical-strain tensile diagram."""

    total_mechanical_strain: float
    stress_pa: float
    elastic_strain: float
    plastic_strain: float
    branch: str


class TensionLaw(Protocol):
    """Minimum interface required by the Feodosyev constitutive equation."""

    elastic_modulus_pa: float
    max_total_mechanical_strain: float

    def evaluate(self, total_mechanical_strain: float) -> UniaxialResponse:
        """Return non-negative tensile stress for a non-negative total strain."""


@dataclass(frozen=True)
class LinearElasticTensionLaw:
    """Simple independent material law used for analytic verification."""

    elastic_modulus_pa: float
    max_total_mechanical_strain: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.elastic_modulus_pa) or self.elastic_modulus_pa <= 0:
            raise ValueError("elastic_modulus_pa must be finite and positive")
        if (
            not math.isfinite(self.max_total_mechanical_strain)
            or self.max_total_mechanical_strain <= 0
        ):
            raise ValueError(
                "max_total_mechanical_strain must be finite and positive"
            )

    def evaluate(self, total_mechanical_strain: float) -> UniaxialResponse:
        strain = _finite_nonnegative(
            total_mechanical_strain, "total mechanical strain"
        )
        if strain > self.max_total_mechanical_strain + ZERO_STRAIN_TOLERANCE:
            raise ValueError("strain intensity exceeds the linear material domain")
        strain = min(strain, self.max_total_mechanical_strain)
        return UniaxialResponse(
            total_mechanical_strain=strain,
            stress_pa=self.elastic_modulus_pa * strain,
            elastic_strain=strain,
            plastic_strain=0.0,
            branch="elastic",
        )


class GrantaSS316TensionLaw:
    """SS316 tensile diagram reconstructed from the local Granta tables.

    Granta's isotropic-hardening table gives stress versus *plastic* strain.
    Feodosyev equation (9.4), however, requires the uniaxial *total mechanical*
    strain.  For every table point this class therefore uses

        total mechanical strain = plastic strain + stress / E(T).

    Evaluation above first yield inverts that relation; it never treats the
    plastic-strain column as total strain.
    """

    def __init__(self, temperature_c: float) -> None:
        temperature_c = _finite(temperature_c, "temperature_c")
        # Keep the licensed/private data dependency lazy so analytic tests can
        # run without copying the local Granta files into Git.
        from interpolation import (  # pylint: disable=import-outside-toplevel
            E_at,
            flow_stress,
            plastic_strain_max_at,
        )

        self.temperature_c = temperature_c
        self._flow_stress = flow_stress
        self.elastic_modulus_pa = float(E_at(temperature_c))
        self.max_plastic_strain = float(plastic_strain_max_at(temperature_c))
        self.yield_stress_pa = float(flow_stress(temperature_c, 0.0))
        self.max_stress_pa = float(
            flow_stress(temperature_c, self.max_plastic_strain)
        )
        self.yield_total_mechanical_strain = (
            self.yield_stress_pa / self.elastic_modulus_pa
        )
        self.max_total_mechanical_strain = (
            self.max_plastic_strain
            + self.max_stress_pa / self.elastic_modulus_pa
        )
        if min(
            self.elastic_modulus_pa,
            self.yield_stress_pa,
            self.max_total_mechanical_strain,
        ) <= 0:
            raise ValueError("invalid non-positive Granta material property")

    def evaluate(self, total_mechanical_strain: float) -> UniaxialResponse:
        strain = _finite_nonnegative(
            total_mechanical_strain, "total mechanical strain"
        )
        domain_tolerance = max(
            ZERO_STRAIN_TOLERANCE,
            1e-12 * self.max_total_mechanical_strain,
        )
        if strain > self.max_total_mechanical_strain + domain_tolerance:
            raise ValueError(
                "strain intensity exceeds the Granta total-mechanical-strain "
                f"domain at {self.temperature_c:g} degC: "
                f"{strain:.9g} > {self.max_total_mechanical_strain:.9g}"
            )
        strain = min(strain, self.max_total_mechanical_strain)

        if strain <= self.yield_total_mechanical_strain:
            stress = self.elastic_modulus_pa * strain
            return UniaxialResponse(
                total_mechanical_strain=strain,
                stress_pa=stress,
                elastic_strain=strain,
                plastic_strain=0.0,
                branch="elastic",
            )

        def total_strain_error(plastic_strain: float) -> float:
            stress = self._flow_stress(self.temperature_c, plastic_strain)
            return (
                plastic_strain + stress / self.elastic_modulus_pa - strain
            )

        if math.isclose(
            strain,
            self.max_total_mechanical_strain,
            rel_tol=0.0,
            abs_tol=domain_tolerance,
        ):
            plastic_strain = self.max_plastic_strain
        else:
            plastic_strain = brentq(
                total_strain_error,
                0.0,
                self.max_plastic_strain,
                xtol=1e-14,
                rtol=1e-14,
            )
        stress = float(self._flow_stress(self.temperature_c, plastic_strain))
        elastic_strain = stress / self.elastic_modulus_pa
        return UniaxialResponse(
            total_mechanical_strain=strain,
            stress_pa=stress,
            elastic_strain=elastic_strain,
            plastic_strain=plastic_strain,
            branch="plastic",
        )


@dataclass(frozen=True)
class ShellResponse:
    """Strains and stresses for one shell at the converged single point."""

    total_axial_strain: float
    total_hoop_strain: float
    thermal_strain: float
    mechanical_axial_strain: float
    mechanical_hoop_strain: float
    strain_intensity: float
    stress_intensity_pa: float
    secant_modulus_pa: float
    axial_stress_pa: float
    hoop_stress_pa: float
    uniaxial_elastic_strain: float
    uniaxial_plastic_strain: float
    material_branch: str


@dataclass(frozen=True)
class EquilibriumResiduals:
    """Force-resultant residuals for equations (9.1), (9.2), and (9.5)."""

    equation_9_1_n_per_m: float
    equation_9_2_n_per_m: float
    equation_9_5_n_per_m: float
    tolerance_n_per_m: float
    maximum_relative: float


@dataclass(frozen=True)
class SinglePointResult:
    """Complete result of one prescribed-hoop-strain calculation."""

    radius_m: float
    inner_thickness_m: float
    outer_thickness_m: float
    k: float
    common_total_hoop_strain: float
    common_total_axial_strain: float
    gas_pressure_pa: float
    inner: ShellResponse
    outer: ShellResponse
    residuals: EquilibriumResiduals
    root_count: int
    root_selection: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def strain_intensity(
    mechanical_axial_strain: float, mechanical_hoop_strain: float
) -> float:
    """Feodosyev strain intensity printed below equation (9.4)."""

    eps_x = _finite(mechanical_axial_strain, "mechanical_axial_strain")
    eps_y = _finite(mechanical_hoop_strain, "mechanical_hoop_strain")
    radicand = eps_x * eps_x + eps_y * eps_y + eps_x * eps_y
    # The quadratic is non-negative analytically.  Clamp only roundoff.
    if radicand < -1e-30:
        raise ArithmeticError("negative strain-intensity radicand")
    return 2.0 / SQRT_THREE * math.sqrt(max(0.0, radicand))


def evaluate_shell(
    total_axial_strain: float,
    total_hoop_strain: float,
    thermal_strain_value: float,
    material: TensionLaw,
) -> ShellResponse:
    """Apply equations (9.3) and (9.4) to one shell."""

    total_x = _finite(total_axial_strain, "total_axial_strain")
    total_y = _finite(total_hoop_strain, "total_hoop_strain")
    thermal = _finite(thermal_strain_value, "thermal_strain")
    eps_x = total_x - thermal
    eps_y = total_y - thermal
    eps_i = strain_intensity(eps_x, eps_y)
    uniaxial = material.evaluate(eps_i)

    if eps_i <= ZERO_STRAIN_TOLERANCE:
        secant_modulus = material.elastic_modulus_pa
    else:
        secant_modulus = uniaxial.stress_pa / eps_i
    coefficient = 4.0 / 3.0 * secant_modulus
    sigma_x = coefficient * (eps_x + 0.5 * eps_y)
    sigma_y = coefficient * (eps_y + 0.5 * eps_x)

    if not all(math.isfinite(value) for value in (sigma_x, sigma_y)):
        raise ArithmeticError("equation (9.4) produced a non-finite stress")

    # Independent invariant check: the stresses produced by (9.4) must have
    # the same intensity as the tensile-diagram stress used in that equation.
    stress_check = math.sqrt(
        max(0.0, sigma_x * sigma_x + sigma_y * sigma_y - sigma_x * sigma_y)
    )
    invariant_tolerance = max(1e-5, 1e-10 * max(uniaxial.stress_pa, 1.0))
    if abs(stress_check - uniaxial.stress_pa) > invariant_tolerance:
        raise ArithmeticError("equation (9.4) stress-intensity check failed")

    return ShellResponse(
        total_axial_strain=total_x,
        total_hoop_strain=total_y,
        thermal_strain=thermal,
        mechanical_axial_strain=eps_x,
        mechanical_hoop_strain=eps_y,
        strain_intensity=eps_i,
        stress_intensity_pa=uniaxial.stress_pa,
        secant_modulus_pa=secant_modulus,
        axial_stress_pa=sigma_x,
        hoop_stress_pa=sigma_y,
        uniaxial_elastic_strain=uniaxial.elastic_strain,
        uniaxial_plastic_strain=uniaxial.plastic_strain,
        material_branch=uniaxial.branch,
    )


def _axial_total_strain_domain(
    total_hoop_strain: float,
    thermal_strain_value: float,
    material: TensionLaw,
) -> tuple[float, float]:
    """Exact axial interval that stays inside one material diagram."""

    mechanical_hoop = total_hoop_strain - thermal_strain_value
    maximum = material.max_total_mechanical_strain
    tolerance = max(ZERO_STRAIN_TOLERANCE, 1e-12 * maximum)
    if abs(mechanical_hoop) > maximum + tolerance:
        raise ValueError(
            "prescribed hoop strain alone exceeds a shell's material domain: "
            f"|{mechanical_hoop:.9g}| > {maximum:.9g}"
        )
    mechanical_hoop = math.copysign(min(abs(mechanical_hoop), maximum), mechanical_hoop)
    span = 0.5 * math.sqrt(
        max(0.0, 3.0 * (maximum * maximum - mechanical_hoop * mechanical_hoop))
    )
    center = thermal_strain_value - 0.5 * mechanical_hoop
    return center - span, center + span


def solve_single_point(
    *,
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    common_total_hoop_strain: float,
    inner_thermal_strain: float,
    outer_thermal_strain: float,
    inner_material: TensionLaw,
    outer_material: TensionLaw,
    k: float = 0.5,
    axial_strain_guess: float | None = None,
    allow_negative_pressure: bool = False,
    root_scan_points: int = 401,
    absolute_force_tolerance_n_per_m: float = 1e-3,
    relative_force_tolerance: float = 1e-10,
) -> SinglePointResult:
    """Solve one Feodosyev point for a prescribed common total hoop strain.

    Equation (9.5) is solved for the common total axial strain.  Equations
    (9.1), (9.2), and (9.5) are then checked independently, and pressure is
    calculated from equation (9.6).

    If more than one root exists inside the common material domain, the
    function refuses to choose silently unless ``axial_strain_guess`` is
    supplied.  A curve calculation would normally select a branch using the
    previous point, but that history is intentionally outside stage 1.
    """

    radius = _finite(radius_m, "radius_m")
    inner_h = _finite(inner_thickness_m, "inner_thickness_m")
    outer_h = _finite(outer_thickness_m, "outer_thickness_m")
    hoop_total = _finite(common_total_hoop_strain, "common_total_hoop_strain")
    inner_thermal = _finite(inner_thermal_strain, "inner_thermal_strain")
    outer_thermal = _finite(outer_thermal_strain, "outer_thermal_strain")
    axial_factor = _finite(k, "k")
    if min(radius, inner_h, outer_h) <= 0:
        raise ValueError("radius and shell thicknesses must be positive")
    if root_scan_points < 3:
        raise ValueError("root_scan_points must be at least 3")
    if absolute_force_tolerance_n_per_m <= 0 or relative_force_tolerance <= 0:
        raise ValueError("force tolerances must be positive")
    if axial_strain_guess is not None:
        axial_strain_guess = _finite(axial_strain_guess, "axial_strain_guess")

    inner_domain = _axial_total_strain_domain(
        hoop_total, inner_thermal, inner_material
    )
    outer_domain = _axial_total_strain_domain(
        hoop_total, outer_thermal, outer_material
    )
    lower = max(inner_domain[0], outer_domain[0])
    upper = min(inner_domain[1], outer_domain[1])
    if lower > upper:
        raise ValueError(
            "the two shells have no common axial-strain material domain"
        )

    def state_at(axial_total: float) -> tuple[ShellResponse, ShellResponse]:
        return (
            evaluate_shell(
                axial_total,
                hoop_total,
                inner_thermal,
                inner_material,
            ),
            evaluate_shell(
                axial_total,
                hoop_total,
                outer_thermal,
                outer_material,
            ),
        )

    def equation_9_5(axial_total: float) -> float:
        inner, outer = state_at(axial_total)
        axial_resultant = (
            inner.axial_stress_pa * inner_h
            + outer.axial_stress_pa * outer_h
        )
        hoop_resultant = (
            inner.hoop_stress_pa * inner_h
            + outer.hoop_stress_pa * outer_h
        )
        return axial_resultant - axial_factor * hoop_resultant

    grid = np.linspace(lower, upper, root_scan_points)
    values = [float(equation_9_5(float(point))) for point in grid]
    force_scale = max(max(abs(value) for value in values), 1.0)
    scan_zero_tolerance = max(
        absolute_force_tolerance_n_per_m,
        relative_force_tolerance * force_scale,
    )

    candidates: list[float] = []
    for point, value in zip(grid, values):
        if abs(value) <= scan_zero_tolerance:
            candidates.append(float(point))
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if f_left == 0.0 or f_right == 0.0 or f_left * f_right > 0.0:
            continue
        candidates.append(
            float(
                brentq(
                    equation_9_5,
                    float(left),
                    float(right),
                    xtol=1e-14,
                    rtol=1e-14,
                )
            )
        )

    candidates.sort()
    distinct_roots: list[float] = []
    root_merge_tolerance = max(1e-12, 1e-9 * max(abs(lower), abs(upper), 1e-3))
    for candidate in candidates:
        if (
            not distinct_roots
            or abs(candidate - distinct_roots[-1]) > root_merge_tolerance
        ):
            distinct_roots.append(candidate)
        elif abs(equation_9_5(candidate)) < abs(
            equation_9_5(distinct_roots[-1])
        ):
            distinct_roots[-1] = candidate

    if not distinct_roots:
        best_index = int(np.argmin(np.abs(values)))
        raise ValueError(
            "equation (9.5) has no root in the common material domain; "
            f"smallest sampled residual is {values[best_index]:.6g} N/m at "
            f"axial strain {float(grid[best_index]):.9g}"
        )

    if len(distinct_roots) == 1:
        axial_total = distinct_roots[0]
        root_selection = "unique root in the common material domain"
    elif axial_strain_guess is None:
        formatted = ", ".join(f"{root:.9g}" for root in distinct_roots)
        raise ValueError(
            "multiple equation (9.5) roots were found; provide "
            f"axial_strain_guess to select a branch: {formatted}"
        )
    else:
        distances = [abs(root - axial_strain_guess) for root in distinct_roots]
        order = np.argsort(distances)
        if len(order) > 1 and math.isclose(
            distances[int(order[0])],
            distances[int(order[1])],
            rel_tol=1e-12,
            abs_tol=1e-14,
        ):
            raise ValueError("axial_strain_guess is equally close to two roots")
        axial_total = distinct_roots[int(order[0])]
        root_selection = "root nearest the supplied axial_strain_guess"

    inner, outer = state_at(axial_total)
    hoop_resultant = (
        inner.hoop_stress_pa * inner_h + outer.hoop_stress_pa * outer_h
    )
    axial_resultant = (
        inner.axial_stress_pa * inner_h + outer.axial_stress_pa * outer_h
    )
    pressure = hoop_resultant / radius  # Equation (9.6)
    equation_9_1_residual = hoop_resultant - pressure * radius
    equation_9_2_residual = axial_resultant - axial_factor * pressure * radius
    equation_9_5_residual = axial_resultant - axial_factor * hoop_resultant
    final_force_scale = max(
        abs(hoop_resultant),
        abs(axial_resultant),
        abs(axial_factor * hoop_resultant),
        1.0,
    )
    final_tolerance = max(
        absolute_force_tolerance_n_per_m,
        relative_force_tolerance * final_force_scale,
    )
    residual_values = (
        equation_9_1_residual,
        equation_9_2_residual,
        equation_9_5_residual,
    )
    if max(abs(value) for value in residual_values) > final_tolerance:
        raise ArithmeticError(
            "the computed point failed the equilibrium residual check"
        )
    pressure_tolerance = final_tolerance / radius
    if not allow_negative_pressure and pressure < -pressure_tolerance:
        raise ValueError(
            f"the mathematical root requires negative gas pressure: {pressure:.9g} Pa"
        )

    return SinglePointResult(
        radius_m=radius,
        inner_thickness_m=inner_h,
        outer_thickness_m=outer_h,
        k=axial_factor,
        common_total_hoop_strain=hoop_total,
        common_total_axial_strain=axial_total,
        gas_pressure_pa=pressure,
        inner=inner,
        outer=outer,
        residuals=EquilibriumResiduals(
            equation_9_1_n_per_m=equation_9_1_residual,
            equation_9_2_n_per_m=equation_9_2_residual,
            equation_9_5_n_per_m=equation_9_5_residual,
            tolerance_n_per_m=final_tolerance,
            maximum_relative=max(abs(value) for value in residual_values)
            / final_force_scale,
        ),
        root_count=len(distinct_roots),
        root_selection=root_selection,
    )


def solve_ss316_single_point(
    *,
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    inner_temperature_c: float,
    outer_temperature_c: float,
    common_total_hoop_strain: float,
    k: float = 0.5,
    axial_strain_guess: float | None = None,
    allow_negative_pressure: bool = False,
) -> SinglePointResult:
    """Convenience wrapper using local Granta SS316 data for both shells."""

    from interpolation import thermal_strain  # pylint: disable=import-outside-toplevel

    inner_temperature = _finite(inner_temperature_c, "inner_temperature_c")
    outer_temperature = _finite(outer_temperature_c, "outer_temperature_c")
    return solve_single_point(
        radius_m=radius_m,
        inner_thickness_m=inner_thickness_m,
        outer_thickness_m=outer_thickness_m,
        common_total_hoop_strain=common_total_hoop_strain,
        inner_thermal_strain=thermal_strain(inner_temperature),
        outer_thermal_strain=thermal_strain(outer_temperature),
        inner_material=GrantaSS316TensionLaw(inner_temperature),
        outer_material=GrantaSS316TensionLaw(outer_temperature),
        k=k,
        axial_strain_guess=axial_strain_guess,
        allow_negative_pressure=allow_negative_pressure,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve one Feodosyev section 9.3 double-shell point"
    )
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument("--inner-thickness-mm", type=float, default=1.0)
    parser.add_argument("--outer-thickness-mm", type=float, default=2.0)
    parser.add_argument("--inner-temperature-c", type=float, default=538.0)
    parser.add_argument("--outer-temperature-c", type=float, default=20.0)
    parser.add_argument("--hoop-strain", type=float, default=0.002)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--axial-strain-guess", type=float)
    parser.add_argument("--allow-negative-pressure", action="store_true")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    result = solve_ss316_single_point(
        radius_m=args.radius_mm * 1e-3,
        inner_thickness_m=args.inner_thickness_mm * 1e-3,
        outer_thickness_m=args.outer_thickness_mm * 1e-3,
        inner_temperature_c=args.inner_temperature_c,
        outer_temperature_c=args.outer_temperature_c,
        common_total_hoop_strain=args.hoop_strain,
        k=args.k,
        axial_strain_guess=args.axial_strain_guess,
        allow_negative_pressure=args.allow_negative_pressure,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

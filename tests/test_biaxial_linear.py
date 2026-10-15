"""Analytic membrane cases derived from force resultants, not solver output."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from feodosyev_shell.materials import load_material
from feodosyev_shell.solvers import (
    BiaxialLinearCase,
    LinearShellCase,
    solve_biaxial_linear_shell,
    solve_linear_shell,
)

MATERIAL = load_material(Path(__file__).resolve().parents[1] / "examples" / "synthetic_shell")


def case(pressure=0, inner_temp=300, outer_temp=300, ratio=0.5):
    return BiaxialLinearCase(
        radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
        inner_temperature_k=inner_temp, outer_temperature_k=outer_temp,
        stress_free_temperature_k=300, gas_pressure_pa=pressure,
        axial_force_ratio=ratio,
    )


def test_closed_end_isothermal_hand_solution():
    # pR / total thickness = 100000 / .003 = 33 1/3 MPa hoop;
    # closed-end axial stress is half the hoop stress.
    result = solve_biaxial_linear_shell(case(pressure=1e6), MATERIAL)
    assert result.inner.hoop_stress_pa == pytest.approx(100e6 / 3)
    assert result.inner.axial_stress_pa == pytest.approx(50e6 / 3)
    assert result.outer.hoop_stress_pa == pytest.approx(result.inner.hoop_stress_pa)
    # eps_theta = (sigma_theta - nu sigma_x)/E = 0.000283333...
    assert result.radial_displacement_m == pytest.approx(0.1 * 85 / 300000)
    assert result.common_axial_strain == pytest.approx(1 / 15000)
    assert result.inner.von_mises_stress_pa == pytest.approx(100e6 / (2 * 3**0.5))
    assert result.hoop_balance_residual_n_per_m == pytest.approx(0, abs=1e-8)
    assert result.axial_balance_residual_n_per_m == pytest.approx(0, abs=1e-8)


def test_differential_heating_self_balances_both_directions():
    # The inner shell's free strain is .001, with 1:2 stiffness weights.
    result = solve_biaxial_linear_shell(case(inner_temp=400, ratio=0), MATERIAL)
    assert result.common_axial_strain == pytest.approx(1 / 3000)
    assert result.inner.total_hoop_strain == pytest.approx(1 / 3000)
    assert result.inner.axial_stress_pa == pytest.approx(-200e6 / 2.1)
    assert result.inner.hoop_stress_pa == pytest.approx(-200e6 / 2.1)
    assert result.outer.axial_stress_pa == pytest.approx(100e6 / 2.1)
    assert result.outer.hoop_stress_pa == pytest.approx(100e6 / 2.1)
    assert result.hoop_balance_residual_n_per_m == pytest.approx(0, abs=1e-8)
    assert result.axial_balance_residual_n_per_m == pytest.approx(0, abs=1e-8)


def test_uniform_heating_is_stress_free_and_zero_poisson_recovers_3a():
    heated = solve_biaxial_linear_shell(case(inner_temp=400, outer_temp=400), MATERIAL)
    assert heated.radial_displacement_m == pytest.approx(0.0001)
    assert heated.inner.hoop_stress_pa == pytest.approx(0, abs=1e-6)
    assert heated.outer.axial_stress_pa == pytest.approx(0, abs=1e-6)
    no_poisson = replace(
        MATERIAL, properties={**MATERIAL.properties,
                              "poisson_ratio": (np.array([300., 400.]), np.array([0., 0.]))},
    )
    biaxial = solve_biaxial_linear_shell(case(pressure=1e6, inner_temp=400, ratio=0),
                                        no_poisson)
    hoop_only = solve_linear_shell(LinearShellCase(
        0.1, 0.001, 0.002, 400, 300, 300, 1e6,
    ), no_poisson)
    assert biaxial.radial_displacement_m == pytest.approx(hoop_only.radial_displacement_m)
    assert biaxial.inner.hoop_stress_pa == pytest.approx(hoop_only.inner.hoop_stress_pa)


def test_missing_poisson_and_temperature_outside_table_rejected():
    no_poisson = replace(
        MATERIAL, properties={key: value for key, value in MATERIAL.properties.items()
                              if key != "poisson_ratio"},
    )
    with pytest.raises(ValueError, match="poisson_ratio"):
        solve_biaxial_linear_shell(case(pressure=1e6), no_poisson)
    with pytest.raises(ValueError, match="outside"):
        solve_biaxial_linear_shell(case(inner_temp=401), MATERIAL)


def test_von_mises_limit_and_invalid_ratio():
    result = solve_biaxial_linear_shell(case(pressure=1e7), MATERIAL)
    assert result.elastic_prediction_credible is True  # von Mises 288.7 MPa < 300 MPa
    result = solve_biaxial_linear_shell(case(pressure=1.1e7), MATERIAL)
    assert result.elastic_prediction_credible is False
    with pytest.raises(ValueError, match="finite"):
        case(ratio=float("nan"))

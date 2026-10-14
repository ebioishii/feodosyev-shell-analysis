"""Independent hand arithmetic for a two-ring membrane, separate from solver logic."""

from dataclasses import replace
from pathlib import Path

import pytest

from feodosyev_shell.materials import load_material
from feodosyev_shell.solvers import LinearShellCase, solve_linear_shell

MATERIAL = load_material(Path(__file__).resolve().parents[1] / "examples" / "synthetic_shell")


def case(pressure=0, inner_temp=300, outer_temp=300, inner_t=0.001, outer_t=0.002):
    return LinearShellCase(0.1, inner_t, outer_t, inner_temp, outer_temp, 300, pressure)


def test_zero_load_is_stress_free():
    result = solve_linear_shell(case(), MATERIAL)
    assert result.radial_displacement_m == 0
    assert result.inner.hoop_stress_pa == 0
    assert result.outer.hoop_stress_pa == 0
    assert result.elastic_prediction_credible is True


def test_isothermal_pressure_hand_calculation_and_balance():
    # 1 MPa * 0.1 m / (100 GPa * 0.003 m) = 1/3000 total strain.
    result = solve_linear_shell(case(1e6), MATERIAL)
    assert result.inner.total_hoop_strain == pytest.approx(1 / 3000)
    assert result.radial_displacement_m == pytest.approx(1 / 30000)
    assert result.inner.hoop_stress_pa == pytest.approx(1e8 / 3)
    assert result.outer.hoop_stress_pa == pytest.approx(1e8 / 3)
    assert result.inner.total_hoop_strain == result.outer.total_hoop_strain
    assert result.pressure_balance_residual_n_per_m == pytest.approx(0, abs=1e-8)


def test_thermal_difference_self_balance_and_signs():
    # Inner free strain is 0.001; stiffness weights are 1:2.
    result = solve_linear_shell(case(inner_temp=400), MATERIAL)
    assert result.inner.thermal_hoop_strain == pytest.approx(0.001)
    assert result.outer.thermal_hoop_strain == 0
    assert result.radial_displacement_m == pytest.approx(1 / 30000)
    assert result.inner.hoop_stress_pa == pytest.approx(-2e8 / 3)
    assert result.outer.hoop_stress_pa == pytest.approx(1e8 / 3)
    assert result.pressure_balance_residual_n_per_m == pytest.approx(0, abs=1e-8)


def test_combined_load_is_sum_of_two_independent_hand_terms():
    # Pressure contributes 1/3000 and differential heating contributes 1/3000.
    result = solve_linear_shell(case(1e6, inner_temp=400), MATERIAL)
    assert result.radial_displacement_m == pytest.approx(1 / 15000)
    assert result.inner.mechanical_hoop_strain == pytest.approx(-1 / 3000)
    assert result.outer.mechanical_hoop_strain == pytest.approx(2 / 3000)
    assert result.inner.hoop_stress_pa == pytest.approx(-1e8 / 3)
    assert result.outer.hoop_stress_pa == pytest.approx(2e8 / 3)


def test_thickness_trend_and_single_shell_limit():
    thick = solve_linear_shell(case(1e6), MATERIAL).radial_displacement_m
    thin = solve_linear_shell(case(1e6, outer_t=0.0001), MATERIAL).radial_displacement_m
    almost_single = solve_linear_shell(case(1e6, outer_t=1e-9), MATERIAL)
    assert thin > thick
    # Single shell: delta_R = p R^2 / (E t_i) = 0.0001 m.
    assert almost_single.radial_displacement_m == pytest.approx(0.0001, rel=2e-6)
    assert almost_single.inner.hoop_stress_pa == pytest.approx(1e8, rel=2e-6)


def test_temperature_bounds_and_elastic_range_flag():
    with pytest.raises(ValueError, match="outside"):
        solve_linear_shell(case(inner_temp=401), MATERIAL)
    with pytest.raises(ValueError, match="outside measured"):
        solve_linear_shell(LinearShellCase(0.1, 0.001, 0.002, 300, 300, 299, 0), MATERIAL)
    result = solve_linear_shell(case(1e7), MATERIAL)
    assert result.elastic_prediction_credible is False
    assert result.inner.elastic_status == "outside_selected_elastic_limit"


def test_missing_elastic_limit_stays_unknown():
    no_limit = replace(
        MATERIAL,
        properties={key: value for key, value in MATERIAL.properties.items()
                    if key != "yield_strength"},
    )
    result = solve_linear_shell(case(1e6), no_limit)
    assert result.elastic_prediction_credible is None
    assert result.inner.elastic_status == "elastic_limit_unknown"


def test_invalid_geometry_or_pressure_rejected():
    with pytest.raises(ValueError, match="finite positive"):
        case(inner_t=0)
    with pytest.raises(ValueError, match="Thin-shell"):
        case(inner_t=0.01, outer_t=0.001)
    with pytest.raises(ValueError, match="nonnegative"):
        case(pressure=-1)

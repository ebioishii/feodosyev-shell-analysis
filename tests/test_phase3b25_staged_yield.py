"""Independent thermal hand check and staged synthetic plasticity acceptance."""

import math

import pytest

from examples.phase3b25_staged_yield import run_path, synthetic_layers
from feodosyev_shell.solvers import (
    BiaxialLinearCase,
    IncrementalShellState,
    solve_incremental_shell,
)


def test_synthetic_layers_are_explicitly_not_ss316():
    inner, outer = synthetic_layers()
    assert inner.material.metadata["source_status"] == "synthetic"
    assert outer.material.metadata["source_status"] == "synthetic"
    assert "NOT_SS316" in inner.material.metadata["material"]
    assert inner.law.initial_yield_pa == 200e6
    assert outer.law.initial_yield_pa == 60e6


def test_thermal_restraint_matches_independent_elastic_hand_calculation():
    rows = run_path()
    heated = rows[4]
    # Free strains are 0.001 and 0. The common strain is thickness-weighted:
    # (1 mm * 0.001 + 2 mm * 0) / 3 mm = 1/3000.
    common_strain = 1 / 3000
    assert heated["radial_displacement_m"] == pytest.approx(0.1 * common_strain)
    # Plane-stress equibiaxial sigma = E/(1-nu) * (eps_common-eps_free).
    inner_stress = 100e9 / 0.7 * (common_strain - 0.001)
    outer_stress = 100e9 / 0.7 * common_strain
    assert heated["inner_hoop_stress_pa"] == pytest.approx(inner_stress, rel=1e-10)
    assert heated["outer_hoop_stress_pa"] == pytest.approx(outer_stress, rel=1e-10)
    assert heated["inner_axial_stress_pa"] == pytest.approx(inner_stress, rel=1e-10)
    assert heated["outer_axial_stress_pa"] == pytest.approx(outer_stress, rel=1e-10)
    assert inner_stress * 0.001 + outer_stress * 0.002 == pytest.approx(0, abs=1e-8)
    assert heated["inner_equivalent_plastic_strain"] == 0
    assert heated["outer_equivalent_plastic_strain"] == 0


def test_outer_yields_first_then_both_plastic_with_equilibrium():
    rows = run_path()
    pressure_rows = {row["pressure_pa"]: row for row in rows if row["pressure_pa"] > 0}
    assert pressure_rows[0.25e6]["stage"] == "elastic"
    assert pressure_rows[0.5e6]["stage"] == "outer_plastic"
    assert pressure_rows[4.25e6]["stage"] == "outer_plastic"
    assert pressure_rows[4.5e6]["stage"] == "both_plastic"
    assert pressure_rows[6e6]["inner_equivalent_plastic_strain"] > 0
    assert pressure_rows[6e6]["outer_equivalent_plastic_strain"] > 0
    for row in rows:
        assert abs(row["hoop_balance_residual_n_per_m"]) < 1e-4
        assert abs(row["axial_balance_residual_n_per_m"]) < 1e-4
    for key in ("inner_equivalent_plastic_strain", "outer_equivalent_plastic_strain"):
        values = [row[key] for row in rows]
        assert all(
            next_value >= value
            for value, next_value in zip(values, values[1:], strict=False)
        )


def test_elastic_outer_yield_bracket_comes_from_hand_formula():
    # Before first yield, outer thermal stress is +47.619 MPa in both axes.
    # Pressure adds q=pR/(ti+to), with hoop +q and axial +q/2.
    thermal_stress = (100e9 / 0.7) / 3000

    def outer_trial_mises(pressure_pa):
        q = pressure_pa * 0.1 / 0.003
        hoop = thermal_stress + q
        axial = thermal_stress + q / 2
        return math.sqrt(axial**2 - axial * hoop + hoop**2)

    assert outer_trial_mises(0.25e6) < 60e6
    assert outer_trial_mises(0.5e6) > 60e6
    # No temperature gradient at 0.5 MPa would still be below yield.
    inner, outer = synthetic_layers()
    cold = BiaxialLinearCase(0.1, 0.001, 0.002, 300, 300, 300, 0.5e6, 0.5)
    result = solve_incremental_shell(cold, inner, IncrementalShellState(), outer)
    assert result.state.inner.equivalent_plastic_strain == 0
    assert result.state.outer.equivalent_plastic_strain == 0


def test_step_refinement_reduces_endpoint_discrepancy():
    coarse, fine, finer = (run_path(step)[-1] for step in (0.25e6, 0.125e6, 0.0625e6))
    for key in ("radial_displacement_m", "inner_equivalent_plastic_strain"):
        first_change = abs(coarse[key] - fine[key])
        second_change = abs(fine[key] - finer[key])
        assert second_change < first_change
    assert abs(coarse["radial_displacement_m"] - finer["radial_displacement_m"]) < 1e-6

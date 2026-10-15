"""Synthetic shell load paths test equilibrium, history and step refinement."""

from dataclasses import replace
from pathlib import Path

import pytest

from feodosyev_shell.constitutive.bilinear_j2 import SyntheticBilinearJ2
from feodosyev_shell.materials import load_material
from feodosyev_shell.solvers import (
    BiaxialLinearCase,
    IncrementalShellState,
    SyntheticShellLayer,
    solve_incremental_shell,
)

MATERIAL = load_material(Path(__file__).resolve().parents[1] / "examples" / "synthetic_shell")
LAYER = SyntheticShellLayer(MATERIAL, SyntheticBilinearJ2(100e9, 0.3, 300e6, 10e9))


def case(pressure):
    return BiaxialLinearCase(0.1, 0.001, 0.002, 300, 300, 300, pressure, 0.5)


def run_path(pressures):
    state = IncrementalShellState()
    results = []
    for pressure in pressures:
        result = solve_incremental_shell(case(pressure), LAYER, state)
        assert result.hoop_balance_residual_n_per_m == pytest.approx(0, abs=1e-4)
        assert result.axial_balance_residual_n_per_m == pytest.approx(0, abs=1e-4)
        assert result.global_evaluations > 0
        state = result.state
        results.append(result)
    return results


def test_elastic_then_yield_and_unload_with_residual_displacement():
    results = run_path([0, 5e6, 8e6, 10e6, 12e6, 6e6, 0])
    assert results[1].state.inner.equivalent_plastic_strain == 0
    assert results[2].state.inner.equivalent_plastic_strain == 0
    # With sigma_x = sigma_theta/2, von Mises reaches 300 MPa
    # at p = 10.392... MPa; 10 MPa remains elastic.
    assert results[3].state.inner.equivalent_plastic_strain == 0
    assert results[4].state.inner.equivalent_plastic_strain > 0
    assert results[5].state.inner.equivalent_plastic_strain == pytest.approx(
        results[4].state.inner.equivalent_plastic_strain
    )
    assert results[-1].state.inner.equivalent_plastic_strain == pytest.approx(
        results[4].state.inner.equivalent_plastic_strain
    )
    assert results[-1].radial_displacement_m > 0
    assert results[-1].inner.hoop_stress_pa == pytest.approx(0, abs=1e-3)
    assert results[-1].outer.axial_stress_pa == pytest.approx(0, abs=1e-3)


def test_step_refinement_converges_and_previous_state_is_unchanged():
    coarse = run_path([0, 3e6, 6e6, 9e6, 12e6])
    fine = run_path([i * 1e6 for i in range(13)])
    assert fine[-1].radial_displacement_m == pytest.approx(
        coarse[-1].radial_displacement_m, rel=1e-5
    )
    assert fine[-1].state.inner.equivalent_plastic_strain == pytest.approx(
        coarse[-1].state.inner.equivalent_plastic_strain, rel=1e-5
    )
    assert coarse[0].state.inner.equivalent_plastic_strain == 0
    assert coarse[1].state.inner.equivalent_plastic_strain == 0


def test_non_synthetic_or_mismatched_material_rejected():
    unverified = replace(MATERIAL, metadata={**MATERIAL.metadata, "source_status": "unverified"})
    with pytest.raises(ValueError, match="synthetic"):
        solve_incremental_shell(case(0), replace(LAYER, material=unverified),
                                IncrementalShellState())
    with pytest.raises(ValueError, match="initial yield"):
        solve_incremental_shell(case(0), replace(LAYER, law=SyntheticBilinearJ2(
            100e9, 0.3, 200e6, 10e9,
        )), IncrementalShellState())
    accepted = solve_incremental_shell(case(0), LAYER, IncrementalShellState())
    with pytest.raises(ValueError, match="geometry"):
        solve_incremental_shell(replace(case(1e6), radius_m=0.11), LAYER, accepted.state)

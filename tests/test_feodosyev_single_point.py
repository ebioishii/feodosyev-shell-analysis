"""Verification tests for the independent Feodosyev single-point solver."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

from feodosyev_single_point import (
    GrantaSS316TensionLaw,
    LinearElasticTensionLaw,
    solve_single_point,
    solve_ss316_single_point,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MATERIAL_DIR = PROJECT_DIR / "database" / "Granta_SS316_annealed_csv"
REQUIRED_MATERIAL_FILES = (
    "Elasticity.csv",
    "Coefficient of Thermal Expansion.csv",
    "Zero-Thermal-Strain Reference Temperature.csv",
    "Isotropic Hardening.csv",
)
HAS_LOCAL_MATERIAL_DATA = all(
    (MATERIAL_DIR / name).is_file() for name in REQUIRED_MATERIAL_FILES
)


class AnalyticSinglePointTests(unittest.TestCase):
    def test_identical_linear_shells_match_closed_form_solution(self) -> None:
        modulus = 200e9
        material = LinearElasticTensionLaw(modulus)
        hoop_total = 0.0012
        thermal = 0.0002
        k = 0.35
        expected_mechanical_axial = (
            (k - 0.5) / (1.0 - 0.5 * k) * (hoop_total - thermal)
        )
        expected_total_axial = thermal + expected_mechanical_axial
        expected_hoop_stress = 4.0 / 3.0 * modulus * (
            (hoop_total - thermal) + 0.5 * expected_mechanical_axial
        )
        expected_pressure = expected_hoop_stress * (0.001 + 0.002) / 0.1

        result = solve_single_point(
            radius_m=0.1,
            inner_thickness_m=0.001,
            outer_thickness_m=0.002,
            common_total_hoop_strain=hoop_total,
            inner_thermal_strain=thermal,
            outer_thermal_strain=thermal,
            inner_material=material,
            outer_material=material,
            k=k,
        )

        self.assertAlmostEqual(
            result.common_total_axial_strain, expected_total_axial, places=12
        )
        self.assertAlmostEqual(
            result.inner.axial_stress_pa,
            k * result.inner.hoop_stress_pa,
            places=3,
        )
        self.assertAlmostEqual(
            result.outer.axial_stress_pa,
            k * result.outer.hoop_stress_pa,
            places=3,
        )
        self.assertGreater(result.gas_pressure_pa, 0.0)
        self.assertAlmostEqual(result.gas_pressure_pa, expected_pressure, places=4)
        self.assertEqual(result.root_count, 1)
        self.assertLessEqual(
            abs(result.residuals.equation_9_5_n_per_m),
            result.residuals.tolerance_n_per_m,
        )

    def test_zero_strain_limit_has_zero_stress_without_division_by_zero(self) -> None:
        material = LinearElasticTensionLaw(100e9)
        result = solve_single_point(
            radius_m=0.2,
            inner_thickness_m=0.001,
            outer_thickness_m=0.001,
            common_total_hoop_strain=0.0,
            inner_thermal_strain=0.0,
            outer_thermal_strain=0.0,
            inner_material=material,
            outer_material=material,
            k=0.5,
        )

        self.assertAlmostEqual(result.common_total_axial_strain, 0.0, places=12)
        self.assertAlmostEqual(result.gas_pressure_pa, 0.0, places=6)
        self.assertEqual(result.inner.stress_intensity_pa, 0.0)
        self.assertEqual(result.outer.stress_intensity_pa, 0.0)
        self.assertTrue(math.isfinite(result.inner.secant_modulus_pa))

    def test_material_domain_is_enforced(self) -> None:
        material = LinearElasticTensionLaw(
            100e9, max_total_mechanical_strain=0.001
        )
        with self.assertRaisesRegex(ValueError, "material domain"):
            solve_single_point(
                radius_m=0.2,
                inner_thickness_m=0.001,
                outer_thickness_m=0.001,
                common_total_hoop_strain=0.002,
                inner_thermal_strain=0.0,
                outer_thermal_strain=0.0,
                inner_material=material,
                outer_material=material,
            )


@unittest.skipUnless(
    HAS_LOCAL_MATERIAL_DATA,
    "local licensed Granta SS316 CSV files are not available",
)
class GrantaSinglePointTests(unittest.TestCase):
    def test_plastic_table_is_converted_to_total_mechanical_strain(self) -> None:
        from interpolation import E_at, flow_stress

        temperature = 538.0
        plastic_strain = 0.01
        expected_stress = flow_stress(temperature, plastic_strain)
        total_mechanical = plastic_strain + expected_stress / E_at(temperature)

        response = GrantaSS316TensionLaw(temperature).evaluate(total_mechanical)

        self.assertAlmostEqual(response.stress_pa, expected_stress, places=2)
        self.assertAlmostEqual(response.plastic_strain, plastic_strain, places=11)
        self.assertAlmostEqual(
            response.total_mechanical_strain,
            response.elastic_strain + response.plastic_strain,
            places=12,
        )
        self.assertEqual(response.branch, "plastic")

    def test_real_ss316_point_returns_all_stresses_pressure_and_residuals(self) -> None:
        result = solve_ss316_single_point(
            radius_m=0.1,
            inner_thickness_m=0.001,
            outer_thickness_m=0.002,
            inner_temperature_c=538.0,
            outer_temperature_c=20.0,
            common_total_hoop_strain=0.002,
            k=0.5,
        )

        self.assertTrue(math.isfinite(result.common_total_axial_strain))
        self.assertGreater(result.gas_pressure_pa, 0.0)
        self.assertEqual(result.root_count, 1)
        for shell in (result.inner, result.outer):
            self.assertTrue(math.isfinite(shell.axial_stress_pa))
            self.assertTrue(math.isfinite(shell.hoop_stress_pa))
            self.assertAlmostEqual(
                shell.strain_intensity,
                shell.uniaxial_elastic_strain
                + shell.uniaxial_plastic_strain,
                places=10,
            )
        for residual in (
            result.residuals.equation_9_1_n_per_m,
            result.residuals.equation_9_2_n_per_m,
            result.residuals.equation_9_5_n_per_m,
        ):
            self.assertLessEqual(abs(residual), result.residuals.tolerance_n_per_m)

    def test_root_is_independent_of_scan_density(self) -> None:
        from interpolation import thermal_strain

        common = dict(
            radius_m=0.1,
            inner_thickness_m=0.001,
            outer_thickness_m=0.002,
            common_total_hoop_strain=0.002,
            inner_thermal_strain=thermal_strain(538.0),
            outer_thermal_strain=thermal_strain(20.0),
            inner_material=GrantaSS316TensionLaw(538.0),
            outer_material=GrantaSS316TensionLaw(20.0),
            k=0.5,
        )
        coarse = solve_single_point(**common, root_scan_points=101)
        fine = solve_single_point(**common, root_scan_points=801)

        self.assertAlmostEqual(
            coarse.common_total_axial_strain,
            fine.common_total_axial_strain,
            places=12,
        )
        self.assertAlmostEqual(
            coarse.gas_pressure_pa,
            fine.gas_pressure_pa,
            places=3,
        )

    def test_k_is_configurable_and_changes_the_axial_solution(self) -> None:
        common = dict(
            radius_m=0.1,
            inner_thickness_m=0.001,
            outer_thickness_m=0.002,
            inner_temperature_c=538.0,
            outer_temperature_c=20.0,
            common_total_hoop_strain=0.002,
        )
        result_default = solve_ss316_single_point(**common, k=0.5)
        result_changed = solve_ss316_single_point(**common, k=0.3)

        self.assertNotAlmostEqual(
            result_default.common_total_axial_strain,
            result_changed.common_total_axial_strain,
            places=8,
        )
        self.assertAlmostEqual(result_changed.k, 0.3)


if __name__ == "__main__":
    unittest.main()

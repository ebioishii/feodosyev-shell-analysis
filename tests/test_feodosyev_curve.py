"""Tests for stage-2 continuation, checks, CSV, and PNG output."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest

from feodosyev_curve import (
    CSV_FIELDS,
    export_curve,
    solve_ss316_uniform_curve,
    solve_uniform_curve,
)
from feodosyev_single_point import LinearElasticTensionLaw


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


class AnalyticCurveTests(unittest.TestCase):
    def make_curve(self):
        material = LinearElasticTensionLaw(200e9)
        return solve_uniform_curve(
            start_hoop_strain=0.0003,
            end_hoop_strain=0.002,
            point_count=8,
            radius_m=0.1,
            inner_thickness_m=0.001,
            outer_thickness_m=0.002,
            inner_thermal_strain=0.0002,
            outer_thermal_strain=0.0002,
            inner_material=material,
            outer_material=material,
            k=0.35,
            root_scan_points=101,
        )

    def test_uniform_curve_matches_closed_form_at_every_point(self) -> None:
        curve = self.make_curve()
        modulus = 200e9
        thermal = 0.0002
        k = 0.35
        ratio = (k - 0.5) / (1.0 - 0.5 * k)

        self.assertEqual(len(curve.points), 8)
        self.assertAlmostEqual(curve.start_hoop_strain, 0.0003)
        self.assertAlmostEqual(curve.end_hoop_strain, 0.002)
        for index, point in enumerate(curve.points):
            solution = point.solution
            mechanical_hoop = solution.common_total_hoop_strain - thermal
            expected_mechanical_axial = ratio * mechanical_hoop
            expected_total_axial = thermal + expected_mechanical_axial
            expected_hoop_stress = 4.0 / 3.0 * modulus * (
                mechanical_hoop + 0.5 * expected_mechanical_axial
            )
            expected_pressure = expected_hoop_stress * 0.003 / 0.1

            self.assertAlmostEqual(
                solution.common_total_axial_strain,
                expected_total_axial,
                places=12,
            )
            self.assertAlmostEqual(
                solution.gas_pressure_pa, expected_pressure, places=4
            )
            self.assertGreaterEqual(solution.gas_pressure_pa, 0.0)
            self.assertEqual(solution.root_count, 1)
            self.assertLessEqual(
                abs(solution.residuals.equation_9_5_n_per_m),
                solution.residuals.tolerance_n_per_m,
            )
            if index == 0:
                self.assertIsNone(point.axial_strain_guess)
            elif index == 1:
                self.assertAlmostEqual(
                    point.axial_strain_guess,
                    curve.points[0].solution.common_total_axial_strain,
                )
            else:
                left = curve.points[index - 2].solution
                right = curve.points[index - 1].solution
                slope = (
                    right.common_total_axial_strain
                    - left.common_total_axial_strain
                ) / (
                    right.common_total_hoop_strain
                    - left.common_total_hoop_strain
                )
                expected_guess = right.common_total_axial_strain + slope * (
                    solution.common_total_hoop_strain
                    - right.common_total_hoop_strain
                )
                self.assertAlmostEqual(
                    point.axial_strain_guess, expected_guess, places=14
                )

        self.assertFalse(curve.diagnostics.endpoint_is_load_limit)
        self.assertFalse(curve.diagnostics.stage_classification_performed)
        self.assertEqual(curve.diagnostics.pressure_decrease_segment_count, 0)
        self.assertEqual(curve.diagnostics.multiple_root_point_count, 0)

    def test_negative_pressure_point_is_rejected_with_point_context(self) -> None:
        material = LinearElasticTensionLaw(100e9)
        with self.assertRaisesRegex(
            ValueError, r"curve point 1/3, total hoop strain 0"
        ):
            solve_uniform_curve(
                start_hoop_strain=0.0,
                end_hoop_strain=0.0002,
                point_count=3,
                radius_m=0.1,
                inner_thickness_m=0.001,
                outer_thickness_m=0.001,
                inner_thermal_strain=0.001,
                outer_thermal_strain=0.0,
                inner_material=material,
                outer_material=material,
                k=0.5,
                root_scan_points=101,
            )

    def test_csv_and_png_include_stresses_and_do_not_overwrite(self) -> None:
        curve = self.make_curve()
        with tempfile.TemporaryDirectory(
            prefix=".test-temp-", dir=PROJECT_DIR
        ) as directory:
            csv_path, png_path = export_curve(
                curve, Path(directory), "analytic-curve"
            )
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), CSV_FIELDS)
            self.assertEqual(len(rows), len(curve.points))
            for required in (
                "inner_axial_stress_mpa",
                "inner_hoop_stress_mpa",
                "outer_axial_stress_mpa",
                "outer_hoop_stress_mpa",
                "equation_9_5_residual_n_per_m",
                "root_count",
            ):
                self.assertIn(required, rows[0])
                self.assertNotEqual(rows[0][required], "")

            with self.assertRaises(FileExistsError):
                export_curve(curve, Path(directory), "analytic-curve")


@unittest.skipUnless(
    HAS_LOCAL_MATERIAL_DATA,
    "local licensed Granta SS316 CSV files are not available",
)
class GrantaCurveTests(unittest.TestCase):
    def test_ss316_curve_checks_every_point_and_preserves_continuity(self) -> None:
        curve = solve_ss316_uniform_curve(
            start_hoop_strain=0.00025,
            end_hoop_strain=0.005,
            point_count=17,
            radius_m=0.1,
            inner_thickness_m=0.001,
            outer_thickness_m=0.002,
            inner_temperature_c=538.0,
            outer_temperature_c=20.0,
            k=0.5,
            root_scan_points=201,
        )

        self.assertEqual(len(curve.points), 17)
        self.assertEqual(curve.diagnostics.multiple_root_point_count, 0)
        self.assertGreaterEqual(curve.diagnostics.minimum_pressure_pa, 0.0)
        self.assertTrue(
            math.isfinite(
                curve.diagnostics.maximum_absolute_axial_predictor_error
            )
        )
        for expected_index, point in enumerate(curve.points):
            self.assertEqual(point.index, expected_index)
            self.assertGreaterEqual(point.solution.gas_pressure_pa, 0.0)
            self.assertLessEqual(
                point.solution.residuals.maximum_relative, 1e-9
            )
            self.assertIn(point.solution.inner.material_branch, {"elastic", "plastic"})
            self.assertIn(point.solution.outer.material_branch, {"elastic", "plastic"})
            if expected_index:
                self.assertIsNotNone(point.axial_strain_guess)
                self.assertIsNotNone(point.axial_strain_increment)
                self.assertIsNotNone(point.axial_predictor_error)


if __name__ == "__main__":
    unittest.main()

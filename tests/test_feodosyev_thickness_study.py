"""Thickness conservation, signed forces, baseline and matched-pressure checks."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from feodosyev_single_point import LinearElasticTensionLaw, solve_single_point
from feodosyev_thickness_study import (
    DEFAULT_RATIOS, ThicknessSettings, membrane_quantities,
    run_ratio_case, run_thickness_study, thicknesses_m,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MATERIAL_DIR = PROJECT_DIR / "database" / "Granta_SS316_annealed_csv"
HAS_MATERIAL = all((MATERIAL_DIR / f).is_file() for f in (
    "Elasticity.csv", "Coefficient of Thermal Expansion.csv",
    "Zero-Thermal-Strain Reference Temperature.csv", "Isotropic Hardening.csv",
))


class AnalyticThicknessTests(unittest.TestCase):
    def test_requested_thicknesses_and_si_conversion(self):
        expected_mm = ((2, 1), (1.5, 1.5), (1.2, 1.8), (1, 2), (0.75, 2.25), (0.6, 2.4))
        for eta, (hi, ho) in zip(DEFAULT_RATIOS, expected_mm):
            inner, outer = thicknesses_m(eta)
            self.assertAlmostEqual(inner * 1000, hi, places=12)
            self.assertAlmostEqual(outer * 1000, ho, places=12)
            self.assertAlmostEqual(inner + outer, 0.003, places=15)
            self.assertAlmostEqual(outer / inner, eta, places=12)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                thicknesses_m(invalid)

    def test_signed_membranes_match_hand_calculation_and_suppress_bad_percentages(self):
        modulus = 100e9
        material = LinearElasticTensionLaw(modulus)
        inner_h, outer_h = thicknesses_m(0.5)
        thermal, axial = 0.001, 2 / 3 * 0.001
        for hoop in (axial, 0.0007, 0.002):
            point = solve_single_point(
                radius_m=0.1, inner_thickness_m=inner_h, outer_thickness_m=outer_h,
                inner_thermal_strain=thermal, outer_thermal_strain=0,
                inner_material=material, outer_material=material,
                common_total_hoop_strain=hoop, k=0.5,
                root_scan_points=101, allow_negative_pressure=True,
            )
            forces = membrane_quantities(point)
            expected_inner_stress = 4 / 3 * modulus * (hoop - thermal + 0.5 * (axial - thermal))
            expected_outer_stress = 4 / 3 * modulus * (hoop + 0.5 * axial)
            self.assertAlmostEqual(forces["inner_hoop_membrane_n_per_m"], inner_h * expected_inner_stress, delta=1e-4)
            self.assertAlmostEqual(forces["outer_hoop_membrane_n_per_m"], outer_h * expected_outer_stress, delta=1e-4)
            self.assertLess(abs(forces["hoop_membrane_residual_n_per_m"]), 1e-4)
            self.assertLess(abs(forces["axial_membrane_residual_n_per_m"]), 1e-4)
            if hoop < 0.001:
                self.assertIsNone(forces["inner_membrane_fraction"])
                self.assertIsNone(forces["outer_membrane_fraction"])
            else:
                self.assertEqual(forces["membrane_share_status"], "nonnegative_resultants")
                self.assertAlmostEqual(forces["inner_membrane_fraction"] + forces["outer_membrane_fraction"], 1, places=10)

    def test_invalid_case_is_retained_and_cli_fails_explicitly(self):
        case = run_ratio_case(0, ThicknessSettings(point_count=5, root_scan_points=51))
        self.assertEqual(case.status, "incomplete")
        self.assertIn("positive", case.summary["error"])
        with tempfile.TemporaryDirectory(prefix=".test-temp-", dir=PROJECT_DIR) as temporary:
            directory = Path(temporary) / "invalid"
            result = subprocess.run(
                [sys.executable, "-m", "feodosyev_thickness_study", "--ratios", "0", "--points", "5",
                 "--output-directory", str(directory)], cwd=PROJECT_DIR,
                capture_output=True, text=True, encoding="utf-8",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete", result.stdout)
            with (directory / "thickness_summary.csv").open(encoding="utf-8-sig") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "incomplete")
            self.assertNotEqual(row["error"], "")
            self.assertEqual(row["inner_minimum_ratio"], "")


@unittest.skipUnless(HAS_MATERIAL, "local licensed Granta material files unavailable")
class GrantaThicknessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = ThicknessSettings(point_count=49, root_scan_points=101,
                                         origin_scan_points=9, reverse_check_points=5,
                                         common_pressure_count=3)
        cls.cases, cls.comparison = run_thickness_study(DEFAULT_RATIOS, cls.settings)

    def test_eta_two_reproduces_stage3_and_stage4a_baseline(self):
        case = next(c for c in self.cases if c.eta == 2)
        self.assertEqual(case.status, "complete", case.summary.get("error"))
        s = case.summary
        self.assertAlmostEqual(s["inner_minimum"]["ratio"], 1.0625307870542, delta=1e-7)
        self.assertAlmostEqual(s["outer_initial_yield"]["first_rising"]["hoop_strain"], 0.0009564477652, delta=1e-10)
        self.assertAlmostEqual(s["inner_hoop_zero"]["first_rising"]["hoop_strain"], 0.0119010767821, delta=1e-10)
        self.assertAlmostEqual(case.analysis.curve.points[-1].solution.gas_pressure_pa, 9870053.090485204, delta=1e-3)

    def test_every_case_has_valid_equilibrium_domain_and_continuation(self):
        self.assertEqual(len(self.cases), 6)
        saw_opposed_increment = False
        for case in self.cases:
            with self.subTest(eta=case.eta):
                self.assertEqual(case.status, "complete", case.summary.get("error"))
                self.assertEqual(case.summary["maximum_reverse_axial_difference"], 0)
                self.assertFalse(case.summary["endpoint_is_load_limit"])
                self.assertFalse(case.summary["stage_classification_performed"])
                for point, row in zip(case.analysis.curve.points, case.rows):
                    solution = point.solution
                    self.assertAlmostEqual(solution.inner_thickness_m + solution.outer_thickness_m, 0.003, places=15)
                    self.assertGreaterEqual(solution.gas_pressure_pa, 0)
                    self.assertEqual(solution.root_count, 1)
                    for field in ("hoop_membrane_residual_n_per_m", "axial_membrane_residual_n_per_m",
                                  "equation_9_5_residual_n_per_m"):
                        self.assertLessEqual(abs(row[field]), solution.residuals.tolerance_n_per_m)
                    self.assertLessEqual(row["inner_domain_utilization"], 1)
                    self.assertLessEqual(row["outer_domain_utilization"], 1)
                    self.assertAlmostEqual(
                        row["pressure_slope_implicit_pa_per_strain"],
                        row["inner_total_pressure_slope_pa_per_strain"] + row["outer_total_pressure_slope_pa_per_strain"], delta=1e-4,
                    )
                    if row["inner_hoop_membrane_n_per_m"] < 0 and row["inner_total_pressure_slope_pa_per_strain"] > 0:
                        saw_opposed_increment = True
                        self.assertIsNone(row["inner_membrane_fraction"])
        self.assertTrue(saw_opposed_increment)

    def test_below_reference_intervals_and_events_are_verified(self):
        for case in self.cases:
            s = case.summary
            self.assertEqual(s["inner_minimum"]["below_initial_yield"], bool(s["below_initial_yield_intervals_with_pressures"]))
            for interval in s["below_initial_yield_intervals_with_pressures"]:
                self.assertLess(interval["verified_interior_ratio"], 1)
                self.assertGreaterEqual(interval["lower_pressure_pa"], 0)
                self.assertGreaterEqual(interval["upper_pressure_pa"], 0)
            for key in ("outer_initial_yield", "inner_hoop_zero", "inner_initial_yield_crossings"):
                tolerance = 1 if key == "inner_hoop_zero" else 1e-7
                self.assertTrue(all(abs(c["condition_residual"]) <= tolerance for c in s[key]["crossings"]))

    def test_equal_pressure_states_use_extra_roots_and_signed_force_balance(self):
        self.assertEqual(self.comparison["status"], "complete")
        targets = set(self.comparison["targets_pa"])
        for case in self.cases:
            self.assertEqual({r["target_pressure_pa"] for r in case.matched_pressure_rows}, targets)
            for row in case.matched_pressure_rows:
                self.assertLessEqual(abs(row["pressure_matching_residual_pa"]), 1)
                force = row["inner_hoop_membrane_n_per_m"] + row["outer_hoop_membrane_n_per_m"]
                self.assertAlmostEqual(force, row["gas_pressure_pa"] * row["radius_m"], delta=1e-3)
                self.assertAlmostEqual(force, row["target_pressure_pa"] * row["radius_m"], delta=0.1)
                if row["inner_hoop_membrane_n_per_m"] < 0:
                    self.assertEqual(row["membrane_share_status"], "opposed_sign_resultants")
                    self.assertIsNone(row["inner_membrane_fraction"])


if __name__ == "__main__":
    unittest.main()

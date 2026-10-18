"""Zero-pressure origins, explicit failures, and the seven-temperature study."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from feodosyev_diagnostics import MaterialReference
from feodosyev_single_point import LinearElasticTensionLaw
from feodosyev_temperature_study import (
    CaseComputationError, DEFAULT_TEMPERATURES, StudySettings,
    export_temperature_study, run_ss316_case, run_temperature_study,
    solve_material_case, temperature_domains,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MATERIAL_DIR = PROJECT_DIR / "database" / "Granta_SS316_annealed_csv"
HAS_MATERIAL = all((MATERIAL_DIR / f).is_file() for f in (
    "Elasticity.csv", "Coefficient of Thermal Expansion.csv",
    "Zero-Thermal-Strain Reference Temperature.csv", "Isotropic Hardening.csv",
))


class AnalyticTemperatureStudyTests(unittest.TestCase):
    def test_zero_pressure_and_minimum_match_elastic_closed_form(self):
        material = LinearElasticTensionLaw(100e9)
        reference = MaterialReference(material, 1e9)
        settings = StudySettings(point_count=17, root_scan_points=101, origin_scan_points=5, reverse_check_points=3)
        thermal = 0.0002
        case = solve_material_case(100, settings, reference, reference, thermal, 0)
        origin = case.summary["zero_pressure_origin"]
        self.assertAlmostEqual(origin["zero_pressure_hoop_strain"], thermal / 3, places=11)
        self.assertGreaterEqual(origin["actual_pressure_pa"], 0)
        self.assertLessEqual(origin["actual_pressure_pa"], origin["pressure_tolerance_pa"])
        self.assertFalse(origin["pressure_clipped"])
        self.assertTrue(any(float(r.get("gas_pressure_pa", 0)) < 0 for r in case.evaluations))
        minimum = case.summary["inner_minimum"]
        self.assertAlmostEqual(minimum["hoop_strain"], 4 / 3 * thermal, delta=1e-8)
        self.assertAlmostEqual(minimum["ratio"], (2 / 3 * thermal) * 100e9 / 1e9, places=7)
        self.assertTrue(minimum["below_initial_yield"])
        expected_slope = 4 / 3 * 100e9 * 0.003 / 0.1
        for row in case.diagnostics.rows:
            self.assertAlmostEqual(row["pressure_slope_implicit_pa_per_strain"] / expected_slope, 1, places=8)
        with tempfile.TemporaryDirectory(prefix=".test-temp-", dir=PROJECT_DIR) as temporary:
            output = Path(temporary) / "run"
            manifest = export_temperature_study((case,), settings, output)
            self.assertTrue(manifest["all_cases_complete"])
            self.assertEqual(len(manifest["plots"]), 3)
            for path in manifest["plots"]:
                self.assertTrue(Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with (output / "temperature_summary.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(next(csv.DictReader(stream))["status"], "complete")
            with self.assertRaises(FileExistsError):
                export_temperature_study((case,), settings, output)

    def test_no_valid_material_interval_raises_with_trace_instead_of_skipping(self):
        material = LinearElasticTensionLaw(100e9, max_total_mechanical_strain=1e-4)
        reference = MaterialReference(material, 1e6)
        with self.assertRaises(CaseComputationError) as caught:
            solve_material_case(100, StudySettings(point_count=5, root_scan_points=51),
                                reference, reference, 0.01, 0)
        self.assertIn("zero-pressure origin", str(caught.exception))
        self.assertTrue(caught.exception.evaluations)
        self.assertTrue(caught.exception.evaluations[-1]["error"])

    def test_invalid_study_settings_are_explicit(self):
        with self.assertRaises(ValueError):
            StudySettings(point_count=2)
        with self.assertRaises(ValueError):
            run_temperature_study((300, 300))


@unittest.skipUnless(HAS_MATERIAL, "local licensed Granta material files unavailable")
class GrantaTemperatureStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = StudySettings(point_count=49, root_scan_points=101,
                                     origin_scan_points=9, reverse_check_points=5)
        cls.cases = run_temperature_study(DEFAULT_TEMPERATURES, cls.settings)

    def test_requested_temperatures_are_valid_and_every_case_passes(self):
        domains = temperature_domains()
        self.assertEqual(len(self.cases), 7)
        for case in self.cases:
            with self.subTest(temperature=case.temperature_c):
                for lower, upper in domains.values():
                    self.assertLessEqual(lower, case.temperature_c)
                    self.assertLessEqual(case.temperature_c, upper)
                self.assertEqual(case.status, "complete", case.summary.get("error"))
                self.assertEqual(case.summary["maximum_reverse_axial_difference"], 0)
                for point in case.curve.points:
                    solution = point.solution
                    self.assertGreaterEqual(solution.gas_pressure_pa, 0)
                    self.assertEqual(solution.root_count, 1)
                    for residual in (solution.residuals.equation_9_1_n_per_m,
                                     solution.residuals.equation_9_2_n_per_m,
                                     solution.residuals.equation_9_5_n_per_m):
                        self.assertLessEqual(abs(residual), solution.residuals.tolerance_n_per_m)
                for row in case.diagnostics.rows:
                    self.assertLessEqual(row["inner_domain_utilization"], 1)
                    self.assertLessEqual(row["outer_domain_utilization"], 1)
                for field in ("outer_initial_yield", "inner_initial_yield_crossings", "inner_hoop_zero"):
                    tolerance = 1.0 if field == "inner_hoop_zero" else 1e-7
                    for event in case.summary[field]["crossings"]:
                        self.assertLessEqual(abs(event["condition_residual"]), tolerance)
                self.assertFalse(case.summary["endpoint_is_load_limit"])
                self.assertFalse(case.summary["stage_classification_performed"])

    def test_538_reproduces_stage3_reference(self):
        case = next(c for c in self.cases if c.temperature_c == 538)
        self.assertEqual(case.status, "complete", case.summary.get("error"))
        s = case.summary
        self.assertAlmostEqual(s["inner_minimum"]["ratio"], 1.06253, delta=5e-6)
        self.assertAlmostEqual(s["outer_initial_yield"]["first_rising"]["hoop_strain"], 0.0009564477652, delta=1e-10)
        self.assertAlmostEqual(s["inner_hoop_zero"]["first_rising"]["hoop_strain"], 0.0119010767821, delta=1e-10)
        self.assertAlmostEqual(case.curve.points[-1].solution.gas_pressure_pa, 9870053.090485204, delta=1e-3)

    def test_outside_temperature_is_incomplete_and_cli_exits_nonzero(self):
        case = run_ss316_case(900, self.settings)
        self.assertEqual(case.status, "incomplete")
        self.assertIn("no extrapolation", case.summary["error"])
        self.assertIsNone(case.curve)
        with tempfile.TemporaryDirectory(prefix=".test-temp-", dir=PROJECT_DIR) as temporary:
            output = Path(temporary) / "failed-run"
            completed = subprocess.run(
                [sys.executable, "-m", "feodosyev_temperature_study", "--temperatures", "900",
                 "--points", "5", "--output-directory", str(output)],
                cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("incomplete", completed.stdout)
            with (output / "temperature_summary.csv").open(encoding="utf-8-sig") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "incomplete")
            self.assertNotEqual(row["error"], "")
            self.assertEqual(row["inner_minimum_ratio"], "")


if __name__ == "__main__":
    unittest.main()

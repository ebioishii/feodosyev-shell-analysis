"""Regression checks for the pre-existing one-dimensional model and UI backend."""

from __future__ import annotations

import csv
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_DIR / "1D html"
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


@unittest.skipUnless(
    HAS_LOCAL_MATERIAL_DATA,
    "local licensed Granta SS316 CSV files are not available",
)
class ExistingOneDimensionalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(APP_DIR))
        cls.app = importlib.import_module("app")

    @classmethod
    def tearDownClass(cls) -> None:
        if sys.path and sys.path[0] == str(APP_DIR):
            sys.path.pop(0)

    @staticmethod
    def base_payload() -> dict[str, object]:
        return {
            "radius_mm": 100.0,
            "inner_thickness_mm": 1.0,
            "outer_thickness_mm": 2.0,
            "inner_temperature_c": 538.0,
            "outer_temperature_c": 20.0,
            "heating_from_20": True,
            "auto_strain_range": True,
            "max_strain_percent": 1.0,
            "sample_count": 101,
        }

    def test_heating_on_auto_range_and_hoop_balance(self) -> None:
        response = self.app.generate_result(self.base_payload())
        stored = self.app.LATEST_RESULT

        self.assertIsNotNone(stored)
        self.assertGreaterEqual(response["point_count"], 2)
        self.assertAlmostEqual(response["pressure_mpa"][0], 0.0, places=7)
        self.assertTrue(all(math.isfinite(value) for value in response["pressure_mpa"]))

        radius = stored["initial_state"]["R_m"]
        inner_h = stored["initial_state"]["h_i_m"]
        outer_h = stored["initial_state"]["h_o_m"]
        for row in stored["history"]:
            residual = (
                inner_h * row["sigma_i"]
                + outer_h * row["sigma_o"]
                - radius * row["p_g"]
            )
            self.assertLessEqual(abs(residual), 1e-8)

    def test_saved_json_loads_with_heating_off_and_manual_range(self) -> None:
        automatic = self.app.generate_result(self.base_payload())
        initial_state = dict(self.app.LATEST_RESULT["initial_state"])

        payload = self.base_payload()
        payload.update(
            heating_from_20=False,
            initial_state=initial_state,
            auto_strain_range=False,
            max_strain_percent=(initial_state["epsilon_theta0"] + 0.001) * 100.0,
            sample_count=11,
        )
        manual = self.app.generate_result(payload)

        self.assertEqual(manual["point_count"], 11)
        self.assertAlmostEqual(
            manual["initial_strain"], initial_state["epsilon_theta0"], places=12
        )
        self.assertAlmostEqual(
            manual["pressure_mpa"][0], initial_state["p_start_Pa"] / 1e6, places=7
        )
        self.assertNotEqual(automatic["result_id"], manual["result_id"])

    def test_png_csv_and_json_exports(self) -> None:
        response = self.app.generate_result(self.base_payload())
        with tempfile.TemporaryDirectory(
            prefix=".test-temp-", dir=PROJECT_DIR
        ) as directory:
            exported = {}
            for extension in ("png", "csv", "json"):
                exported[extension] = self.app.save_result(
                    {
                        "result_id": response["result_id"],
                        "file_path": directory,
                        "filename": f"stage0-{extension}",
                    },
                    extension,
                )

            self.assertTrue(exported["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with exported["csv"].open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), response["point_count"])

            state = json.loads(exported["json"].read_text(encoding="utf-8"))
            self.assertEqual(state["format"], "double_shell_initial_state_v1")

    def test_required_static_ui_files_exist(self) -> None:
        for path in self.app.STATIC_FILES.values():
            self.assertTrue(path.is_file(), path)


if __name__ == "__main__":
    unittest.main()

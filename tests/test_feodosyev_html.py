"""Input/export/API acceptance for the separate Feodosyev HTML adapter."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from feodosyev_curve import solve_ss316_uniform_curve

ROOT = Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "database/Granta_SS316_annealed_csv"
HAS_DATA = all((MATERIAL / name).is_file() for name in (
    "Elasticity.csv", "Coefficient of Thermal Expansion.csv",
    "Zero-Thermal-Strain Reference Temperature.csv", "Isotropic Hardening.csv"))


@unittest.skipUnless(HAS_DATA, "local licensed Granta SS316 CSV files are not available")
class FeodosyevHTMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("feodosyev_html_app", ROOT / "Feodosyev html/app.py")
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)
        cls.default_response = cls.app.generate_result(dict(cls.app.DEFAULTS))
        cls.default_stored = cls.app.LATEST_RESULT

    def settings(self, **changes):
        return {**self.app.DEFAULTS, **changes}

    def document(self, **changes):
        return {"format": self.app.CONFIG_FORMAT, "units": dict(self.app.UNITS),
                "settings": self.settings(**changes)}

    def test_default_auto_and_stage2_reference(self):
        response = self.default_response
        self.assertEqual(response["point_count"], 481)
        self.assertAlmostEqual(response["strain_percent"][-1], 3)
        origin = response["summary"]["zero_pressure_origin"]
        self.assertGreaterEqual(response["pressure_mpa"][0], 0)
        self.assertLessEqual(response["pressure_mpa"][0] * 1e6, origin["pressure_tolerance_pa"])
        self.assertAlmostEqual(response["pressure_mpa"][-1] * 1e6, 9870053.090485204, delta=1e-3)
        self.assertFalse(response["summary"]["endpoint_is_load_limit"])
        # Same prescribed range as stage 2, compared at every point and stress.
        settings = self.settings(auto_strain_range=False, max_strain_percent=0.5, sample_count=17)
        self.app.generate_result(settings)
        actual = self.app.LATEST_RESULT["curve"]
        expected = solve_ss316_uniform_curve(
            start_hoop_strain=0.00025, end_hoop_strain=0.005, point_count=17,
            radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
            inner_temperature_c=538, outer_temperature_c=20, k=0.5)
        self.assertEqual(actual, expected)

    def test_k_changes_solution_and_passes_through(self):
        settings = self.settings(auto_strain_range=False, max_strain_percent=0.5, sample_count=3, k=0.35)
        response = self.app.generate_result(settings)
        expected = solve_ss316_uniform_curve(
            start_hoop_strain=0.00025, end_hoop_strain=0.005, point_count=3,
            radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
            inner_temperature_c=538, outer_temperature_c=20, k=0.35)
        self.assertEqual(self.app.LATEST_RESULT["curve"], expected)
        baseline = self.app.generate_result({**settings, "k": 0.5})
        self.assertGreater(abs(response["pressure_mpa"][-1] - baseline["pressure_mpa"][-1]), 0.01)

    def test_full_import_validation_and_no_mutation(self):
        document = self.document(radius_mm=125, inner_thickness_mm=2, outer_thickness_mm=3,
                                 inner_temperature_c=400, outer_temperature_c=100, k=0.35,
                                 auto_strain_range=False, start_strain_percent=0.2,
                                 max_strain_percent=2, sample_count=21)
        before = copy.deepcopy(document)
        self.assertEqual(self.app.import_settings(document), document["settings"])
        self.assertEqual(document, before)
        invalid = []
        for path, value in (("k", float("inf")), ("k", True), ("k", "0.5"),
                            ("radius_mm", 0), ("inner_temperature_c", 900),
                            ("sample_count", 2.5), ("sample_count", 5001),
                            ("auto_strain_range", 1), ("max_strain_percent", 0.1)):
            item = copy.deepcopy(document)
            item["settings"][path] = value
            invalid.append(item)
        missing = copy.deepcopy(document)
        del missing["settings"]["k"]
        invalid.extend([missing, [], {"format": "double_shell_initial_state_v1"},
                        {**document, "units": {**document["units"], "strain": "fraction"}}])
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(self.app.UserFacingError):
                self.app.import_settings(item)
        self.assertEqual(document, before)

    def test_invalid_scans_and_solver_k_fail_clearly(self):
        for changes, expected in (
            ({"start_strain_percent": 0, "max_strain_percent": 0.1}, "negative gas pressure"),
            ({"max_strain_percent": 100}, "material"),
            ({"max_strain_percent": 0.01}, "End Strain"),
            ({"k": 2}, "reliable|root|根|可靠"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(self.app.UserFacingError, expected):
                self.app.generate_result(self.settings(auto_strain_range=False, sample_count=3, **changes))
            self.assertIsNone(self.app.LATEST_RESULT)

    def test_png_csv_json_roundtrip_and_preserve_outputs(self):
        self.app.LATEST_RESULT = self.default_stored
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            payload = {"result_id": self.default_response["result_id"], "file_path": directory, "filename": "curve"}
            paths = {ext: self.app.save_result(payload, ext) for ext in ("png", "csv", "json")}
            self.assertEqual(paths["png"].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            from PIL import Image
            with Image.open(paths["png"]) as image:
                image.verify()
            with paths["csv"].open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 481)
            for key in ("total_axial_strain", "total_hoop_strain", "gas_pressure_pa",
                        "inner_axial_stress_pa", "inner_hoop_stress_pa",
                        "outer_axial_stress_pa", "outer_hoop_stress_pa"):
                self.assertNotEqual(rows[0][key], "")
            loaded = self.app.import_settings(json.loads(paths["json"].read_text(encoding="utf-8")))
            self.assertEqual(loaded, self.app.DEFAULTS)
            original = self.default_stored["curve"]
            # Reproduce the full scan, not just one saved point.
            rerun = self.app.generate_result(loaded)
            self.assertEqual(self.app.LATEST_RESULT["curve"], original)
            payload["result_id"] = rerun["result_id"]
            contents = {ext: path.read_bytes() for ext, path in paths.items()}
            for ext in paths:
                with self.assertRaisesRegex(self.app.UserFacingError, "已存在"):
                    self.app.save_result(payload, ext)
                self.assertEqual(paths[ext].read_bytes(), contents[ext])
            with self.assertRaisesRegex(self.app.UserFacingError, "檔名無效"):
                self.app.save_result({**payload, "filename": "../outside"}, "json")
            with self.assertRaisesRegex(self.app.UserFacingError, "Generate"):
                self.app.save_result({**payload, "result_id": "stale"}, "json")

    def test_http_contract_and_assets(self):
        server = self.app.ThreadingHTTPServer(("127.0.0.1", 0), self.app.RequestHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for route in ("/", "/app.js", "/styles.css", "/api/config",
                          "/assets/generating.gif", "/assets/cat-ok.png", "/assets/cat-vegetable.jpg"):
                with urlopen(base + route) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            request = Request(base + "/api/import", data=json.dumps(self.document()).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request) as response:
                self.assertEqual(json.load(response)["settings"], self.app.DEFAULTS)
            for raw in (b"{", b"[]", b'{"format":"double_shell_initial_state_v1"}'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + "/api/import", data=raw))
                self.assertEqual(error.exception.code, 400)
                self.assertFalse(json.loads(error.exception.read())["ok"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_launch_directory_is_detected_in_another_working_folder(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            code = ("import json,runpy,sys; "
                    "app=runpy.run_path(sys.argv[1],run_name='launch_check'); "
                    "print(json.dumps(str(app['DEFAULT_OUTPUT_DIR'])))")
            process = subprocess.run([sys.executable, "-c", code, str(ROOT / "Feodosyev html/app.py")],
                                     cwd=directory, capture_output=True, text=True, check=True)
            self.assertEqual(Path(json.loads(process.stdout)), Path(directory).resolve())

    def test_spinning_gif_preserves_original_frames_without_standing_intro(self):
        from PIL import Image
        import numpy as np
        with Image.open(ROOT / "cat meme/oiiaioiiiai.gif") as original, Image.open(
                ROOT / "Feodosyev html/assets/generating.gif") as animation:
            self.assertEqual(animation.n_frames, 29)
            duration = 0
            for index in range(animation.n_frames):
                original.seek(index + 17)
                animation.seek(index)
                a, b = np.array(original.convert("RGBA")), np.array(animation.convert("RGBA"))
                self.assertTrue(np.array_equal(a[:, :, 3], b[:, :, 3]))
                self.assertTrue(np.array_equal(a[a[:, :, 3] > 0], b[b[:, :, 3] > 0]))
                self.assertEqual(original.info["duration"], animation.info["duration"])
                duration += animation.info["duration"]
            self.assertEqual(duration, 1930)


if __name__ == "__main__":
    unittest.main()

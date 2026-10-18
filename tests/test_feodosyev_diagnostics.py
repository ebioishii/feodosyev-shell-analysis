"""Independent tangent checks and physical-state diagnostic acceptance."""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import numpy as np

from feodosyev_curve import solve_uniform_curve
from feodosyev_diagnostics import (
    MaterialReference, constitutive_tangent, diagnose_curve, export_diagnostics,
)
from feodosyev_single_point import (
    GrantaSS316TensionLaw, LinearElasticTensionLaw, UniaxialResponse,
    evaluate_shell, solve_single_point,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MATERIAL_DIR = PROJECT_DIR / "database" / "Granta_SS316_annealed_csv"
HAS_MATERIAL = all((MATERIAL_DIR / f).is_file() for f in (
    "Elasticity.csv", "Coefficient of Thermal Expansion.csv",
    "Zero-Thermal-Strain Reference Temperature.csv", "Isotropic Hardening.csv",
))


@dataclass(frozen=True)
class PerfectPlasticLaw:
    elastic_modulus_pa: float = 100e9
    yield_stress_pa: float = 100e6
    max_total_mechanical_strain: float = 0.1

    def evaluate(self, strain: float) -> UniaxialResponse:
        if not 0 <= strain <= self.max_total_mechanical_strain:
            raise ValueError("outside test material domain")
        stress = min(self.elastic_modulus_pa * strain, self.yield_stress_pa)
        elastic = stress / self.elastic_modulus_pa
        return UniaxialResponse(strain, stress, elastic, strain - elastic,
                                "plastic" if strain > elastic else "elastic")


class AnalyticDiagnosticsTests(unittest.TestCase):
    def make_curve(self):
        material = LinearElasticTensionLaw(200e9)
        curve = solve_uniform_curve(
            start_hoop_strain=0.0003, end_hoop_strain=0.002, point_count=9,
            radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
            inner_thermal_strain=0.0002, outer_thermal_strain=0.0002,
            inner_material=material, outer_material=material, k=0.35,
            root_scan_points=101,
        )
        return curve, MaterialReference(material, 1e9)

    def test_linear_pressure_slope_matches_closed_form_and_force_split(self):
        curve, reference = self.make_curve()
        result = diagnose_curve(curve, reference, reference)
        ratio = (0.35 - 0.5) / (1 - 0.5 * 0.35)
        expected = 4 / 3 * 200e9 * (1 + 0.5 * ratio) * 0.003 / 0.1
        for row in result.rows:
            for field in ("pressure_slope_implicit_pa_per_strain", "pressure_slope_grid_pa_per_strain"):
                self.assertAlmostEqual(row[field] / expected, 1.0, places=8)
            self.assertAlmostEqual(row["axial_strain_slope"], ratio, places=9)
            self.assertAlmostEqual(row["inner_total_pressure_slope_pa_per_strain"] / expected, 1 / 3, places=9)
            for name in ("inner", "outer"):
                self.assertAlmostEqual(
                    row[f"{name}_total_pressure_slope_pa_per_strain"],
                    row[f"{name}_magnitude_pressure_slope_pa_per_strain"] + row[f"{name}_direction_pressure_slope_pa_per_strain"],
                    delta=1e-5,
                )

    def test_plastic_direction_tangent_matches_independent_stress_difference(self):
        material = PerfectPlasticLaw()
        axial, hoop = -0.004, 0.002  # hoop stress is zero; axial compression still yields
        shell = evaluate_shell(axial, hoop, 0.0, material)
        jacobian, magnitude, direction, tangent, _ = constitutive_tangent(shell, material)
        self.assertAlmostEqual(shell.hoop_stress_pa, 0.0)
        self.assertEqual(shell.material_branch, "plastic")
        self.assertAlmostEqual(tangent, 0.0)
        self.assertGreater(abs(direction[1, 1]), 1e9)
        np.testing.assert_allclose(magnitude, 0, atol=1e-3)
        delta = 1e-8
        independent = np.empty((2, 2))
        for column, perturbation in enumerate(((delta, 0), (0, delta))):
            dx, dy = perturbation
            left = evaluate_shell(axial - dx, hoop - dy, 0.0, material)
            right = evaluate_shell(axial + dx, hoop + dy, 0.0, material)
            independent[:, column] = np.array([
                right.axial_stress_pa - left.axial_stress_pa,
                right.hoop_stress_pa - left.hoop_stress_pa,
            ]) / (2 * delta)
        np.testing.assert_allclose(jacobian, independent, rtol=1e-8, atol=1e-2)

    def test_export_and_invalid_yield_reference(self):
        curve, reference = self.make_curve()
        result = diagnose_curve(curve, reference, reference)
        with tempfile.TemporaryDirectory(prefix=".test-temp-", dir=PROJECT_DIR) as directory:
            paths = export_diagnostics(result, Path(directory), "diagnostic")
            self.assertTrue(paths[2].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("initial_yield_stress_pa", paths[1].read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                export_diagnostics(result, Path(directory), "diagnostic")
        with self.assertRaises(ValueError):
            MaterialReference(reference.material, 0)


@unittest.skipUnless(HAS_MATERIAL, "local licensed Granta material files unavailable")
class GrantaDiagnosticsTests(unittest.TestCase):
    def test_actual_sign_and_invariant_and_local_resolve_slope(self):
        from interpolation import thermal_strain
        inner, outer = GrantaSS316TensionLaw(538), GrantaSS316TensionLaw(20)
        arguments = dict(
            radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
            inner_thermal_strain=thermal_strain(538), outer_thermal_strain=thermal_strain(20),
            inner_material=inner, outer_material=outer, k=0.5, root_scan_points=201,
        )
        curve = solve_uniform_curve(start_hoop_strain=0.00025, end_hoop_strain=0.03, point_count=31, **arguments)
        result = diagnose_curve(curve, MaterialReference(inner, inner.yield_stress_pa, 538),
                                MaterialReference(outer, outer.yield_stress_pa, 20))
        self.assertEqual({r["inner_material_branch"] for r in result.rows}, {"plastic"})
        self.assertEqual({r["inner_hoop_stress_sign"] for r in result.rows}, {"compression", "tension"})
        self.assertGreater(result.summary["inner"]["minimum_stress_intensity_over_initial_yield"], 1)
        # Re-solve equilibrium locally instead of relying on plotted grid slopes.
        for index in (1, 7, 12, 19, 29):
            point = curve.points[index].solution
            delta = 1e-7
            left = solve_single_point(common_total_hoop_strain=point.common_total_hoop_strain - delta,
                                      axial_strain_guess=point.common_total_axial_strain, **arguments)
            right = solve_single_point(common_total_hoop_strain=point.common_total_hoop_strain + delta,
                                       axial_strain_guess=point.common_total_axial_strain, **arguments)
            slope = (right.gas_pressure_pa - left.gas_pressure_pa) / (2 * delta)
            self.assertAlmostEqual(result.rows[index]["pressure_slope_implicit_pa_per_strain"] / slope, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

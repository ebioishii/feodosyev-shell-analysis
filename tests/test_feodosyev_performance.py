"""Exact material-cache and globally certified axial-root acceptance tests."""

import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

import feodosyev_single_point as single
from feodosyev_curve import solve_uniform_curve

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = all((ROOT / "database/Granta_SS316_annealed_csv" / name).is_file() for name in (
    "Elasticity.csv", "Coefficient of Thermal Expansion.csv",
    "Zero-Thermal-Strain Reference Temperature.csv", "Isotropic Hardening.csv"))


class CertifiedRootTests(unittest.TestCase):
    def arguments(self, material, **changes):
        return dict(radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
                    common_total_hoop_strain=0.001, inner_thermal_strain=0,
                    outer_thermal_strain=0, inner_material=material, outer_material=material,
                    k=0.5, **changes)

    def test_proven_unique_root_avoids_full_scan(self):
        arguments = self.arguments(single.LinearElasticTensionLaw(100e9))
        with patch.object(single, "evaluate_shell", wraps=single.evaluate_shell) as evaluate:
            result = single.solve_single_point(**arguments, root_scan_points=801)
        self.assertEqual(result.root_count, 1)
        self.assertLess(evaluate.call_count, 50)

    def test_unknown_material_retains_full_scan(self):
        class CustomLaw:
            elastic_modulus_pa = 100e9
            max_total_mechanical_strain = 0.25
            def evaluate(self, strain):
                return single.LinearElasticTensionLaw(self.elastic_modulus_pa).evaluate(strain)
        with patch.object(single, "evaluate_shell", wraps=single.evaluate_shell) as evaluate:
            result = single.solve_single_point(**self.arguments(CustomLaw()), root_scan_points=201)
        self.assertEqual(result.root_count, 1)
        self.assertGreaterEqual(evaluate.call_count, 402)

    def test_zero_tangent_and_multiple_roots_are_not_certified(self):
        arguments = self.arguments(single.LinearElasticTensionLaw(100e9))
        arguments.update(k=2, common_total_hoop_strain=0)
        with self.assertRaisesRegex(ValueError, "multiple"):
            single.solve_single_point(**arguments, root_scan_points=41)
        result = single.solve_single_point(**arguments, root_scan_points=41, axial_strain_guess=0)
        self.assertGreater(result.root_count, 1)
        self.assertEqual(result.root_selection, "root nearest the supplied axial_strain_guess")

    def test_fast_path_still_validates_scan_count(self):
        for count in (True, 2, 3.5, float("inf")):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "integer at least 3"):
                single.solve_single_point(**self.arguments(single.LinearElasticTensionLaw(100e9)),
                                          root_scan_points=count)


@unittest.skipUnless(HAS_DATA, "local licensed Granta SS316 CSV files are not available")
class GrantaCacheTests(unittest.TestCase):
    @staticmethod
    def legacy_response(material, strain):
        """Independent reference: the previous brentq inversion, same tolerances."""
        from interpolation import flow_stress
        if strain <= material.yield_total_mechanical_strain:
            return material.elastic_modulus_pa * strain, 0.0, "elastic"
        tolerance = max(single.ZERO_STRAIN_TOLERANCE, 1e-12 * material.max_total_mechanical_strain)
        if math.isclose(strain, material.max_total_mechanical_strain, rel_tol=0, abs_tol=tolerance):
            plastic = material.max_plastic_strain
        else:
            plastic = brentq(lambda p: p + flow_stress(material.temperature_c, p)/material.elastic_modulus_pa-strain,
                             0, material.max_plastic_strain, xtol=1e-14, rtol=1e-14)
        return flow_stress(material.temperature_c, plastic), plastic, "plastic"

    def test_cached_curve_matches_previous_inverse_at_nodes_and_between(self):
        from interpolation import T_hard, T_E, T_alpha
        lo, hi = max(min(T_hard), min(T_E), min(T_alpha)), min(max(T_hard), max(T_E), max(T_alpha))
        temperatures = sorted({20., 100., 400., 538., 650.,
            *(float(t) for t in T_hard if lo <= t <= hi),
            *(float((a+b)/2) for a, b in zip(T_hard[:-1], T_hard[1:]) if lo <= (a+b)/2 <= hi)})
        for temperature in temperatures:
            material = single.GrantaSS316TensionLaw(temperature)
            nodes = material._total_nodes
            strains = np.unique(np.concatenate(([0., material.yield_total_mechanical_strain], nodes,
                (nodes[:-1]+nodes[1:])/2, np.linspace(0, material.max_total_mechanical_strain, 31))))
            for strain in strains:
                with self.subTest(temperature=temperature, strain=strain):
                    actual = material.evaluate(float(strain))
                    stress, plastic, branch = self.legacy_response(material, float(strain))
                    self.assertAlmostEqual(actual.stress_pa, stress, delta=0.02)
                    self.assertAlmostEqual(actual.plastic_strain, plastic, delta=2e-12)
                    self.assertEqual(actual.branch, branch)

    def test_combines_different_temperature_grids_without_approximating(self):
        import interpolation
        synthetic = {0.: (np.array([0., .01, .04]), np.array([2e8, 3e8, 4e8])),
                     100.: (np.array([0., .02, .03]), np.array([1e8, 2e8, 3e8]))}
        single._granta_curve_at_temperature.cache_clear()
        try:
            with patch.multiple(interpolation, hardening=synthetic, T_hard=np.array([0., 100.]),
                                E_at=lambda t: 100e9):
                material = single.GrantaSS316TensionLaw(50)
                self.assertEqual(list(material._plastic_nodes), [0, .01, .02, .03])
                for strain in np.linspace(0, material.max_total_mechanical_strain, 101):
                    stress, plastic, branch = self.legacy_response(material, strain)
                    actual = material.evaluate(strain)
                    self.assertAlmostEqual(actual.stress_pa, stress, delta=0.02)
                    self.assertAlmostEqual(actual.plastic_strain, plastic, delta=2e-12)
                    self.assertEqual(actual.branch, branch)
        finally:
            single._granta_curve_at_temperature.cache_clear()

    def test_repeated_materials_and_evaluations_do_not_repeat_interpolation(self):
        import interpolation
        single._granta_curve_at_temperature.cache_clear()
        with patch.object(interpolation, "flow_stress", wraps=interpolation.flow_stress) as flow:
            first = single.GrantaSS316TensionLaw(538)
            setup_count = flow.call_count
            second = single.GrantaSS316TensionLaw(538)
            for strain in np.linspace(0, first.max_total_mechanical_strain, 501):
                second.evaluate(strain)
            self.assertGreater(setup_count, 0)
            self.assertEqual(flow.call_count, setup_count)
        with self.assertRaises(ValueError):
            first.evaluate(first.max_total_mechanical_strain + 1e-5)

    def test_analytic_derivative_bound_covers_material_corners_and_extrema(self):
        for temperature in (20., 538.):
            material = single.GrantaSS316TensionLaw(temperature)
            for hoop in (0., .00025, .005, .03):
                from interpolation import thermal_strain
                thermal = thermal_strain(temperature)
                lower, upper = single._axial_total_strain_domain(hoop, thermal, material)
                for k in (-1., 0., .35, .5, 1.9, 2., 2.5):
                    bound = single._certified_axial_derivative_lower_bound(material, lower, upper, hoop, thermal, k)
                    self.assertIsNotNone(bound)
                    step = (upper-lower)*1e-7
                    for x in np.linspace(lower+2*step, upper-2*step, 51):
                        def residual(axial):
                            shell = single.evaluate_shell(axial, hoop, thermal, material)
                            return shell.axial_stress_pa-k*shell.hoop_stress_pa
                        derivative = (residual(x+step)-residual(x-step))/(2*step)
                        self.assertLessEqual(bound, derivative + 1e-4*material.elastic_modulus_pa)

    def test_curves_match_full_scan_pressure_stresses_branches_and_residuals(self):
        from interpolation import thermal_strain
        for k, hi, ho, start in ((.5, .001, .002, .00025), (.35, .001, .002, .00025),
                                 (.5, .0015, .0015, .005)):
            arguments = dict(start_hoop_strain=start, end_hoop_strain=.03, point_count=41,
                radius_m=.1, inner_thickness_m=hi, outer_thickness_m=ho, k=k,
                inner_thermal_strain=thermal_strain(538), outer_thermal_strain=thermal_strain(20),
                inner_material=single.GrantaSS316TensionLaw(538), outer_material=single.GrantaSS316TensionLaw(20))
            actual = solve_uniform_curve(**arguments)
            with patch.object(single, "_certified_axial_derivative_lower_bound", return_value=None):
                expected = solve_uniform_curve(**arguments)
            for a, b in zip(actual.points, expected.points):
                a, b = a.solution, b.solution
                self.assertEqual(a.root_count, b.root_count)
                self.assertAlmostEqual(a.gas_pressure_pa, b.gas_pressure_pa, delta=.001)
                self.assertAlmostEqual(a.common_total_axial_strain, b.common_total_axial_strain, delta=1e-12)
                for sa, sb in ((a.inner, b.inner), (a.outer, b.outer)):
                    self.assertEqual(sa.material_branch, sb.material_branch)
                    self.assertAlmostEqual(sa.axial_stress_pa, sb.axial_stress_pa, delta=.02)
                    self.assertAlmostEqual(sa.hoop_stress_pa, sb.hoop_stress_pa, delta=.02)
                for residual in (a.residuals, b.residuals):
                    self.assertLessEqual(max(abs(residual.equation_9_1_n_per_m),
                        abs(residual.equation_9_2_n_per_m), abs(residual.equation_9_5_n_per_m)),
                        residual.tolerance_n_per_m)


if __name__ == "__main__":
    unittest.main()

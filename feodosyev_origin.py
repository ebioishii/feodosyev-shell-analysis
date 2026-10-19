"""Zero-pressure starting point for the Feodosyev HTML interface.

Moved unchanged from the temperature study; uses the existing point solver.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from feodosyev_curve import CurvePoint, _csv_row
from feodosyev_single_point import SinglePointResult, solve_single_point


class _PointEvaluator:
    def __init__(self, arguments: dict[str, object]):
        self.arguments = arguments
        self.cache: dict[float, SinglePointResult] = {}
        self.trace: list[dict[str, object]] = []

    def at(self, strain: float, purpose: str, *, signed: bool = False,
           guess: float | None = None, fresh: bool = False) -> SinglePointResult:
        strain = float(strain)
        cached = strain in self.cache and not fresh
        try:
            point = self.cache[strain] if cached else solve_single_point(
                common_total_hoop_strain=strain, axial_strain_guess=guess,
                allow_negative_pressure=signed, **self.arguments,
            )
            row = _csv_row(CurvePoint(len(self.trace), point, guess, None, None))
            row.update(purpose=purpose, cached=cached, signed_search=signed, error="")
            self.trace.append(row)
            if point.root_count != 1:
                raise ValueError("this study requires a unique detected axial root; branch ambiguity remains")
            if not signed and point.gas_pressure_pa < -point.residuals.tolerance_n_per_m / point.radius_m:
                raise ValueError("negative pressure outside the signed zero-pressure search")
            self.cache[strain] = point
            return point
        except (ValueError, ArithmeticError) as exc:
            self.trace.append({"purpose": purpose, "total_hoop_strain": strain,
                               "signed_search": signed, "cached": cached, "error": str(exc)})
            raise


def _zero_pressure_origin(evaluator: _PointEvaluator, inner_thermal: float,
                          outer_thermal: float, scan_points: int) -> dict[str, object]:
    """Bracket on the free-thermal-strain interval, then enclose the p=0 root.

    Failure anywhere on this natural bracket is reported, not skipped. No
    pressure clipping or search for a visually convenient starting point occurs.
    """
    lower, upper = sorted((inner_thermal, outer_thermal))
    nodes = np.linspace(lower, upper, scan_points) if upper > lower else [lower]
    scanned = [evaluator.at(float(x), "origin_scan", signed=True) for x in nodes]
    exact = [p for p in scanned if p.gas_pressure_pa == 0.0]
    brackets = [(a, b) for a, b in zip(scanned[:-1], scanned[1:])
                if a.gas_pressure_pa < 0 < b.gas_pressure_pa]
    downward = any(a.gas_pressure_pa > 0 > b.gas_pressure_pa
                   for a, b in zip(scanned[:-1], scanned[1:]))
    if len(exact) + len(brackets) != 1 or downward:
        raise ValueError("no unique negative-to-positive p=0 crossing on the free-thermal-strain bracket")
    if exact:
        zero = exact[0]
        left = right = zero.common_total_hoop_strain
    else:
        a, b = brackets[0]
        left, right = a.common_total_hoop_strain, b.common_total_hoop_strain
        root = brentq(lambda x: evaluator.at(x, "origin_brentq", signed=True).gas_pressure_pa,
                      left, right, xtol=1e-14, rtol=1e-14)
        root_point = evaluator.at(root, "origin_root_check", signed=True)
        if root_point.gas_pressure_pa == 0:
            left = right = root
        else:
            # Keep the enclosing signed values, including roundoff near zero.
            negatives = [p.common_total_hoop_strain for p in evaluator.cache.values()
                         if left <= p.common_total_hoop_strain <= right and p.gas_pressure_pa < 0]
            positives = [p.common_total_hoop_strain for p in evaluator.cache.values()
                         if left <= p.common_total_hoop_strain <= right and p.gas_pressure_pa >= 0]
            left, right = max(negatives), min(positives)
            for _ in range(80):
                positive = evaluator.at(right, "origin_positive_bound", signed=True)
                tolerance = positive.residuals.tolerance_n_per_m / positive.radius_m
                if right - left <= 1e-12 and positive.gas_pressure_pa <= tolerance:
                    break
                midpoint = 0.5 * (left + right)
                if midpoint in (left, right):
                    raise ArithmeticError("p=0 enclosure cannot be tightened further")
                trial = evaluator.at(midpoint, "origin_enclosure", signed=True)
                if trial.gas_pressure_pa < 0:
                    left = midpoint
                else:
                    right = midpoint
            else:
                raise ArithmeticError("p=0 enclosure did not converge")
        zero = evaluator.at(right, "origin_physical_check")
    tolerance = zero.residuals.tolerance_n_per_m / zero.radius_m
    if not 0 <= zero.gas_pressure_pa <= tolerance:
        raise ArithmeticError("zero-pressure starting point failed its signed pressure tolerance")
    return {
        "method": "free-thermal-strain bracket scan; brentq; positive-side numerical root enclosure",
        "search_lower_hoop_strain": lower, "search_upper_hoop_strain": upper,
        "root_lower_hoop_strain": left, "root_upper_hoop_strain": right,
        "root_enclosure_width": right - left,
        "zero_pressure_hoop_strain": zero.common_total_hoop_strain,
        "actual_pressure_pa": zero.gas_pressure_pa, "pressure_tolerance_pa": tolerance,
        "pressure_clipped": False,
    }

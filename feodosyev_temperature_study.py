"""Stage 4A: fixed-temperature SS316 parameter study using unchanged solvers.

Only inner temperature varies. Signed trials used to locate p=0 are retained;
successful curves begin at a verified non-negative side of that root enclosure.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from feodosyev_curve import CurvePoint, CurveResult, _csv_row, solve_uniform_curve
from feodosyev_diagnostics import DiagnosticResult, MaterialReference, diagnose_curve
from feodosyev_single_point import SinglePointResult, solve_single_point


BASELINE_COMMIT = "d5233127576ad20fb35fc249a6c4ce5e8e66a4a9"
DEFAULT_TEMPERATURES = (300.0, 400.0, 450.0, 500.0, 538.0, 600.0, 650.0)


@dataclass(frozen=True)
class StudySettings:
    radius_m: float = 0.1
    inner_thickness_m: float = 0.001
    outer_thickness_m: float = 0.002
    outer_temperature_c: float = 20.0
    k: float = 0.5
    end_hoop_strain: float = 0.03
    point_count: int = 481
    root_scan_points: int = 401
    origin_scan_points: int = 17
    reverse_check_points: int = 17

    def __post_init__(self) -> None:
        for name in ("radius_m", "inner_thickness_m", "outer_thickness_m", "end_hoop_strain"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not all(math.isfinite(x) for x in (self.k, self.outer_temperature_c)):
            raise ValueError("k and outer temperature must be finite")
        for name in ("point_count", "root_scan_points", "origin_scan_points", "reverse_check_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 3:
                raise ValueError(f"{name} must be an integer >= 3")


@dataclass(frozen=True)
class TemperatureCase:
    temperature_c: float
    status: str
    summary: dict[str, object]
    curve: CurveResult | None
    diagnostics: DiagnosticResult | None
    evaluations: tuple[dict[str, object], ...]


class CaseComputationError(ValueError):
    def __init__(self, message: str, evaluations: Sequence[dict[str, object]] = ()):
        super().__init__(message)
        self.evaluations = tuple(evaluations)


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


def _refine_events(curve: CurveResult, evaluator: _PointEvaluator, reference: MaterialReference,
                   shell_name: str, quantity: str,
                   additional_points: Sequence[SinglePointResult] = ()) -> dict[str, object]:
    if quantity == "yield":
        value = lambda point: getattr(point, shell_name).strain_intensity / reference.initial_yield_strain - 1.0
    else:
        value = lambda point: getattr(point, shell_name).hoop_stress_pa
    by_strain = {p.solution.common_total_hoop_strain: p.solution for p in curve.points}
    by_strain.update({p.common_total_hoop_strain: p for p in additional_points})
    points = [by_strain[x] for x in sorted(by_strain)]
    first_value = value(points[0])
    crossings = []
    for a, b in zip(points[:-1], points[1:]):
        fa, fb = value(a), value(b)
        if fa * fb < 0 or fa == 0 or fb == 0:
            if fa == fb == 0:
                continue
            x0, x1 = a.common_total_hoop_strain, b.common_total_hoop_strain
            root = (x0 if fa == 0 else x1 if fb == 0 else brentq(
                lambda x: value(evaluator.at(x, f"{shell_name}_{quantity}_refine")),
                x0, x1, xtol=1e-13, rtol=1e-13,
            ))
            if crossings and abs(root - float(crossings[-1]["hoop_strain"])) < 1e-12:
                continue
            point = evaluator.at(root, f"{shell_name}_{quantity}_verification")
            condition_residual = value(point)
            condition_tolerance = 1e-7 if quantity == "yield" else 1.0
            if abs(condition_residual) > condition_tolerance:
                raise ArithmeticError(f"{shell_name} {quantity} root failed its condition-residual check")
            crossings.append({"hoop_strain": root, "pressure_pa": point.gas_pressure_pa,
                              "bracket_lower": x0, "bracket_upper": x1,
                              "direction": "rising" if fb > fa else "falling",
                              "condition_residual": condition_residual})
    rising = next((c for c in crossings if c["direction"] == "rising"), None)
    already = first_value >= 0
    return {"status": "already_at_or_above_at_start" if already else "crossing_found" if rising else "not_reached",
            "first_rising": None if already else rising, "crossings": crossings,
            "start_condition_value": first_value}


def _refine_inner_minimum(curve: CurveResult, evaluator: _PointEvaluator,
                          inner: MaterialReference) -> dict[str, object]:
    points = [p.solution for p in curve.points]
    ratio = lambda p: p.inner.stress_intensity_pa / inner.initial_yield_stress_pa
    values = [ratio(p) for p in points]
    sampled = points[int(np.argmin(values))]
    candidates = [sampled, points[0], points[-1]]
    refined_brackets = []
    intervals = []
    if values[0] < values[1]:
        intervals.append((points[0].common_total_hoop_strain, points[1].common_total_hoop_strain))
    if values[-1] < values[-2]:
        intervals.append((points[-2].common_total_hoop_strain, points[-1].common_total_hoop_strain))
    for index in range(1, len(points) - 1):
        if values[index] <= min(values[index - 1], values[index + 1]) and values[index] < max(values[index - 1], values[index + 1]):
            intervals.append((points[index - 1].common_total_hoop_strain, points[index + 1].common_total_hoop_strain))
    for left, right in intervals:
        optimum = minimize_scalar(
            lambda x: ratio(evaluator.at(float(x), "inner_minimum_refine")),
            bounds=(left, right), method="bounded", options={"xatol": 1e-11, "maxiter": 100},
        )
        if not optimum.success:
            raise ArithmeticError("bounded inner-intensity minimum refinement failed")
        candidates.append(evaluator.at(float(optimum.x), "inner_minimum_verification"))
        refined_brackets.append({"lower": left, "upper": right})
    best = min(candidates, key=ratio)
    return {"method": "sample minimum; bounded refinement of resolved local minima and endpoint-adjacent candidate intervals",
            "ratio": ratio(best), "hoop_strain": best.common_total_hoop_strain,
            "pressure_pa": best.gas_pressure_pa,
            "sampled_ratio": ratio(sampled), "sampled_hoop_strain": sampled.common_total_hoop_strain,
            "refined_brackets": refined_brackets,
            "below_initial_yield": ratio(best) < 1.0 - 1e-8,
            "material_branch": best.inner.material_branch}


def solve_material_case(
    temperature_c: float, settings: StudySettings, inner: MaterialReference,
    outer: MaterialReference, inner_thermal_strain: float, outer_thermal_strain: float,
) -> TemperatureCase:
    """Compute one case, raising a contextual error with all trial evidence."""
    arguments = dict(radius_m=settings.radius_m, inner_thickness_m=settings.inner_thickness_m,
                     outer_thickness_m=settings.outer_thickness_m, k=settings.k,
                     inner_material=inner.material, outer_material=outer.material,
                     inner_thermal_strain=inner_thermal_strain, outer_thermal_strain=outer_thermal_strain,
                     root_scan_points=settings.root_scan_points)
    evaluator = _PointEvaluator(arguments)
    phase = "zero-pressure origin"
    try:
        origin = _zero_pressure_origin(evaluator, inner_thermal_strain, outer_thermal_strain, settings.origin_scan_points)
        start = float(origin["zero_pressure_hoop_strain"])
        if start >= settings.end_hoop_strain:
            raise ValueError("zero-pressure origin is not below the prescribed scan endpoint")
        phase = "non-negative pressure curve"
        curve = solve_uniform_curve(start_hoop_strain=start, end_hoop_strain=settings.end_hoop_strain,
                                    point_count=settings.point_count, **arguments)
        if any(p.solution.root_count != 1 for p in curve.points):
            raise ValueError("multiple detected axial roots on the forward curve")
        if any(p.solution.gas_pressure_pa < 0 for p in curve.points):
            raise ValueError("a negative forward pressure was returned; no point was removed or clipped")
        for point in curve.points:
            evaluator.cache[point.solution.common_total_hoop_strain] = point.solution
        phase = "diagnostics and branch audit"
        diagnostics = diagnose_curve(curve, inner, outer)
        indices = np.unique(np.linspace(0, len(curve.points) - 1,
                                       min(settings.reverse_check_points, len(curve.points)), dtype=int))[::-1]
        reverse_differences = []
        previous = curve.points[-1].solution.common_total_axial_strain
        for index in indices:
            forward = curve.points[int(index)].solution
            reverse = evaluator.at(forward.common_total_hoop_strain, "reverse_branch_audit",
                                   guess=previous, fresh=True)
            reverse_differences.append(abs(reverse.common_total_axial_strain - forward.common_total_axial_strain))
            previous = reverse.common_total_axial_strain
        if max(reverse_differences) > 1e-10:
            raise ArithmeticError("forward and reverse axial-root selections disagree")
        phase = "physical event and minimum refinement"
        outer_yield = _refine_events(curve, evaluator, outer, "outer", "yield")
        inner_zero = _refine_events(curve, evaluator, inner, "inner", "hoop_zero")
        minimum = _refine_inner_minimum(curve, evaluator, inner)
        minimum_point = evaluator.at(float(minimum["hoop_strain"]), "inner_minimum_event_anchor")
        inner_yield = _refine_events(curve, evaluator, inner, "inner", "yield", (minimum_point,))
        bounds = sorted({start, settings.end_hoop_strain,
                         *(float(c["hoop_strain"]) for c in inner_yield["crossings"])})
        below_intervals = []
        for left, right in zip(bounds[:-1], bounds[1:]):
            middle = evaluator.at(0.5 * (left + right), "inner_below_yield_interval_check")
            if middle.inner.stress_intensity_pa < inner.initial_yield_stress_pa * (1 - 1e-8):
                below_intervals.append({"lower_hoop_strain": left, "upper_hoop_strain": right})
        if bool(below_intervals) != bool(minimum["below_initial_yield"]):
            raise ArithmeticError("minimum and resolved below-yield intervals disagree; refine the strain grid")
        slopes = [float(row["pressure_slope_implicit_pa_per_strain"]) for row in diagnostics.rows]
        peak_index = int(np.argmax(slopes))
        peak_row = diagnostics.rows[peak_index]
        summary = {
            "inner_temperature_c": temperature_c, "status": "complete", "error": "",
            "settings": asdict(settings), "zero_pressure_origin": origin,
            "inner_yield_stress_pa": inner.initial_yield_stress_pa,
            "outer_yield_stress_pa": outer.initial_yield_stress_pa,
            "inner_thermal_strain": inner_thermal_strain, "outer_thermal_strain": outer_thermal_strain,
            "outer_initial_yield": outer_yield, "inner_hoop_zero": inner_zero,
            "inner_initial_yield_crossings": inner_yield, "inner_minimum": minimum,
            "inner_below_initial_yield_intervals": below_intervals,
            "curve_diagnostics": asdict(curve.diagnostics),
            "reverse_audit_point_count": len(indices),
            "maximum_reverse_axial_difference": max(reverse_differences),
            "branch_check": "unique detected roots at all forward, reverse, and refined points; finite implicit tangents",
            "maximum_domain_utilization": {name: max(float(r[f"{name}_domain_utilization"]) for r in diagnostics.rows)
                                           for name in ("inner", "outer")},
            "local_slope": {"start_pa_per_strain": slopes[0], "end_pa_per_strain": slopes[-1],
                            "sampled_minimum_pa_per_strain": min(slopes), "sampled_maximum_pa_per_strain": max(slopes),
                            "hoop_strain_at_sampled_maximum": peak_row["total_hoop_strain"]},
            "stage_classification_performed": False, "endpoint_is_load_limit": False,
        }
        target = inner_zero["first_rising"]
        if target is not None:
            nearest = min(diagnostics.rows, key=lambda r: abs(float(r["total_hoop_strain"]) - float(target["hoop_strain"])))
            fields = ("total_hoop_strain", "gas_pressure_pa", "inner_stress_intensity_over_initial_yield",
                      "pressure_slope_implicit_pa_per_strain", "inner_total_pressure_slope_pa_per_strain",
                      "outer_total_pressure_slope_pa_per_strain", "inner_magnitude_pressure_slope_pa_per_strain",
                      "inner_direction_pressure_slope_pa_per_strain")
            summary["load_sharing_nearest_inner_hoop_zero"] = {field: nearest[field] for field in fields}
        return TemperatureCase(temperature_c, "complete", summary, curve, diagnostics, tuple(evaluator.trace))
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        raise CaseComputationError(f"Inner {temperature_c:g} C; {phase}: {exc}", evaluator.trace) from exc


def temperature_domains() -> dict[str, list[float]]:
    from interpolation import T_E, T_alpha, T_hard
    return {name: [float(min(values)), float(max(values))]
            for name, values in (("elasticity", T_E), ("thermal_expansion", T_alpha), ("hardening", T_hard))}


def run_ss316_case(temperature_c: float, settings: StudySettings) -> TemperatureCase:
    """Keep an explicit incomplete result for a failed case, including its trials."""
    domains: dict[str, list[float]] = {}
    try:
        from feodosyev_single_point import GrantaSS316TensionLaw
        from interpolation import thermal_strain
        domains = temperature_domains()
        for label, temperature in (("Inner", temperature_c), ("Outer", settings.outer_temperature_c)):
            for property_name, (lower, upper) in domains.items():
                if not math.isfinite(temperature) or not lower <= temperature <= upper:
                    raise ValueError(f"{label} {temperature:g} C outside {property_name} temperature domain [{lower:g}, {upper:g}] C; no extrapolation")
        inner = GrantaSS316TensionLaw(temperature_c)
        outer = GrantaSS316TensionLaw(settings.outer_temperature_c)
        result = solve_material_case(
            temperature_c, settings,
            MaterialReference(inner, inner.yield_stress_pa, temperature_c),
            MaterialReference(outer, outer.yield_stress_pa, settings.outer_temperature_c),
            thermal_strain(temperature_c), thermal_strain(settings.outer_temperature_c),
        )
        result.summary["material_temperature_domains_c"] = domains
        return result
    except (ValueError, ArithmeticError, FileNotFoundError) as exc:
        return TemperatureCase(temperature_c, "incomplete",
                               {"inner_temperature_c": temperature_c, "status": "incomplete", "error": str(exc),
                                "settings": asdict(settings), "material_temperature_domains_c": domains},
                               None, None, tuple(getattr(exc, "evaluations", ())))


def run_temperature_study(temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
                          settings: StudySettings = StudySettings(), *, workers: int = 1,
                          progress: Callable[[TemperatureCase], None] | None = None) -> tuple[TemperatureCase, ...]:
    values = tuple(float(t) for t in temperatures)
    if not values or len(set(values)) != len(values) or not all(math.isfinite(t) for t in values):
        raise ValueError("temperatures must be finite, non-empty, and unique")
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    results = {}
    if workers == 1:
        for temperature in values:
            result = run_ss316_case(temperature, settings)
            results[temperature] = result
            if progress:
                progress(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_ss316_case, t, settings): t for t in values}
            for future in as_completed(futures):
                result = future.result()
                results[result.temperature_c] = result
                if progress:
                    progress(result)
    return tuple(results[t] for t in values)


def _summary_row(case: TemperatureCase) -> dict[str, object]:
    row: dict[str, object] = {"inner_temperature_c": case.temperature_c, "status": case.status,
                             "error": case.summary["error"]}
    fields = ("zero_pressure_hoop_strain", "origin_actual_pressure_pa", "outer_yield_hoop_strain",
              "outer_yield_pressure_pa", "outer_yield_status", "inner_hoop_zero_strain", "inner_hoop_zero_pressure_pa",
              "inner_hoop_zero_status", "inner_minimum_ratio", "inner_minimum_hoop_strain", "inner_minimum_pressure_pa",
              "inner_below_initial_yield", "inner_sampled_minimum_ratio", "pressure_at_scan_endpoint_pa",
              "slope_start_pa_per_strain", "slope_min_pa_per_strain", "slope_max_pa_per_strain", "slope_end_pa_per_strain",
              "maximum_equilibrium_residual_n_per_m", "maximum_relative_equilibrium_residual",
              "maximum_reverse_axial_difference", "maximum_inner_domain_utilization", "maximum_outer_domain_utilization")
    row.update({field: None for field in fields})
    if case.status == "complete":
        s = case.summary
        outer = s["outer_initial_yield"]["first_rising"] or {}
        inner = s["inner_hoop_zero"]["first_rising"] or {}
        row.update({
            "zero_pressure_hoop_strain": s["zero_pressure_origin"]["zero_pressure_hoop_strain"],
            "origin_actual_pressure_pa": s["zero_pressure_origin"]["actual_pressure_pa"],
            "outer_yield_hoop_strain": outer.get("hoop_strain"), "outer_yield_pressure_pa": outer.get("pressure_pa"),
            "outer_yield_status": s["outer_initial_yield"]["status"],
            "inner_hoop_zero_strain": inner.get("hoop_strain"), "inner_hoop_zero_pressure_pa": inner.get("pressure_pa"),
            "inner_hoop_zero_status": s["inner_hoop_zero"]["status"],
            "inner_minimum_ratio": s["inner_minimum"]["ratio"],
            "inner_minimum_hoop_strain": s["inner_minimum"]["hoop_strain"],
            "inner_minimum_pressure_pa": s["inner_minimum"]["pressure_pa"],
            "inner_below_initial_yield": s["inner_minimum"]["below_initial_yield"],
            "inner_sampled_minimum_ratio": s["inner_minimum"]["sampled_ratio"],
            "pressure_at_scan_endpoint_pa": case.curve.points[-1].solution.gas_pressure_pa,
            "slope_start_pa_per_strain": s["local_slope"]["start_pa_per_strain"],
            "slope_min_pa_per_strain": s["local_slope"]["sampled_minimum_pa_per_strain"],
            "slope_max_pa_per_strain": s["local_slope"]["sampled_maximum_pa_per_strain"],
            "slope_end_pa_per_strain": s["local_slope"]["end_pa_per_strain"],
            "maximum_equilibrium_residual_n_per_m": s["curve_diagnostics"]["maximum_absolute_equilibrium_residual_n_per_m"],
            "maximum_relative_equilibrium_residual": s["curve_diagnostics"]["maximum_relative_equilibrium_residual"],
            "maximum_reverse_axial_difference": s["maximum_reverse_axial_difference"],
            "maximum_inner_domain_utilization": s["maximum_domain_utilization"]["inner"],
            "maximum_outer_domain_utilization": s["maximum_domain_utilization"]["outer"],
        })
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_plots(cases: Sequence[TemperatureCase], directory: Path) -> tuple[Path, ...]:
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    specs = (
        ("pressure_comparison.png", "gas_pressure_mpa", "Gas pressure (MPa)", "Pressure vs common total hoop strain", None),
        ("inner_yield_ratio_comparison.png", "inner_stress_intensity_over_initial_yield", r"$\sigma_{eq,inner}/\sigma_{y0,inner}(T)$", "Inner stress intensity relative to initial yield", 1.0),
        ("inner_hoop_stress_comparison.png", "inner_hoop_stress_mpa", "Inner hoop stress (MPa)", "Inner hoop compression-to-tension transition", 0.0),
    )
    paths = []
    for filename, field, ylabel, title, reference in specs:
        figure = Figure(figsize=(10, 6), dpi=180)
        try:
            axes = figure.subplots()
            for index, case in enumerate(cases):
                if case.status != "complete":
                    continue
                rows = case.diagnostics.rows
                color = colormaps["tab10"](index % 10)
                axes.plot([float(r["total_hoop_strain_percent"]) for r in rows],
                          [float(r[field]) for r in rows], color=color, label=f"{case.temperature_c:g} C")
                if reference == 1.0:
                    minimum = case.summary["inner_minimum"]
                    axes.scatter([float(minimum["hoop_strain"]) * 100], [minimum["ratio"]], color=color, s=14, zorder=3)
            if reference is not None:
                axes.axhline(reference, color="0.25", ls="--", lw=1, label=f"Reference = {reference:g}")
            incomplete = [f"{c.temperature_c:g} C" for c in cases if c.status != "complete"]
            if incomplete:
                axes.text(0.02, 0.98, "Incomplete: " + ", ".join(incomplete), transform=axes.transAxes, va="top", color="darkred")
            axes.set(xlabel="Common total hoop strain (%)", ylabel=ylabel, title=title + "\nFixed operating-temperature cases; prescribed endpoint is not a limit")
            axes.grid(alpha=0.3)
            if any(c.status == "complete" for c in cases) or reference is not None:
                axes.legend(ncol=2, fontsize=9)
            figure.tight_layout()
            path = directory / filename
            with path.open("xb"):
                pass
            figure.savefig(path, format="png")
            paths.append(path)
        finally:
            figure.clear()
    return tuple(paths)


def export_temperature_study(cases: Sequence[TemperatureCase], settings: StudySettings, directory: Path) -> dict[str, object]:
    """Use a new run directory; incomplete cases and signed trials remain visible."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write_csv(directory / "temperature_summary.csv", [_summary_row(c) for c in cases])
    for case in cases:
        case_dir = directory / f"inner_{case.temperature_c:g}C"
        case_dir.mkdir()
        if case.diagnostics is not None:
            _write_csv(case_dir / "points.csv", case.diagnostics.rows)
        if case.evaluations:
            _write_csv(case_dir / "evaluations.csv", case.evaluations)
        with (case_dir / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(case.summary, stream, indent=2, allow_nan=False)
    plots = _write_comparison_plots(cases, directory)
    root = Path(__file__).resolve().parent
    frozen = ("feodosyev_single_point.py", "feodosyev_curve.py", "feodosyev_diagnostics.py", "interpolation.py")
    manifest = {
        "baseline_commit": BASELINE_COMMIT, "settings": asdict(settings),
        "case_status": {str(c.temperature_c): c.status for c in cases},
        "all_cases_complete": all(c.status == "complete" for c in cases),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in frozen},
        "scope": "Inner-temperature combined thermal-strain and material-law sensitivity; fixed-temperature states, not a heating history",
        "endpoint_is_load_limit": False, "stage_classification_performed": False,
        "plots": [str(p.resolve()) for p in plots],
    }
    with (directory / "study_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--points", type=int, default=481)
    parser.add_argument("--root-scan-points", type=int, default=401)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        parser.error("output directory already exists; choose a new run directory")
    settings = StudySettings(point_count=args.points, root_scan_points=args.root_scan_points)
    def report(case: TemperatureCase) -> None:
        print(f"Inner {case.temperature_c:g} C: {case.status}; {case.summary['error']}", flush=True)
    cases = run_temperature_study(args.temperatures, settings, workers=args.workers, progress=report)
    manifest = export_temperature_study(cases, settings, args.output_directory)
    print(json.dumps(manifest, indent=2, allow_nan=False), flush=True)
    if not manifest["all_cases_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

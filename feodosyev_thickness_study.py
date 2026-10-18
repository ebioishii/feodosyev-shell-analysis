"""Stage 4B-1: thickness allocation at fixed total thickness and temperatures.

The established stage-4A case calculation is reused without changing its
solvers. Signed membrane resultants and equal-pressure states are added here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq

from feodosyev_curve import CurvePoint, _csv_row
from feodosyev_diagnostics import MaterialReference
from feodosyev_single_point import SinglePointResult
from feodosyev_temperature_study import (
    StudySettings, TemperatureCase, _PointEvaluator, _summary_row, _write_csv,
    solve_material_case, temperature_domains,
)


BASELINE_COMMIT = "dd670d5765d1a9b76aab57e4dd98b3deced0119a"
DEFAULT_RATIOS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
INNER_TEMPERATURE_C = 538.0
OUTER_TEMPERATURE_C = 20.0


def thicknesses_m(eta: float, total_m: float = 0.003) -> tuple[float, float]:
    """eta=h_outer/h_inner, keeping the total wall thickness fixed (SI)."""
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if not math.isfinite(total_m) or total_m <= 0:
        raise ValueError("total thickness must be finite and positive")
    inner, outer = total_m / (1 + eta), total_m * eta / (1 + eta)
    if not math.isclose(inner + outer, total_m, rel_tol=1e-14, abs_tol=1e-15):
        raise ArithmeticError("total thickness conversion failed")
    return inner, outer


@dataclass(frozen=True)
class ThicknessSettings:
    total_thickness_m: float = 0.003
    radius_m: float = 0.1
    end_hoop_strain: float = 0.03
    point_count: int = 481
    root_scan_points: int = 401
    origin_scan_points: int = 17
    reverse_check_points: int = 17
    common_pressure_count: int = 5

    def __post_init__(self) -> None:
        self.case_settings(2.0)
        if (isinstance(self.common_pressure_count, bool)
                or int(self.common_pressure_count) != self.common_pressure_count
                or self.common_pressure_count < 1):
            raise ValueError("common_pressure_count must be a positive integer")

    def case_settings(self, eta: float) -> StudySettings:
        inner, outer = thicknesses_m(eta, self.total_thickness_m)
        return StudySettings(
            radius_m=self.radius_m, inner_thickness_m=inner, outer_thickness_m=outer,
            outer_temperature_c=OUTER_TEMPERATURE_C, k=0.5,
            end_hoop_strain=self.end_hoop_strain, point_count=self.point_count,
            root_scan_points=self.root_scan_points, origin_scan_points=self.origin_scan_points,
            reverse_check_points=self.reverse_check_points,
        )


@dataclass(frozen=True)
class ThicknessCase:
    eta: float
    status: str
    analysis: TemperatureCase | None
    summary: dict[str, object]
    rows: tuple[dict[str, object], ...] = ()
    evaluations: tuple[dict[str, object], ...] = ()
    matched_pressure_rows: tuple[dict[str, object], ...] = ()


def membrane_quantities(point: SinglePointResult) -> dict[str, object]:
    """Actual signed forces per length; shares are blank for cancellation."""
    ni = point.inner_thickness_m * point.inner.hoop_stress_pa
    no = point.outer_thickness_m * point.outer.hoop_stress_pa
    axial = (point.inner_thickness_m * point.inner.axial_stress_pa
             + point.outer_thickness_m * point.outer.axial_stress_pa)
    load = point.gas_pressure_pa * point.radius_m
    hoop_residual, axial_residual = ni + no - load, axial - point.k * load
    if max(abs(hoop_residual), abs(axial_residual)) > point.residuals.tolerance_n_per_m:
        raise ArithmeticError("signed membrane forces failed hoop or axial equilibrium")
    pressure_floor = max(1.0, 100 * point.residuals.tolerance_n_per_m / point.radius_m)
    if abs(point.gas_pressure_pa) <= pressure_floor:
        share_status = "near_zero_pressure"
    elif ni < 0 or no < 0:
        share_status = "opposed_sign_resultants"
    else:
        share_status = "nonnegative_resultants"
    meaningful = share_status == "nonnegative_resultants"
    return {
        "inner_hoop_membrane_n_per_m": ni, "outer_hoop_membrane_n_per_m": no,
        "inner_hoop_membrane_kn_per_m": ni / 1000, "outer_hoop_membrane_kn_per_m": no / 1000,
        "pressure_radius_n_per_m": load,
        "hoop_membrane_residual_n_per_m": hoop_residual,
        "axial_membrane_residual_n_per_m": axial_residual,
        "membrane_share_status": share_status,
        "share_pressure_floor_pa": pressure_floor,
        "inner_membrane_fraction": ni / load if meaningful else None,
        "outer_membrane_fraction": no / load if meaningful else None,
    }


def _materials():
    from feodosyev_single_point import GrantaSS316TensionLaw
    from interpolation import thermal_strain
    domains = temperature_domains()
    for temperature in (INNER_TEMPERATURE_C, OUTER_TEMPERATURE_C):
        for name, (lower, upper) in domains.items():
            if not lower <= temperature <= upper:
                raise ValueError(f"{temperature:g} C outside {name} range; no extrapolation")
    inner, outer = GrantaSS316TensionLaw(INNER_TEMPERATURE_C), GrantaSS316TensionLaw(OUTER_TEMPERATURE_C)
    return (MaterialReference(inner, inner.yield_stress_pa, INNER_TEMPERATURE_C),
            MaterialReference(outer, outer.yield_stress_pa, OUTER_TEMPERATURE_C),
            thermal_strain(INNER_TEMPERATURE_C), thermal_strain(OUTER_TEMPERATURE_C), domains)


def _evaluator(eta: float, settings: ThicknessSettings) -> _PointEvaluator:
    inner, outer, ti, to, _ = _materials()
    geometry = settings.case_settings(eta)
    return _PointEvaluator(dict(
        radius_m=geometry.radius_m, inner_thickness_m=geometry.inner_thickness_m,
        outer_thickness_m=geometry.outer_thickness_m, k=geometry.k,
        inner_material=inner.material, outer_material=outer.material,
        inner_thermal_strain=ti, outer_thermal_strain=to, root_scan_points=geometry.root_scan_points,
    ))


def run_ratio_case(eta: float, settings: ThicknessSettings) -> ThicknessCase:
    analysis = None
    extra_trace: list[dict[str, object]] = []
    try:
        geometry = settings.case_settings(eta)
        inner, outer, ti, to, domains = _materials()
        analysis = solve_material_case(INNER_TEMPERATURE_C, geometry, inner, outer, ti, to)
        rows = []
        for point, diagnostic in zip(analysis.curve.points, analysis.diagnostics.rows):
            row = dict(diagnostic)
            row.update(eta=eta, total_thickness_m=settings.total_thickness_m)
            row.update(membrane_quantities(point.solution))
            row["inner_membrane_tangent_n_per_m_per_strain"] = geometry.radius_m * float(row["inner_total_pressure_slope_pa_per_strain"])
            row["outer_membrane_tangent_n_per_m_per_strain"] = geometry.radius_m * float(row["outer_total_pressure_slope_pa_per_strain"])
            rows.append(row)
        evaluator = _evaluator(eta, settings)
        extra_trace = evaluator.trace
        evaluator.cache.update({p.solution.common_total_hoop_strain: p.solution for p in analysis.curve.points})
        intervals = []
        for interval in analysis.summary["inner_below_initial_yield_intervals"]:
            left, right = interval["lower_hoop_strain"], interval["upper_hoop_strain"]
            a = evaluator.at(left, "below_yield_lower_boundary")
            b = evaluator.at(right, "below_yield_upper_boundary")
            middle = evaluator.at(0.5 * (left + right), "below_yield_interval_verification")
            ratio = middle.inner.stress_intensity_pa / inner.initial_yield_stress_pa
            if ratio >= 1 - 1e-8 or middle.inner.material_branch != "elastic":
                raise ArithmeticError("below-initial-yield interval did not verify on the tensile diagram")
            intervals.append({**interval, "lower_pressure_pa": a.gas_pressure_pa,
                              "upper_pressure_pa": b.gas_pressure_pa, "verified_interior_ratio": ratio,
                              "boundary_inclusion": "strictly below inside; yield-equality endpoints excluded, physical scan endpoints checked separately"})
        summary = dict(analysis.summary)
        summary.update(eta=eta, total_thickness_m=settings.total_thickness_m,
                       inner_thickness_mm=geometry.inner_thickness_m * 1000,
                       outer_thickness_mm=geometry.outer_thickness_m * 1000,
                       material_temperature_domains_c=domains,
                       below_initial_yield_intervals_with_pressures=intervals,
                       equal_pressure_comparison_status="pending",
                       maximum_membrane_balance_residual_n_per_m=max(abs(float(r["hoop_membrane_residual_n_per_m"])) for r in rows))
        return ThicknessCase(eta, "complete", analysis, summary, tuple(rows),
                             analysis.evaluations + tuple(extra_trace))
    except (ValueError, ArithmeticError, FileNotFoundError) as exc:
        trace = analysis.evaluations if analysis is not None else tuple(getattr(exc, "evaluations", ()))
        return ThicknessCase(eta, "incomplete", analysis,
                             {"eta": eta, "status": "incomplete", "error": str(exc),
                              "equal_pressure_comparison_status": "not_available"},
                             evaluations=trace + tuple(extra_trace))


def match_pressure_states(case: ThicknessCase, settings: ThicknessSettings,
                          targets_pa: Sequence[float]) -> ThicknessCase:
    """Resolve every detected strain root of p(strain)=target; do not pick silently."""
    if case.status != "complete" or case.analysis is None:
        raise ValueError("equal-pressure matching requires a completed strain curve")
    evaluator = _evaluator(case.eta, settings)
    points = [p.solution for p in case.analysis.curve.points]
    evaluator.cache.update({p.common_total_hoop_strain: p for p in points})
    matched = []
    try:
        for target in targets_pa:
            if not math.isfinite(target) or target <= 0:
                raise ValueError("matched pressures must be finite and positive")
            roots = []
            for a, b in zip(points[:-1], points[1:]):
                fa, fb = a.gas_pressure_pa - target, b.gas_pressure_pa - target
                if fa == fb == 0:
                    raise ValueError("a pressure plateau prevents unique strain-root enumeration")
                if fa * fb < 0 or fa == 0 or fb == 0:
                    left, right = a.common_total_hoop_strain, b.common_total_hoop_strain
                    root = left if fa == 0 else right if fb == 0 else brentq(
                        lambda x: evaluator.at(x, "matched_pressure_root").gas_pressure_pa - target,
                        left, right, xtol=1e-13, rtol=1e-13,
                    )
                    if not roots or abs(root - roots[-1]) > 1e-12:
                        roots.append(root)
            if not roots:
                raise ValueError(f"target pressure {target:.12g} Pa has no bracket on the computed curve")
            for index, root in enumerate(roots):
                point = evaluator.at(root, "matched_pressure_verification")
                if abs(point.gas_pressure_pa - target) > 1.0:
                    raise ArithmeticError("matched pressure residual exceeds 1 Pa")
                row = _csv_row(CurvePoint(len(matched), point, None, None, None))
                row.update(eta=case.eta, target_pressure_pa=target,
                           target_pressure_mpa=target / 1e6,
                           pressure_matching_residual_pa=point.gas_pressure_pa - target,
                           pressure_strain_root_count=len(roots), pressure_strain_root_index=index,
                           radius_m=point.radius_m, inner_thickness_m=point.inner_thickness_m,
                           outer_thickness_m=point.outer_thickness_m)
                row.update(membrane_quantities(point))
                matched.append(row)
        summary = dict(case.summary, equal_pressure_comparison_status="complete",
                       equal_pressure_target_count=len(targets_pa), equal_pressure_state_count=len(matched))
        return replace(case, summary=summary, matched_pressure_rows=tuple(matched),
                       evaluations=case.evaluations + tuple(evaluator.trace))
    except (ValueError, ArithmeticError) as exc:
        return replace(case, status="incomplete",
                       summary=dict(case.summary, status="incomplete", error=f"equal-pressure comparison: {exc}",
                                    equal_pressure_comparison_status="incomplete"),
                       matched_pressure_rows=tuple(matched),
                       evaluations=case.evaluations + tuple(evaluator.trace))


def compare_common_pressures(cases: Sequence[ThicknessCase], settings: ThicknessSettings):
    completed = [c for c in cases if c.status == "complete"]
    if not completed:
        return tuple(cases), {"status": "incomplete", "error": "no completed strain curves", "targets_pa": []}
    lower = max(min(p.solution.gas_pressure_pa for p in c.analysis.curve.points) for c in completed)
    upper = min(max(p.solution.gas_pressure_pa for p in c.analysis.curve.points) for c in completed)
    if upper <= max(lower, 1.0):
        failed = tuple(replace(c, status="incomplete", summary=dict(c.summary, status="incomplete",
                       error="no resolved positive-pressure overlap", equal_pressure_comparison_status="incomplete"))
                       if c.status == "complete" else c for c in cases)
        return failed, {"status": "incomplete", "error": "no resolved positive-pressure overlap", "targets_pa": []}
    targets = np.linspace(lower, upper, settings.common_pressure_count + 2)[1:-1].tolist()
    results = tuple(match_pressure_states(c, settings, targets) if c.status == "complete" else c for c in cases)
    return results, {
        "status": "complete" if all(c.status == "complete" for c in results) else "incomplete",
        "participant_ratios": [c.eta for c in completed],
        "overlap_lower_pressure_pa": lower, "overlap_upper_pressure_pa": upper,
        "targets_pa": targets,
        "method": "interior fractions of sampled positive-pressure overlap; extra equilibrium solves enumerate all detected strain roots",
        "upper_overlap_is_load_limit": False,
    }


def run_thickness_study(ratios: Sequence[float] = DEFAULT_RATIOS,
                        settings: ThicknessSettings = ThicknessSettings(), *, workers: int = 1,
                        progress: Callable[[ThicknessCase], None] | None = None):
    values = tuple(float(x) for x in ratios)
    if not values or len(set(values)) != len(values) or not all(math.isfinite(x) for x in values):
        raise ValueError("ratios must be finite, non-empty, and unique")
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    cases = {}
    if workers == 1:
        for eta in values:
            case = run_ratio_case(eta, settings)
            cases[eta] = case
            if progress:
                progress(case)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_ratio_case, eta, settings) for eta in values]
            for future in as_completed(futures):
                case = future.result()
                cases[case.eta] = case
                if progress:
                    progress(case)
    return compare_common_pressures(tuple(cases[eta] for eta in values), settings)


def summary_row(case: ThicknessCase) -> dict[str, object]:
    prefix = {"eta": case.eta, "inner_thickness_mm": case.summary.get("inner_thickness_mm"),
              "outer_thickness_mm": case.summary.get("outer_thickness_mm"),
              "total_thickness_mm": None if "total_thickness_m" not in case.summary else case.summary["total_thickness_m"] * 1000,
              "status": case.status, "error": case.summary.get("error", ""),
              "equal_pressure_comparison_status": case.summary["equal_pressure_comparison_status"]}
    if case.analysis is not None:
        prefix.update({k: v for k, v in _summary_row(case.analysis).items() if k not in {"status", "error"}})
        prefix["inner_below_initial_yield_intervals"] = json.dumps(case.summary.get("below_initial_yield_intervals_with_pressures", []))
    else:
        prefix.update({key: None for key in ("zero_pressure_hoop_strain", "outer_yield_hoop_strain",
                       "inner_hoop_zero_strain", "inner_minimum_ratio", "inner_minimum_hoop_strain",
                       "inner_minimum_pressure_pa", "inner_below_initial_yield", "pressure_at_scan_endpoint_pa")})
    return prefix


def _plots(cases: Sequence[ThicknessCase], directory: Path) -> list[Path]:
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    paths = []
    valid = [(index, c) for index, c in enumerate(cases) if c.rows]
    incomplete = [f"eta={c.eta:g}" for c in cases if c.status != "complete"]
    def decorate(figure, axes):
        for ax in np.asarray(axes, dtype=object).flat:
            ax.grid(alpha=0.3)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=9, ncol=2)
        if incomplete:
            figure.text(0.02, 0.01, "Incomplete: " + ", ".join(incomplete), color="darkred", fontsize=9)
        figure.tight_layout(rect=(0, 0.03 if incomplete else 0, 1, 0.94))
    def save(figure, name):
        path = directory / name
        with path.open("xb"):
            pass
        figure.savefig(path, format="png")
        figure.clear()
        paths.append(path)
    specs = (
        ("pressure_comparison.png", "gas_pressure_mpa", "Gas pressure (MPa)", "Pressure vs total hoop strain", None),
        ("inner_yield_ratio_comparison.png", "inner_stress_intensity_over_initial_yield", r"$\sigma_{eq,i}/\sigma_{y0,i}(538\ C)$", "Inner intensity relative to initial yield", 1.0),
        ("inner_hoop_stress_comparison.png", "inner_hoop_stress_mpa", "Inner hoop stress (MPa)", "Inner hoop stress", 0.0),
    )
    for filename, field, ylabel, title, reference in specs:
        fig = Figure(figsize=(10, 6), dpi=180)
        ax = fig.subplots()
        for index, case in valid:
            color = colormaps["tab10"](index % 10)
            ax.plot([float(r["total_hoop_strain_percent"]) for r in case.rows],
                    [float(r[field]) for r in case.rows], color=color, label=f"eta={case.eta:g}")
            if reference == 1:
                minimum = case.summary["inner_minimum"]
                ax.scatter([float(minimum["hoop_strain"]) * 100], [minimum["ratio"]], color=color, s=16, zorder=3)
        if reference is not None:
            ax.axhline(reference, color="0.3", ls="--", lw=1, label=f"Reference={reference:g}")
        ax.set(xlabel="Common total hoop strain (%)", ylabel=ylabel)
        fig.suptitle(title + "\nTotal wall = 3 mm; inner/outer = 538/20 C; k = 0.5; endpoint is prescribed")
        decorate(fig, [ax])
        save(fig, filename)
    fig = Figure(figsize=(12, 6), dpi=180)
    axes = fig.subplots(1, 2, sharex=True)
    for index, case in valid:
        for ax, name in zip(axes, ("inner", "outer")):
            ax.plot([float(r["total_hoop_strain_percent"]) for r in case.rows],
                    [float(r[f"{name}_hoop_membrane_kn_per_m"]) for r in case.rows],
                    color=colormaps["tab10"](index % 10), label=f"eta={case.eta:g}")
    for ax, name in zip(axes, ("Inner", "Outer")):
        ax.axhline(0, color="0.3", ls="--", lw=1)
        ax.set(title=name, xlabel="Common total hoop strain (%)", ylabel="Signed hoop membrane force (kN/m)")
    fig.suptitle("Actual signed membrane forces: N = h * sigma_theta\nCompression/tension cancellation is retained; these are not tangent contributions")
    decorate(fig, axes)
    save(fig, "hoop_membrane_comparison.png")
    fig = Figure(figsize=(12, 10), dpi=180)
    axes = fig.subplots(2, 2, sharex=True)
    total_field = "pressure_slope_implicit_pa_per_strain"
    all_slopes = [float(r[total_field]) / 1e8 for _, c in valid for r in c.rows]
    for index, case in valid:
        x = [float(r["total_hoop_strain_percent"]) for r in case.rows]
        for ax, field in ((axes[0, 0], total_field), (axes[0, 1], total_field),
                          (axes[1, 0], "inner_total_pressure_slope_pa_per_strain"),
                          (axes[1, 1], "outer_total_pressure_slope_pa_per_strain")):
            ax.plot(x, [float(r[field]) / 1e8 for r in case.rows],
                    color=colormaps["tab10"](index % 10), label=f"eta={case.eta:g}")
    for ax, title in zip(axes.flat, ("Total tangent (full scale)", "Total tangent (vertical zoom; high values clipped)",
                                   "Inner tangent contribution", "Outer tangent contribution")):
        ax.set(title=title, ylabel="Pressure slope (MPa / % strain)")
    if all_slopes:
        axes[0, 1].set_ylim(min(0.0, min(all_slopes) * 1.1), max(0.1, float(np.percentile(all_slopes, 95)) * 1.1))
    for ax in axes[1]:
        ax.set_xlabel("Common total hoop strain (%)")
    fig.suptitle("Local pressure tangent and shell contributions\nFixed thicknesses within each curve; no A-D or limit identification")
    decorate(fig, axes)
    save(fig, "pressure_tangent_comparison.png")
    fig = Figure(figsize=(12, 6), dpi=180)
    axes = fig.subplots(1, 2, sharex=True)
    targets = sorted({float(r["target_pressure_pa"]) for c in cases for r in c.matched_pressure_rows})
    for index, target in enumerate(targets):
        rows = [r for c in cases for r in c.matched_pressure_rows if r["target_pressure_pa"] == target]
        for ax, name in zip(axes, ("inner", "outer")):
            ax.scatter([r["eta"] for r in rows], [r[f"{name}_hoop_membrane_kn_per_m"] for r in rows],
                       color=colormaps["tab10"](index % 10), label=f"p={target / 1e6:.4f} MPa", s=25)
    for ax, name in zip(axes, ("Inner", "Outer")):
        ax.axhline(0, color="0.3", ls="--", lw=1)
        ax.set(title=name, xlabel="Thickness ratio eta = h_outer / h_inner", ylabel="Signed hoop membrane force (kN/m)")
    fig.suptitle("Matched-pressure load sharing: extra equilibrium solves\nAll detected strain roots retained; signed forces rather than cancellation percentages")
    decorate(fig, axes)
    save(fig, "equal_pressure_membrane_comparison.png")
    return paths


def export_thickness_study(cases: Sequence[ThicknessCase], settings: ThicknessSettings,
                            comparison: dict[str, object], directory: Path) -> dict[str, object]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write_csv(directory / "thickness_summary.csv", [summary_row(c) for c in cases])
    matches = [r for c in cases for r in c.matched_pressure_rows]
    _write_csv(directory / "equal_pressure_states.csv", matches or [{"status": "incomplete", "error": "no matched states"}])
    for case in cases:
        case_dir = directory / f"eta_{case.eta:g}"
        case_dir.mkdir()
        if case.rows:
            _write_csv(case_dir / "points.csv", case.rows)
        if case.evaluations:
            _write_csv(case_dir / "evaluations.csv", [dict(r, evaluation_index=i) for i, r in enumerate(case.evaluations)])
        with (case_dir / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(case.summary, stream, indent=2, allow_nan=False)
    plots = _plots(cases, directory)
    root = Path(__file__).resolve().parent
    frozen = ("feodosyev_single_point.py", "feodosyev_curve.py", "feodosyev_diagnostics.py",
              "feodosyev_temperature_study.py", "interpolation.py")
    manifest = {
        "baseline_commit": BASELINE_COMMIT, "settings": asdict(settings),
        "inner_temperature_c": INNER_TEMPERATURE_C, "outer_temperature_c": OUTER_TEMPERATURE_C, "k": 0.5,
        "case_status": {str(c.eta): c.status for c in cases},
        "all_cases_complete": all(c.status == "complete" for c in cases),
        "equal_pressure_comparison": comparison,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in frozen},
        "endpoint_is_load_limit": False, "stage_classification_performed": False,
        "plots": [str(p.resolve()) for p in plots],
    }
    with (directory / "study_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--points", type=int, default=481)
    parser.add_argument("--root-scan-points", type=int, default=401)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        parser.error("output directory already exists; choose a new run directory")
    settings = ThicknessSettings(point_count=args.points, root_scan_points=args.root_scan_points)
    def report(case: ThicknessCase) -> None:
        minimum = case.summary.get("inner_minimum", {}).get("ratio")
        print(f"eta={case.eta:g}: {case.status}; min_ratio={minimum}; {case.summary.get('error', '')}", flush=True)
    cases, comparison = run_thickness_study(args.ratios, settings, workers=args.workers, progress=report)
    manifest = export_thickness_study(cases, settings, comparison, args.output_directory)
    print(json.dumps(manifest, indent=2, allow_nan=False), flush=True)
    if not manifest["all_cases_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

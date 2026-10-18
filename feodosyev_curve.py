"""Stage 2 pressure-strain curve built from the verified single-point solver.

The module performs continuation over a prescribed increasing hoop-strain
range.  It does not classify Feodosyev Fig. 9.7 stages A-D and does not treat
the last sampled point as a load-bearing limit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.figure import Figure

from feodosyev_single_point import (
    GrantaSS316TensionLaw,
    SinglePointResult,
    TensionLaw,
    solve_single_point,
)


@dataclass(frozen=True)
class CurvePoint:
    """One solved point plus the continuation information used to reach it."""

    index: int
    solution: SinglePointResult
    axial_strain_guess: float | None
    axial_strain_increment: float | None
    axial_predictor_error: float | None


@dataclass(frozen=True)
class CurveDiagnostics:
    """Checks and descriptive statistics for the completed scan."""

    point_count: int
    minimum_pressure_pa: float
    maximum_pressure_pa: float
    maximum_absolute_equilibrium_residual_n_per_m: float
    maximum_relative_equilibrium_residual: float
    maximum_absolute_axial_strain_increment: float
    maximum_absolute_axial_predictor_error: float
    multiple_root_point_count: int
    pressure_decrease_segment_count: int
    endpoint_is_load_limit: bool = False
    stage_classification_performed: bool = False


@dataclass(frozen=True)
class CurveResult:
    """A complete prescribed-range continuation result."""

    points: tuple[CurvePoint, ...]
    diagnostics: CurveDiagnostics

    @property
    def start_hoop_strain(self) -> float:
        return self.points[0].solution.common_total_hoop_strain

    @property
    def end_hoop_strain(self) -> float:
        return self.points[-1].solution.common_total_hoop_strain


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_hoop_strains(values: Iterable[float]) -> tuple[float, ...]:
    strains = tuple(_finite(value, "hoop strain") for value in values)
    if len(strains) < 2:
        raise ValueError("a curve requires at least two hoop-strain points")
    if any(right <= left for left, right in zip(strains[:-1], strains[1:])):
        raise ValueError("hoop-strain points must be strictly increasing")
    return strains


def _predict_axial_strain(
    completed: list[CurvePoint], next_hoop_strain: float
) -> float:
    """Use constant or linear extrapolation from the already accepted branch."""

    latest = completed[-1].solution
    if len(completed) == 1:
        return latest.common_total_axial_strain

    previous = completed[-2].solution
    hoop_step = (
        latest.common_total_hoop_strain - previous.common_total_hoop_strain
    )
    axial_slope = (
        latest.common_total_axial_strain
        - previous.common_total_axial_strain
    ) / hoop_step
    return latest.common_total_axial_strain + axial_slope * (
        next_hoop_strain - latest.common_total_hoop_strain
    )


def _validate_point(
    result: SinglePointResult,
    inner_material: TensionLaw,
    outer_material: TensionLaw,
) -> None:
    """Recheck all stage-2 acceptance conditions for one returned point."""

    residuals = result.residuals
    for label, value in (
        ("equation (9.1)", residuals.equation_9_1_n_per_m),
        ("equation (9.2)", residuals.equation_9_2_n_per_m),
        ("equation (9.5)", residuals.equation_9_5_n_per_m),
    ):
        if not math.isfinite(value) or abs(value) > residuals.tolerance_n_per_m:
            raise ArithmeticError(
                f"{label} residual exceeds tolerance: "
                f"{value:.9g} N/m > {residuals.tolerance_n_per_m:.9g} N/m"
            )

    pressure_tolerance = residuals.tolerance_n_per_m / result.radius_m
    if result.gas_pressure_pa < -pressure_tolerance:
        raise ArithmeticError(
            f"negative gas pressure passed the single-point check: "
            f"{result.gas_pressure_pa:.9g} Pa"
        )

    for name, shell, material in (
        ("inner", result.inner, inner_material),
        ("outer", result.outer, outer_material),
    ):
        material_tolerance = max(
            1e-14, 1e-12 * material.max_total_mechanical_strain
        )
        if (
            shell.strain_intensity
            > material.max_total_mechanical_strain + material_tolerance
        ):
            raise ArithmeticError(
                f"{name} shell strain intensity exceeds its material domain"
            )
        if not math.isclose(
            shell.strain_intensity,
            shell.uniaxial_elastic_strain + shell.uniaxial_plastic_strain,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ArithmeticError(
                f"{name} shell elastic/plastic strain decomposition is inconsistent"
            )


def solve_curve(
    *,
    hoop_strains: Iterable[float],
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    inner_thermal_strain: float,
    outer_thermal_strain: float,
    inner_material: TensionLaw,
    outer_material: TensionLaw,
    k: float = 0.5,
    initial_axial_strain_guess: float | None = None,
    root_scan_points: int = 401,
) -> CurveResult:
    """Solve a prescribed increasing sequence with axial-root continuation.

    The first point uses ``initial_axial_strain_guess`` when supplied.  Every
    later point uses the accepted branch as a predictor: one prior point gives
    a constant predictor, while two or more points give linear extrapolation.
    If the single-point equation has multiple roots, its existing rule selects
    the root nearest this predictor.  A multiple-root first point without an
    explicit guess is rejected rather than silently choosing a branch.
    """

    strains = _validated_hoop_strains(hoop_strains)
    if initial_axial_strain_guess is not None:
        initial_axial_strain_guess = _finite(
            initial_axial_strain_guess, "initial_axial_strain_guess"
        )

    completed: list[CurvePoint] = []
    for index, hoop_strain in enumerate(strains):
        if index == 0:
            guess = initial_axial_strain_guess
        else:
            guess = _predict_axial_strain(completed, hoop_strain)

        try:
            solution = solve_single_point(
                radius_m=radius_m,
                inner_thickness_m=inner_thickness_m,
                outer_thickness_m=outer_thickness_m,
                common_total_hoop_strain=hoop_strain,
                inner_thermal_strain=inner_thermal_strain,
                outer_thermal_strain=outer_thermal_strain,
                inner_material=inner_material,
                outer_material=outer_material,
                k=k,
                axial_strain_guess=guess,
                allow_negative_pressure=False,
                root_scan_points=root_scan_points,
            )
            _validate_point(solution, inner_material, outer_material)
        except (ValueError, ArithmeticError) as exc:
            raise type(exc)(
                f"curve point {index + 1}/{len(strains)}, "
                f"total hoop strain {hoop_strain:.12g}: {exc}"
            ) from exc

        previous_axial = (
            None
            if not completed
            else completed[-1].solution.common_total_axial_strain
        )
        completed.append(
            CurvePoint(
                index=index,
                solution=solution,
                axial_strain_guess=guess,
                axial_strain_increment=(
                    None
                    if previous_axial is None
                    else solution.common_total_axial_strain - previous_axial
                ),
                axial_predictor_error=(
                    None
                    if guess is None
                    else solution.common_total_axial_strain - guess
                ),
            )
        )

    pressures = [point.solution.gas_pressure_pa for point in completed]
    equilibrium_residuals = [
        abs(value)
        for point in completed
        for value in (
            point.solution.residuals.equation_9_1_n_per_m,
            point.solution.residuals.equation_9_2_n_per_m,
            point.solution.residuals.equation_9_5_n_per_m,
        )
    ]
    axial_increments = [
        abs(point.axial_strain_increment)
        for point in completed
        if point.axial_strain_increment is not None
    ]
    predictor_errors = [
        abs(point.axial_predictor_error)
        for point in completed
        if point.axial_predictor_error is not None
    ]
    diagnostics = CurveDiagnostics(
        point_count=len(completed),
        minimum_pressure_pa=min(pressures),
        maximum_pressure_pa=max(pressures),
        maximum_absolute_equilibrium_residual_n_per_m=max(
            equilibrium_residuals
        ),
        maximum_relative_equilibrium_residual=max(
            point.solution.residuals.maximum_relative for point in completed
        ),
        maximum_absolute_axial_strain_increment=max(axial_increments),
        maximum_absolute_axial_predictor_error=max(predictor_errors),
        multiple_root_point_count=sum(
            point.solution.root_count > 1 for point in completed
        ),
        pressure_decrease_segment_count=sum(
            right < left for left, right in zip(pressures[:-1], pressures[1:])
        ),
    )
    return CurveResult(points=tuple(completed), diagnostics=diagnostics)


def solve_uniform_curve(
    *,
    start_hoop_strain: float,
    end_hoop_strain: float,
    point_count: int,
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    inner_thermal_strain: float,
    outer_thermal_strain: float,
    inner_material: TensionLaw,
    outer_material: TensionLaw,
    k: float = 0.5,
    initial_axial_strain_guess: float | None = None,
    root_scan_points: int = 401,
) -> CurveResult:
    """Convenience wrapper for an endpoint-inclusive uniform strain grid."""

    start = _finite(start_hoop_strain, "start_hoop_strain")
    end = _finite(end_hoop_strain, "end_hoop_strain")
    if isinstance(point_count, bool) or int(point_count) != point_count:
        raise ValueError("point_count must be an integer")
    point_count = int(point_count)
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    if end <= start:
        raise ValueError("end_hoop_strain must be greater than start_hoop_strain")
    return solve_curve(
        hoop_strains=np.linspace(start, end, point_count),
        radius_m=radius_m,
        inner_thickness_m=inner_thickness_m,
        outer_thickness_m=outer_thickness_m,
        inner_thermal_strain=inner_thermal_strain,
        outer_thermal_strain=outer_thermal_strain,
        inner_material=inner_material,
        outer_material=outer_material,
        k=k,
        initial_axial_strain_guess=initial_axial_strain_guess,
        root_scan_points=root_scan_points,
    )


def solve_ss316_uniform_curve(
    *,
    start_hoop_strain: float,
    end_hoop_strain: float,
    point_count: int,
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    inner_temperature_c: float,
    outer_temperature_c: float,
    k: float = 0.5,
    initial_axial_strain_guess: float | None = None,
    root_scan_points: int = 401,
) -> CurveResult:
    """Build a curve using the local Granta SS316 data for both shells."""

    from interpolation import thermal_strain  # pylint: disable=import-outside-toplevel

    inner_temperature = _finite(inner_temperature_c, "inner_temperature_c")
    outer_temperature = _finite(outer_temperature_c, "outer_temperature_c")
    return solve_uniform_curve(
        start_hoop_strain=start_hoop_strain,
        end_hoop_strain=end_hoop_strain,
        point_count=point_count,
        radius_m=radius_m,
        inner_thickness_m=inner_thickness_m,
        outer_thickness_m=outer_thickness_m,
        inner_thermal_strain=thermal_strain(inner_temperature),
        outer_thermal_strain=thermal_strain(outer_temperature),
        inner_material=GrantaSS316TensionLaw(inner_temperature),
        outer_material=GrantaSS316TensionLaw(outer_temperature),
        k=k,
        initial_axial_strain_guess=initial_axial_strain_guess,
        root_scan_points=root_scan_points,
    )


CSV_FIELDS = (
    "point_index",
    "total_hoop_strain",
    "total_hoop_strain_percent",
    "total_axial_strain",
    "gas_pressure_pa",
    "gas_pressure_mpa",
    "axial_strain_guess",
    "axial_strain_increment",
    "axial_predictor_error",
    "root_count",
    "root_selection",
    "equation_9_1_residual_n_per_m",
    "equation_9_2_residual_n_per_m",
    "equation_9_5_residual_n_per_m",
    "equilibrium_tolerance_n_per_m",
    "maximum_relative_equilibrium_residual",
    "inner_axial_stress_pa",
    "inner_axial_stress_mpa",
    "inner_hoop_stress_pa",
    "inner_hoop_stress_mpa",
    "inner_stress_intensity_pa",
    "inner_mechanical_axial_strain",
    "inner_mechanical_hoop_strain",
    "inner_strain_intensity",
    "inner_uniaxial_elastic_strain",
    "inner_uniaxial_plastic_strain",
    "inner_material_branch",
    "outer_axial_stress_pa",
    "outer_axial_stress_mpa",
    "outer_hoop_stress_pa",
    "outer_hoop_stress_mpa",
    "outer_stress_intensity_pa",
    "outer_mechanical_axial_strain",
    "outer_mechanical_hoop_strain",
    "outer_strain_intensity",
    "outer_uniaxial_elastic_strain",
    "outer_uniaxial_plastic_strain",
    "outer_material_branch",
)


def _csv_row(point: CurvePoint) -> dict[str, object]:
    result = point.solution
    inner = result.inner
    outer = result.outer
    residuals = result.residuals
    return {
        "point_index": point.index,
        "total_hoop_strain": result.common_total_hoop_strain,
        "total_hoop_strain_percent": result.common_total_hoop_strain * 100.0,
        "total_axial_strain": result.common_total_axial_strain,
        "gas_pressure_pa": result.gas_pressure_pa,
        "gas_pressure_mpa": result.gas_pressure_pa / 1e6,
        "axial_strain_guess": point.axial_strain_guess,
        "axial_strain_increment": point.axial_strain_increment,
        "axial_predictor_error": point.axial_predictor_error,
        "root_count": result.root_count,
        "root_selection": result.root_selection,
        "equation_9_1_residual_n_per_m": residuals.equation_9_1_n_per_m,
        "equation_9_2_residual_n_per_m": residuals.equation_9_2_n_per_m,
        "equation_9_5_residual_n_per_m": residuals.equation_9_5_n_per_m,
        "equilibrium_tolerance_n_per_m": residuals.tolerance_n_per_m,
        "maximum_relative_equilibrium_residual": residuals.maximum_relative,
        "inner_axial_stress_pa": inner.axial_stress_pa,
        "inner_axial_stress_mpa": inner.axial_stress_pa / 1e6,
        "inner_hoop_stress_pa": inner.hoop_stress_pa,
        "inner_hoop_stress_mpa": inner.hoop_stress_pa / 1e6,
        "inner_stress_intensity_pa": inner.stress_intensity_pa,
        "inner_mechanical_axial_strain": inner.mechanical_axial_strain,
        "inner_mechanical_hoop_strain": inner.mechanical_hoop_strain,
        "inner_strain_intensity": inner.strain_intensity,
        "inner_uniaxial_elastic_strain": inner.uniaxial_elastic_strain,
        "inner_uniaxial_plastic_strain": inner.uniaxial_plastic_strain,
        "inner_material_branch": inner.material_branch,
        "outer_axial_stress_pa": outer.axial_stress_pa,
        "outer_axial_stress_mpa": outer.axial_stress_pa / 1e6,
        "outer_hoop_stress_pa": outer.hoop_stress_pa,
        "outer_hoop_stress_mpa": outer.hoop_stress_pa / 1e6,
        "outer_stress_intensity_pa": outer.stress_intensity_pa,
        "outer_mechanical_axial_strain": outer.mechanical_axial_strain,
        "outer_mechanical_hoop_strain": outer.mechanical_hoop_strain,
        "outer_strain_intensity": outer.strain_intensity,
        "outer_uniaxial_elastic_strain": outer.uniaxial_elastic_strain,
        "outer_uniaxial_plastic_strain": outer.uniaxial_plastic_strain,
        "outer_material_branch": outer.material_branch,
    }


def write_curve_csv(curve: CurveResult, path: Path) -> None:
    """Write every solved point, stress component, and check to CSV."""

    path = Path(path)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in curve.points:
            writer.writerow(_csv_row(point))


def write_pressure_strain_png(curve: CurveResult, path: Path) -> None:
    """Plot only the actual computed points; no smoothing or stage labels."""

    path = Path(path)
    try:
        with path.open("xb"):
            pass
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing file: {path}"
        ) from exc
    figure: Figure | None = None
    try:
        hoop_percent = [
            point.solution.common_total_hoop_strain * 100.0
            for point in curve.points
        ]
        pressure_mpa = [
            point.solution.gas_pressure_pa / 1e6 for point in curve.points
        ]
        marker_spacing = max(1, len(curve.points) // 25)
        figure = Figure(figsize=(9.5, 5.8), dpi=180)
        axes = figure.subplots()
        axes.plot(
            hoop_percent,
            pressure_mpa,
            color="#287fbd",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            markevery=marker_spacing,
            label="Computed points",
        )
        axes.set_xlabel(
            r"Common total hoop strain $\varepsilon_{\theta,tot}$ (%)"
        )
        axes.set_ylabel(r"Gas pressure $p_g$ (MPa)")
        axes.set_title(
            "Feodosyev section 9.3 prescribed-strain scan\n"
            "No A-D classification; the scan endpoint is not a load limit"
        )
        axes.grid(True, alpha=0.35)
        axes.legend()
        figure.tight_layout()
        figure.savefig(path, format="png")
    except Exception:
        if path.exists():
            path.unlink()
        raise
    finally:
        if figure is not None:
            figure.clear()


def export_curve(
    curve: CurveResult, output_directory: Path, prefix: str
) -> tuple[Path, Path]:
    """Export CSV and PNG without overwriting existing research outputs."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be a non-empty filename prefix without a path")
    csv_path = directory / f"{prefix}.csv"
    png_path = directory / f"{prefix}.png"
    existing = [path for path in (csv_path, png_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing)
        )
    created: list[Path] = []
    try:
        write_curve_csv(curve, csv_path)
        created.append(csv_path)
        write_pressure_strain_png(curve, png_path)
        created.append(png_path)
    except Exception:
        for path in created:
            if path.exists():
                path.unlink()
        raise
    return csv_path, png_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prescribed-range Feodosyev pressure-strain curve without "
            "A-D stage classification or load-limit inference"
        )
    )
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument("--inner-thickness-mm", type=float, default=1.0)
    parser.add_argument("--outer-thickness-mm", type=float, default=2.0)
    parser.add_argument("--inner-temperature-c", type=float, default=538.0)
    parser.add_argument("--outer-temperature-c", type=float, default=20.0)
    parser.add_argument("--start-hoop-strain", type=float, default=0.00025)
    parser.add_argument("--end-hoop-strain", type=float, default=0.03)
    parser.add_argument("--points", type=int, default=121)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--initial-axial-strain-guess", type=float)
    parser.add_argument("--root-scan-points", type=int, default=401)
    parser.add_argument("--output-directory", type=Path, default=Path("outputs"))
    parser.add_argument("--prefix", default="feodosyev_stage2_curve")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    curve = solve_ss316_uniform_curve(
        start_hoop_strain=args.start_hoop_strain,
        end_hoop_strain=args.end_hoop_strain,
        point_count=args.points,
        radius_m=args.radius_mm * 1e-3,
        inner_thickness_m=args.inner_thickness_mm * 1e-3,
        outer_thickness_m=args.outer_thickness_mm * 1e-3,
        inner_temperature_c=args.inner_temperature_c,
        outer_temperature_c=args.outer_temperature_c,
        k=args.k,
        initial_axial_strain_guess=args.initial_axial_strain_guess,
        root_scan_points=args.root_scan_points,
    )
    csv_path, png_path = export_curve(
        curve, args.output_directory, args.prefix
    )
    print(
        json.dumps(
            {
                "start_hoop_strain": curve.start_hoop_strain,
                "end_hoop_strain": curve.end_hoop_strain,
                "diagnostics": asdict(curve.diagnostics),
                "csv": str(csv_path.resolve()),
                "png": str(png_path.resolve()),
                "note": (
                    "The endpoint is only the prescribed scan endpoint; "
                    "no A-D stage or load limit was identified."
                ),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

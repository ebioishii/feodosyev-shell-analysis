"""Stress, yield-reference, and local-slope diagnostics for an existing curve.

This differentiates Feodosyev's deformation-theory equations at fixed
temperature, geometry, and k. It does not implement incremental plasticity,
identify A-D, or estimate a load limit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from feodosyev_curve import CurveResult, _csv_row, export_curve, solve_uniform_curve
from feodosyev_single_point import GrantaSS316TensionLaw, ShellResponse, TensionLaw


@dataclass(frozen=True)
class MaterialReference:
    material: TensionLaw
    initial_yield_stress_pa: float
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_yield_stress_pa) or self.initial_yield_stress_pa <= 0:
            raise ValueError("initial yield stress must be finite and positive")
        if self.temperature_c is not None and not math.isfinite(self.temperature_c):
            raise ValueError("temperature must be finite")

    @property
    def initial_yield_strain(self) -> float:
        return self.initial_yield_stress_pa / self.material.elastic_modulus_pa


@dataclass(frozen=True)
class DiagnosticResult:
    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]


def _material_tangent(material: TensionLaw, strain: float, step: float) -> float:
    """Local slope of the *total mechanical* tensile diagram, not plastic slope."""
    left = max(0.0, strain - step)
    right = min(material.max_total_mechanical_strain, strain + step)
    if right <= left:
        raise ValueError("no material interval available for the tangent")
    return (material.evaluate(right).stress_pa - material.evaluate(left).stress_pa) / (right - left)


def constitutive_tangent(
    shell: ShellResponse, material: TensionLaw, step: float = 1e-7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return d(sigma_x,sigma_y)/d(e_x,e_y), split by magnitude/direction.

    The magnitude term varies sigma_i with epsilon_i. The direction term
    varies the normalized stress direction at fixed sigma_i. Their sum is
    exactly the derivative of (9.4) for the estimated tensile tangent.
    """
    if not math.isfinite(step) or step <= 0:
        raise ValueError("tensile tangent step must be finite and positive")
    intensity = shell.strain_intensity
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    if intensity <= 1e-14:
        jacobian = 4.0 / 3.0 * material.elastic_modulus_pa * matrix
        # No unique normalized direction exists at zero; do not invent a split.
        return jacobian, jacobian.copy(), np.zeros((2, 2)), material.elastic_modulus_pa, 0.0
    tangent = _material_tangent(material, intensity, step)
    half_step_tangent = _material_tangent(material, intensity, step / 2.0)
    vector = matrix @ np.array([
        shell.mechanical_axial_strain, shell.mechanical_hoop_strain
    ])
    dyad = 16.0 / 9.0 * np.outer(vector, vector) / intensity**2
    magnitude = tangent * dyad
    direction = shell.secant_modulus_pa * (4.0 / 3.0 * matrix - dyad)
    return magnitude + direction, magnitude, direction, tangent, abs(tangent - half_step_tangent)


def _sign(stress: float, tolerance_pa: float) -> str:
    if stress < -tolerance_pa:
        return "compression"
    if stress > tolerance_pa:
        return "tension"
    return "near_zero"


def _crossings(rows: list[dict[str, object]], field: str, target: float) -> list[dict[str, float]]:
    """Report observed brackets and linear estimates, never stage boundaries."""
    brackets = []
    for left, right in zip(rows[:-1], rows[1:]):
        a, b = float(left[field]) - target, float(right[field]) - target
        if a * b < 0.0 or (a == 0.0 and b != 0.0) or (b == 0.0 and a != 0.0):
            x0, x1 = float(left["total_hoop_strain"]), float(right["total_hoop_strain"])
            brackets.append({
                "lower_hoop_strain": x0,
                "upper_hoop_strain": x1,
                "linear_estimate_hoop_strain": x0 - a * (x1 - x0) / (b - a),
            })
    return brackets


def diagnose_curve(
    curve: CurveResult,
    inner: MaterialReference,
    outer: MaterialReference,
    *,
    tensile_tangent_step: float = 1e-7,
    stress_zero_tolerance_pa: float = 1.0,
) -> DiagnosticResult:
    """Compare finite-difference slopes with implicit equilibrium tangents."""
    if len(curve.points) < 3:
        raise ValueError("slope diagnostics require at least three curve points")
    if not math.isfinite(stress_zero_tolerance_pa) or stress_zero_tolerance_pa < 0:
        raise ValueError("stress zero tolerance must be finite and non-negative")
    strains = np.array([p.solution.common_total_hoop_strain for p in curve.points])
    if not np.all(np.isfinite(strains)) or not np.all(np.diff(strains) > 0):
        raise ValueError("curve strain points must be finite and strictly increasing")
    pressures = np.array([p.solution.gas_pressure_pa for p in curve.points])
    gradient = np.gradient(pressures, strains, edge_order=2)
    rows = []
    first = curve.points[0].solution
    constants = lambda s: (s.radius_m, s.inner_thickness_m, s.outer_thickness_m, s.k,
                           s.inner.thermal_strain, s.outer.thermal_strain)
    for index, point in enumerate(curve.points):
        solution = point.solution
        if constants(solution) != constants(first):
            raise ValueError("diagnostic curve must have fixed geometry, temperatures, and k")
        residual = solution.residuals
        if max(abs(residual.equation_9_1_n_per_m), abs(residual.equation_9_2_n_per_m),
               abs(residual.equation_9_5_n_per_m)) > residual.tolerance_n_per_m:
            raise ValueError("curve contains a point that fails equilibrium")
        if solution.gas_pressure_pa < -residual.tolerance_n_per_m / solution.radius_m:
            raise ValueError("curve contains negative pressure")
        row = _csv_row(point)
        row.update({
            "radius_m": solution.radius_m,
            "inner_thickness_m": solution.inner_thickness_m,
            "outer_thickness_m": solution.outer_thickness_m,
            "k": solution.k,
            "inner_thermal_strain": solution.inner.thermal_strain,
            "outer_thermal_strain": solution.outer.thermal_strain,
        })
        tangents = []
        for name, shell, reference in (("inner", solution.inner, inner), ("outer", solution.outer, outer)):
            response = reference.material.evaluate(shell.strain_intensity)
            if not math.isclose(response.stress_pa, shell.stress_intensity_pa, rel_tol=1e-9, abs_tol=1e-4):
                raise ValueError(f"{name} diagnostic material does not match the solved curve")
            jacobian, magnitude, direction, tangent, sensitivity = constitutive_tangent(
                shell, reference.material, tensile_tangent_step
            )
            tangents.append((jacobian, magnitude, direction))
            ratio = shell.stress_intensity_pa / reference.initial_yield_stress_pa
            row.update({
                f"{name}_temperature_c": reference.temperature_c,
                f"{name}_initial_yield_stress_pa": reference.initial_yield_stress_pa,
                f"{name}_initial_yield_strain": reference.initial_yield_strain,
                f"{name}_stress_intensity_over_initial_yield": ratio,
                f"{name}_hoop_stress_over_initial_yield": shell.hoop_stress_pa / reference.initial_yield_stress_pa,
                f"{name}_strain_intensity_over_initial_yield_strain": shell.strain_intensity / reference.initial_yield_strain,
                f"{name}_initial_yield_comparison": (
                    "above" if ratio > 1.0 + 1e-8 else "below" if ratio < 1.0 - 1e-8 else "near"
                ),
                f"{name}_hoop_stress_sign": _sign(shell.hoop_stress_pa, stress_zero_tolerance_pa),
                f"{name}_axial_stress_sign": _sign(shell.axial_stress_pa, stress_zero_tolerance_pa),
                f"{name}_domain_utilization": shell.strain_intensity / reference.material.max_total_mechanical_strain,
                f"{name}_uniaxial_tangent_pa": tangent,
                f"{name}_tangent_half_step_difference_pa": sensitivity,
            })
        hi, ho, radius, k = solution.inner_thickness_m, solution.outer_thickness_m, solution.radius_m, solution.k
        ji, jo = tangents[0][0], tangents[1][0]
        axial_derivative = hi * (ji[0, 0] - k * ji[1, 0]) + ho * (jo[0, 0] - k * jo[1, 0])
        hoop_derivative = hi * (ji[0, 1] - k * ji[1, 1]) + ho * (jo[0, 1] - k * jo[1, 1])
        scale = hi * np.linalg.norm(ji) + ho * np.linalg.norm(jo)
        if abs(axial_derivative) <= 1e-12 * max(scale, 1.0):
            raise ArithmeticError("local axial tangent is singular; slope is not uniquely defined")
        axial_slope = -hoop_derivative / axial_derivative
        direction_vector = np.array([axial_slope, 1.0])
        total_slope = 0.0
        for name, thickness, parts in (("inner", hi, tangents[0]), ("outer", ho, tangents[1])):
            for label, matrix in zip(("total", "magnitude", "direction"), parts):
                row[f"{name}_{label}_pressure_slope_pa_per_strain"] = float(thickness / radius * (matrix @ direction_vector)[1])
            total_slope += float(row[f"{name}_total_pressure_slope_pa_per_strain"])
        row.update({
            "axial_strain_slope": float(axial_slope),
            "pressure_slope_implicit_pa_per_strain": total_slope,
            "pressure_slope_grid_pa_per_strain": float(gradient[index]),
            "pressure_slope_grid_minus_implicit_pa_per_strain": float(gradient[index] - total_slope),
            "grid_derivative_location": "one_sided_endpoint" if index in (0, len(strains) - 1) else "interior",
            "grid_stencil_material_branch_change": any(
                getattr(curve.points[j].solution, name).material_branch != getattr(solution, name).material_branch
                for j in range(max(0, index - 1), min(len(strains), index + 2))
                for name in ("inner", "outer")
            ),
        })
        if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (float, int))):
            raise ArithmeticError("non-finite diagnostic result")
        rows.append(row)
    summary: dict[str, object] = {
        "point_count": len(rows),
        "radius_m": first.radius_m, "inner_thickness_m": first.inner_thickness_m,
        "outer_thickness_m": first.outer_thickness_m, "k": first.k,
        "start_hoop_strain": float(strains[0]), "end_hoop_strain": float(strains[-1]),
        "tensile_tangent_step": tensile_tangent_step,
        "stress_zero_tolerance_pa": stress_zero_tolerance_pa,
        "curve_diagnostics": asdict(curve.diagnostics),
        "stage_classification_performed": False, "endpoint_is_load_limit": False,
        "interpretation": "Material branch is a current tensile-diagram state, not accumulated plastic history. Crossings are sampled brackets, not A-D boundaries.",
    }
    for name, reference in (("inner", inner), ("outer", outer)):
        minimum = min(rows, key=lambda r: float(r[f"{name}_stress_intensity_over_initial_yield"]))
        summary[name] = {
            "temperature_c": reference.temperature_c,
            "initial_yield_stress_pa": reference.initial_yield_stress_pa,
            "initial_yield_strain": reference.initial_yield_strain,
            "minimum_stress_intensity_over_initial_yield": minimum[f"{name}_stress_intensity_over_initial_yield"],
            "hoop_strain_at_minimum_intensity": minimum["total_hoop_strain"],
            "material_branches": sorted({str(r[f"{name}_material_branch"]) for r in rows}),
            "plastic_diagram_decrease_segment_count": sum(
                float(b[f"{name}_uniaxial_plastic_strain"]) < float(a[f"{name}_uniaxial_plastic_strain"]) - 1e-12
                for a, b in zip(rows[:-1], rows[1:])
            ),
            "hoop_zero_brackets": _crossings(rows, f"{name}_hoop_stress_pa", 0.0),
            "initial_yield_brackets": _crossings(rows, f"{name}_strain_intensity_over_initial_yield_strain", 1.0),
            "material_branch_change_brackets": [
                {"lower_hoop_strain": a["total_hoop_strain"], "upper_hoop_strain": b["total_hoop_strain"],
                 "from": a[f"{name}_material_branch"], "to": b[f"{name}_material_branch"]}
                for a, b in zip(rows[:-1], rows[1:]) if a[f"{name}_material_branch"] != b[f"{name}_material_branch"]
            ],
            "maximum_tangent_half_step_difference_pa": max(float(r[f"{name}_tangent_half_step_difference_pa"]) for r in rows),
        }
    return DiagnosticResult(tuple(rows), summary)


def write_diagnostic_plot(result: DiagnosticResult, path: Path) -> None:
    """Plot stress signs, invariant yield references, and slope contributions."""
    with Path(path).open("xb"):
        pass
    figure = Figure(figsize=(13, 14), dpi=160)
    try:
        axes = figure.subplots(3, 2, sharex=True)
        x = np.array([float(r["total_hoop_strain_percent"]) for r in result.rows])
        values = lambda field: np.array([float(r[field]) for r in result.rows])
        for name, color in (("inner", "#c55432"), ("outer", "#287fbd")):
            axes[0, 0].plot(x, values(f"{name}_hoop_stress_mpa"), color=color, label=name.title())
            axes[0, 1].plot(x, values(f"{name}_stress_intensity_over_initial_yield"), color=color, label=name.title())
            for ax in axes[1]:
                ax.plot(x, values(f"{name}_total_pressure_slope_pa_per_strain") / 1e8, color=color, label=name.title() + " contribution")
            axes[2, 1].plot(x, values(f"{name}_uniaxial_plastic_strain") * 100, color=color, label=name.title())
        axes[0, 0].axhline(0, color="0.35", ls="--", lw=1)
        axes[0, 0].set(title="Hoop stress: compression / tension", ylabel="Hoop stress (MPa)")
        axes[0, 1].axhline(1, color="0.35", ls="--", lw=1, label="Initial yield reference")
        axes[0, 1].set(title="Biaxial stress intensity vs temperature-specific yield", ylabel=r"$\sigma_i/\sigma_{y,0}(T)$")
        for ax in axes[1]:
            ax.plot(x, values("pressure_slope_implicit_pa_per_strain") / 1e8, color="black", label="Implicit tangent")
            ax.plot(x, values("pressure_slope_grid_pa_per_strain") / 1e8, color="#8c6db0", ls=":", label="Grid finite difference")
        axes[1, 0].set(title="Local pressure slope and shell contributions", ylabel=r"$dp_g/d\varepsilon_\theta$ (MPa / % strain)")
        slopes = values("pressure_slope_implicit_pa_per_strain") / 1e8
        upper = max(float(np.percentile(slopes, 95)) * 1.15, 0.1)
        lower = min(float(np.min(slopes)) * 1.15, 0.0)
        axes[1, 1].set(title="Same slopes: magnified vertical scale (high values clipped)", ylabel="Pressure slope (MPa / % strain)", ylim=(lower, upper))
        for label, style in (("total", "-"), ("magnitude", "--"), ("direction", ":")):
            axes[2, 0].plot(x, values(f"inner_{label}_pressure_slope_pa_per_strain") / 1e8, ls=style, label="Inner " + label)
        axes[2, 0].set(title="Inner contribution: intensity magnitude + direction", ylabel="Pressure slope contribution (MPa / % strain)")
        axes[2, 1].set(title="Current tensile-diagram plastic strain (not history)", ylabel="Mapped uniaxial plastic strain (%)")
        for ax in axes.flat:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        for ax in axes[2]:
            ax.set_xlabel("Common total hoop strain (%)")
        figure.suptitle("Feodosyev physical diagnostics at fixed temperatures and k\nCurrent diagram state; no A-D assignment or load-limit inference")
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(path, format="png")
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    finally:
        figure.clear()


def export_diagnostics(result: DiagnosticResult, directory: Path, prefix: str) -> tuple[Path, Path, Path]:
    directory = Path(directory)
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be a filename without a path")
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / f"{prefix}.{extension}" for extension in ("csv", "json", "png"))
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite diagnostic outputs")
    # Each exclusive open owns only its newly created output.
    with paths[0].open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)
    with paths[1].open("x", encoding="utf-8") as stream:
        json.dump(result.summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    write_diagnostic_plot(result, paths[2])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument("--inner-thickness-mm", type=float, default=1.0)
    parser.add_argument("--outer-thickness-mm", type=float, default=2.0)
    parser.add_argument("--inner-temperature-c", type=float, default=538.0)
    parser.add_argument("--outer-temperature-c", type=float, default=20.0)
    parser.add_argument("--start-hoop-strain", type=float, default=0.00025)
    parser.add_argument("--end-hoop-strain", type=float, default=0.03)
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--initial-axial-strain-guess", type=float)
    parser.add_argument("--root-scan-points", type=int, default=401)
    parser.add_argument("--tensile-tangent-step", type=float, default=1e-7)
    parser.add_argument("--output-directory", type=Path, default=Path("outputs"))
    parser.add_argument("--prefix", default="feodosyev_physical_diagnostics")
    args = parser.parse_args()
    from interpolation import thermal_strain

    inner_law = GrantaSS316TensionLaw(args.inner_temperature_c)
    outer_law = GrantaSS316TensionLaw(args.outer_temperature_c)
    curve = solve_uniform_curve(
        start_hoop_strain=args.start_hoop_strain, end_hoop_strain=args.end_hoop_strain,
        point_count=args.points, radius_m=args.radius_mm * 1e-3,
        inner_thickness_m=args.inner_thickness_mm * 1e-3,
        outer_thickness_m=args.outer_thickness_mm * 1e-3,
        inner_thermal_strain=thermal_strain(args.inner_temperature_c),
        outer_thermal_strain=thermal_strain(args.outer_temperature_c),
        inner_material=inner_law, outer_material=outer_law, k=args.k,
        initial_axial_strain_guess=args.initial_axial_strain_guess,
        root_scan_points=args.root_scan_points,
    )
    result = diagnose_curve(
        curve, MaterialReference(inner_law, inner_law.yield_stress_pa, args.inner_temperature_c),
        MaterialReference(outer_law, outer_law.yield_stress_pa, args.outer_temperature_c),
        tensile_tangent_step=args.tensile_tangent_step,
    )
    diagnostic_paths = export_diagnostics(result, args.output_directory, args.prefix)
    curve_paths = export_curve(curve, args.output_directory, args.prefix + "_pressure_curve")
    print(json.dumps({"summary": result.summary, "outputs": [str(p.resolve()) for p in (*diagnostic_paths, *curve_paths)]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

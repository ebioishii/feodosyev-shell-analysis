"""Produce local Phase 3A figures from synthetic data or a specified dataset.

Run from project root: uv run --locked python examples/phase3a_demo.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feodosyev_shell.materials import load_material
from feodosyev_shell.solvers import LinearShellCase, solve_linear_shell

ROOT = Path(__file__).resolve().parents[1]


def _dataset_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.csv")) + [directory / "material.json"]
    }


def _git(*args: str) -> str | None:
    process = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3A synthetic double-shell example")
    parser.add_argument("--material", type=Path, default=ROOT / "examples" / "synthetic_shell")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "phase3a_demo")
    parser.add_argument("--inner-temperature-k", type=float, default=400)
    parser.add_argument("--outer-temperature-k", type=float, default=300)
    parser.add_argument("--reference-temperature-k", type=float, default=300)
    parser.add_argument("--pressure-pa", type=float, default=1e6)
    arguments = parser.parse_args()
    material = load_material(arguments.material)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    case = LinearShellCase(
        radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
        inner_temperature_k=arguments.inner_temperature_k,
        outer_temperature_k=arguments.outer_temperature_k,
        stress_free_temperature_k=arguments.reference_temperature_k,
        gas_pressure_pa=arguments.pressure_pa,
    )
    pressures = np.linspace(0, case.gas_pressure_pa, 51)
    results = [solve_linear_shell(replace(case, gas_pressure_pa=float(p)), material)
               for p in pressures]
    columns = [
        "pressure_pa", "radial_displacement_m", "inner_stress_pa", "outer_stress_pa",
        "inner_thermal_strain", "outer_thermal_strain", "inner_elastic_status",
        "outer_elastic_status", "elastic_prediction_credible",
    ]
    with (output / "pressure_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for pressure, result in zip(pressures, results, strict=True):
            writer.writerow([
                pressure, result.radial_displacement_m, result.inner.hoop_stress_pa,
                result.outer.hoop_stress_pa, result.inner.thermal_hoop_strain,
                result.outer.thermal_hoop_strain, result.inner.elastic_status,
                result.outer.elastic_status, result.elastic_prediction_credible,
            ])

    fig, axis = plt.subplots(figsize=(7, 4.5))
    valid = np.array([r.elastic_prediction_credible is True for r in results])
    displacements = np.array([r.radial_displacement_m * 1e3 for r in results])
    axis.plot(pressures / 1e6, displacements, color="0.75", linestyle="--",
              label="Linear-elastic equation")
    if valid.any():
        axis.plot(pressures / 1e6, np.where(valid, displacements, np.nan),
                  color="C0", marker=".",
                  label="Within selected elastic limit")
    if not valid.all():
        axis.text(0.02, 0.98, "Grey dashed: elastic limit exceeded or unknown",
                  transform=axis.transAxes, va="top", fontsize=8)
    axis.set(xlabel="Gas pressure (MPa)", ylabel="Radial displacement (mm)",
             title=f"Phase 3A: pressure–displacement ({material.metadata['material']})")
    axis.grid(True)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "pressure_displacement.png", dpi=160)
    plt.close(fig)

    target = results[-1]
    fig, (stress_axis, thermal_axis) = plt.subplots(1, 2, figsize=(9, 4.5))
    names = ["Inner", "Outer"]
    stress_axis.bar(names, [target.inner.hoop_stress_pa / 1e6,
                            target.outer.hoop_stress_pa / 1e6])
    stress_axis.set(ylabel="Hoop stress (MPa)",
                    title=f"Hoop stress at {case.gas_pressure_pa / 1e6:g} MPa")
    thermal_axis.bar(names, [target.inner.thermal_hoop_strain * 100,
                             target.outer.thermal_hoop_strain * 100])
    thermal_axis.set(ylabel="Free thermal strain (%)",
                     title=f"Relative to {case.stress_free_temperature_k:g} K")
    fig.suptitle(
        f"Phase 3A: {material.metadata['material']} | "
        f"elastic range: {target.elastic_prediction_credible}"
    )
    fig.tight_layout()
    fig.savefig(output / "shell_comparison.png", dpi=160)
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "phase3a_hoop_only_linear_elastic_v1",
        "sources": ["Feodosyev Chapter 9 section 9.3, equations 9.1 and 9.3",
                    "Korotkaya (2017) section 2.1.1, equations 2.2–2.5"],
        "assumptions_document": "docs/phase3a_plan.md",
        "case": asdict(case),
        "material": material.metadata["material"],
        "dataset_version": material.metadata["dataset_version"],
        "source_status": material.metadata.get("source_status", "unknown"),
        "material_sha256": _dataset_hashes(material.root),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "units": {"temperature": "K", "pressure_and_stress": "Pa", "length": "m",
                  "strain": "1"},
        "interpolation": (
            "piecewise linear E(T) and accumulated mean-expansion beta(T); no extrapolation"
        ),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "pandas", "matplotlib")},
        "elastic_limit_interpretation": (
            "Tabulated yield strength is an indicative threshold only; exceeding it invalidates "
            "the linear-elastic prediction, not proof of failure."
        ),
        "target_elastic_prediction_credible": target.elastic_prediction_credible,
        "target_inner_elastic_status": target.inner.elastic_status,
        "target_outer_elastic_status": target.outer.elastic_status,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved Phase 3A results in {output}")
    print(
        f"At {case.gas_pressure_pa / 1e6:g} MPa: "
        f"radial displacement = {target.radial_displacement_m * 1e3:.6g} mm"
    )
    print(f"Within selected elastic limit: {target.elastic_prediction_credible}")


if __name__ == "__main__":
    main()

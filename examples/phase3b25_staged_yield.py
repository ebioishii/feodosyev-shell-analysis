"""Public synthetic heating/pressure path; this is not an SS316 prediction."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feodosyev_shell.constitutive.bilinear_j2 import SyntheticBilinearJ2
from feodosyev_shell.materials import load_material
from feodosyev_shell.solvers import (
    BiaxialLinearCase,
    IncrementalShellState,
    SyntheticShellLayer,
    solve_incremental_shell,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "synthetic_shell"
OUTPUT = ROOT / "outputs" / "phase3b25_staged_yield"
INNER_YIELD_PA = 200e6
OUTER_YIELD_PA = 60e6
PRESSURE_STEP_PA = 0.25e6
MAX_PRESSURE_PA = 6e6


def synthetic_layers() -> tuple[SyntheticShellLayer, SyntheticShellLayer]:
    """Explicitly override only synthetic yield thresholds; keep tables consistent."""
    base = load_material(FIXTURE)
    layers = []
    for label, yield_pa in (("inner", INNER_YIELD_PA), ("outer", OUTER_YIELD_PA)):
        properties = {
            **base.properties,
            "yield_strength": (
                np.array([300.0, 400.0]), np.array([yield_pa, yield_pa])
            ),
        }
        material = replace(
            base, properties=properties,
            metadata={**base.metadata, "material": f"SYNTHETIC_{label.upper()}_NOT_SS316",
                      "dataset_version": "synthetic-staged-yield-1",
                      "source_note": f"Artificial {label} yield threshold: {yield_pa:g} Pa"},
        )
        layers.append(SyntheticShellLayer(
            material, SyntheticBilinearJ2(100e9, 0.3, yield_pa, 10e9)
        ))
    return layers[0], layers[1]


def run_path(pressure_step_pa: float = PRESSURE_STEP_PA) -> list[dict[str, float | str | int]]:
    """Heat at zero pressure, then pressurize; pass each accepted state onward."""
    if not math.isfinite(pressure_step_pa) or pressure_step_pa <= 0:
        raise ValueError("Pressure step must be finite and positive")
    inner, outer = synthetic_layers()
    state = IncrementalShellState()
    rows = []
    heating = [(temperature, 0.0) for temperature in (300, 325, 350, 375, 400)]
    # All step-size comparisons must end at the same pressure. Shorten the final
    # increment when the requested step does not divide the pressure range.
    pressures = np.append(
        np.arange(pressure_step_pa, MAX_PRESSURE_PA, pressure_step_pa), MAX_PRESSURE_PA
    )
    loading = [(400, float(p)) for p in pressures]
    for index, (inner_temperature, pressure) in enumerate(heating + loading):
        case = BiaxialLinearCase(
            0.1, 0.001, 0.002, inner_temperature, 300, 300, pressure, 0.5,
        )
        result = solve_incremental_shell(case, inner, state, outer)
        state = result.state
        inner_q = state.inner.equivalent_plastic_strain
        outer_q = state.outer.equivalent_plastic_strain
        stage = "both_plastic" if inner_q > 0 else "outer_plastic" if outer_q > 0 else "elastic"
        rows.append({
            "step": index, "path_segment": "heating" if pressure == 0 else "pressure",
            "inner_temperature_k": inner_temperature, "outer_temperature_k": 300,
            "pressure_pa": pressure, "radial_displacement_m": result.radial_displacement_m,
            "inner_axial_stress_pa": result.inner.axial_stress_pa,
            "inner_hoop_stress_pa": result.inner.hoop_stress_pa,
            "outer_axial_stress_pa": result.outer.axial_stress_pa,
            "outer_hoop_stress_pa": result.outer.hoop_stress_pa,
            "inner_equivalent_plastic_strain": inner_q,
            "outer_equivalent_plastic_strain": outer_q,
            "hoop_balance_residual_n_per_m": result.hoop_balance_residual_n_per_m,
            "axial_balance_residual_n_per_m": result.axial_balance_residual_n_per_m,
            "stage": stage,
        })
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = run_path()
    csv_path = OUTPUT / "staged_path.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Include the heated zero-pressure state so the plot shows the full pressure path.
    pressure_rows = [rows[4], *(row for row in rows if row["path_segment"] == "pressure")]
    pressure_mpa = [float(row["pressure_pa"]) / 1e6 for row in pressure_rows]
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(pressure_mpa, [float(row["radial_displacement_m"]) * 1e3
                                for row in pressure_rows], color="black")
    axes[0].set_ylabel("Radial displacement (mm)")
    axes[1].plot(pressure_mpa, [float(row["inner_equivalent_plastic_strain"])
                                for row in pressure_rows], label="Inner shell")
    axes[1].plot(pressure_mpa, [float(row["outer_equivalent_plastic_strain"])
                                for row in pressure_rows], label="Outer shell")
    axes[1].set(xlabel="Gas pressure (MPa)", ylabel="Equivalent plastic strain")
    for stage, label, color in (("outer_plastic", "Outer first", "tab:orange"),
                                ("both_plastic", "Both plastic", "tab:blue")):
        first = next(row for row in pressure_rows if row["stage"] == stage)
        for axis in axes:
            axis.axvline(float(first["pressure_pa"]) / 1e6, color=color,
                         linestyle="--", alpha=0.65)
        axes[0].text(float(first["pressure_pa"]) / 1e6 + 0.04,
                     0.9 if stage == "outer_plastic" else 0.72, label,
                     transform=axes[0].get_xaxis_transform(), color=color)
    for axis in axes:
        axis.grid(True)
    axes[1].legend()
    fig.suptitle("Synthetic staged yielding after thermal restraint")
    fig.tight_layout()
    fig.savefig(OUTPUT / "staged_yield.png", dpi=160)
    plt.close(fig)

    commit = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    dirty = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
         "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=False,
    )
    manifest = {
        "model": "synthetic_bilinear_j2_plane_stress_incremental_v1",
        "source_status": "synthetic",
        "fixture_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(FIXTURE.iterdir()) if path.is_file()},
        "inner_law": asdict(synthetic_layers()[0].law),
        "outer_law": asdict(synthetic_layers()[1].law),
        "geometry_m": {"radius": 0.1, "inner_thickness": 0.001, "outer_thickness": 0.002},
        "axial_force_ratio": 0.5,
        "units": {"temperature": "K", "pressure_and_stress": "Pa", "length": "m",
                  "strain": "1"},
        "reference_temperature_k": 300,
        "outer_temperature_k": 300,
        "inner_heating_temperatures_k": [300, 325, 350, 375, 400],
        "pressure_step_pa": PRESSURE_STEP_PA,
        "max_pressure_pa": MAX_PRESSURE_PA,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_files_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "python_version": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "pandas", "scipy", "matplotlib")},
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for stage in ("outer_plastic", "both_plastic"):
        first = next(row for row in pressure_rows if row["stage"] == stage)
        print(f"First {stage}: {float(first['pressure_pa']) / 1e6:g} MPa")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

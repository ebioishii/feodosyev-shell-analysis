"""Synthetic 3B-2 pressure load/unload example; never uses SS316 plastic data."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
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
MATERIAL_DIR = ROOT / "examples" / "synthetic_shell"
OUTPUT = ROOT / "outputs" / "phase3b_synthetic_cycle"


def main() -> None:
    material = load_material(MATERIAL_DIR)
    law = SyntheticBilinearJ2(100e9, 0.3, 300e6, 10e9)
    layer = SyntheticShellLayer(material, law)
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    loading = np.linspace(0, 12e6, 25)
    unloading = np.linspace(12e6, 0, 25)[1:]
    pressures = np.concatenate([loading, unloading])
    state = IncrementalShellState()
    rows = []
    for index, pressure in enumerate(pressures):
        case = BiaxialLinearCase(
            radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
            inner_temperature_k=300, outer_temperature_k=300,
            stress_free_temperature_k=300, gas_pressure_pa=float(pressure),
            axial_force_ratio=0.5,
        )
        result = solve_incremental_shell(case, layer, state)
        state = result.state
        rows.append({
            "step": index,
            "pressure_pa": float(pressure),
            "radial_displacement_m": result.radial_displacement_m,
            "axial_stress_pa": result.inner.axial_stress_pa,
            "hoop_stress_pa": result.inner.hoop_stress_pa,
            "equivalent_plastic_strain": state.inner.equivalent_plastic_strain,
            "hoop_balance_residual_n_per_m": result.hoop_balance_residual_n_per_m,
            "axial_balance_residual_n_per_m": result.axial_balance_residual_n_per_m,
            "global_evaluations": result.global_evaluations,
        })
    with (output / "pressure_cycle.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    midpoint = len(loading)
    axis.plot([r["pressure_pa"] / 1e6 for r in rows[:midpoint]],
              [r["radial_displacement_m"] * 1e3 for r in rows[:midpoint]],
              label="Loading", marker=".")
    axis.plot([r["pressure_pa"] / 1e6 for r in rows[midpoint - 1:]],
              [r["radial_displacement_m"] * 1e3 for r in rows[midpoint - 1:]],
              label="Unloading", marker=".")
    axis.set(xlabel="Gas pressure (MPa)", ylabel="Radial displacement (mm)",
             title="Synthetic bilinear J2 two-shell cycle")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "pressure_displacement_cycle.png", dpi=160)
    plt.close(fig)

    material_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(MATERIAL_DIR.iterdir()) if path.is_file()
    }
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "synthetic_bilinear_j2_plane_stress_incremental_v1",
        "material": material.metadata["material"],
        "material_source_status": material.metadata["source_status"],
        "material_sha256": material_hashes,
        "law": asdict(law),
        "pressure_path_pa": pressures.tolist(),
        "geometry_m": {"radius": 0.1, "inner_thickness": 0.001,
                       "outer_thickness": 0.002},
        "temperature_k": 300,
        "axial_force_ratio": 0.5,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "units": {"pressure_and_stress": "Pa", "length": "m", "strain": "1",
                  "temperature": "K"},
        "limitation": (
            "Synthetic constant properties only; no SS316 plastic calibration or life claim"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote synthetic cycle to {output}")
    print(f"Residual radial displacement: {rows[-1]['radial_displacement_m'] * 1e3:.6g} mm")


if __name__ == "__main__":
    main()

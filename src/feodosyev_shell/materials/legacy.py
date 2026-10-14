"""Explicit SS316 CSV mapping; provisional image traces remain audit-only.

These source files are not copied into the tracked examples directory.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feodosyev_shell.materials.dataset import load_material

LEGACY_FILES = {
    "youngs_modulus": ("SS316_Youngs_Modulus.csv", "Youngs_Modulus_Pa", "Pa"),
    "yield_strength": ("SS316_yield_strength.csv", "Yield_Strength_Pa", "Pa"),
    "poisson_ratio": ("SS316_Poissons_ratio.csv", "Poisson's Ratio", "1"),
}
THERMAL_FILE = "SS316_mean_thermal_expansion.csv"
PLASTIC_FILE = "SS316_plastic_StressStrain.csv"
PLASTIC_COLUMNS = [
    "temperature_C", "engineering_strain", "engineering_stress_MPa",
    "source_status", "trace_quality", "source", "source_doi", "pixel_x", "pixel_y",
]


def inspect_provisional_curves(path: str | Path) -> dict:
    """Summarize digitized points without converting them into model curves."""
    source = Path(path)
    frame = pd.read_csv(source)
    if list(frame.columns) != PLASTIC_COLUMNS or frame.empty:
        raise ValueError(f"Unexpected columns or empty provisional curve file: {source.name}")
    numeric_columns = [
        "temperature_C", "engineering_strain", "engineering_stress_MPa", "pixel_x", "pixel_y",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Provisional curve file contains nonfinite numbers")
    if (frame["engineering_strain"] < 0).any() or (frame["engineering_stress_MPa"] < 0).any():
        raise ValueError("Provisional curve file contains negative tensile values")
    if frame[["source_status", "trace_quality", "source", "source_doi"]].isna().any().any():
        raise ValueError("Provisional curve file has missing source or quality labels")
    groups = []
    for temperature, group in frame.groupby("temperature_C", sort=True):
        groups.append({
            "temperature_c": float(temperature), "rows": int(len(group)),
            "duplicate_strain_rows": int(group.duplicated("engineering_strain").sum()),
            "minimum_strain": float(group["engineering_strain"].min()),
            "maximum_strain": float(group["engineering_strain"].max()),
        })
    return {
        "file": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "curve_kind": "engineering",
        "source_status_counts": frame["source_status"].value_counts().to_dict(),
        "trace_quality_counts": frame["trace_quality"].value_counts().to_dict(),
        "source_labels": sorted(frame["source"].unique().tolist()),
        "source_dois": sorted(frame["source_doi"].unique().tolist()),
        "by_temperature": groups,
        "imported_as_model_curves": False,
        "reason": (
            "Provisional digitization has repeated strain coordinates and incomplete "
            "initial curves; selecting or averaging points needs source validation."
        ),
    }


def import_legacy_ss316(
    source_dir: str | Path, output_dir: str | Path, *, source_note: str = "unknown"
):
    """Convert known columns into schema v1; preserve uncertain provenance as unknown."""
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    properties = []
    for name, (filename, value_column, unit) in LEGACY_FILES.items():
        frame = pd.read_csv(source / filename, skipinitialspace=True)
        expected = ["Temperature_C", value_column]
        if list(frame.columns) != expected:
            raise ValueError(f"Unexpected columns in {filename}: {list(frame.columns)}")
        normalized = pd.DataFrame({
            "temperature_k": pd.to_numeric(frame["Temperature_C"], errors="raise") + 273.15,
            "value": pd.to_numeric(frame[value_column], errors="raise"),
        })
        target = f"{name}.csv"
        normalized.to_csv(output / target, index=False)
        properties.append({"name": name, "file": target, "unit": unit,
                           "source_file": filename, "source_location": "unknown"})
    thermal_path = source / THERMAL_FILE
    if thermal_path.exists():
        thermal = pd.read_csv(thermal_path)
        expected = ["reference_temperature_C", "temperature_C", "mean_alpha_per_K"]
        if list(thermal.columns) != expected or thermal.empty:
            raise ValueError(f"Unexpected columns or empty thermal expansion file: {THERMAL_FILE}")
        for column in expected:
            thermal[column] = pd.to_numeric(thermal[column], errors="raise")
        if not np.isfinite(thermal.to_numpy(dtype=float)).all():
            raise ValueError("Thermal expansion file contains nonfinite values")
        references = thermal["reference_temperature_C"].unique()
        if len(references) != 1:
            raise ValueError("Thermal expansion rows must share one reference temperature")
        normalized = pd.DataFrame({
            "temperature_k": thermal["temperature_C"] + 273.15,
            "value": thermal["mean_alpha_per_K"],
        })
        normalized.to_csv(output / "thermal_expansion.csv", index=False)
        properties.append({
            "name": "thermal_expansion", "file": "thermal_expansion.csv", "unit": "1/K",
            "coefficient_type": "mean", "reference_temperature_k": float(references[0] + 273.15),
            "source_file": THERMAL_FILE, "source_location": "unknown",
        })
    provisional_path = source / PLASTIC_FILE
    curve_audit = (
        inspect_provisional_curves(provisional_path) if provisional_path.exists() else None
    )
    metadata = {
        "schema_version": 1, "dataset_version": "local-import-1", "material": "SS316",
        "condition": "unknown", "source_status": "unverified", "source_note": source_note,
        "test_conditions": "unknown", "processing": "C to K; original values retained",
        "redistribution_permission": "unknown", "properties": properties, "curves": [],
    }
    if curve_audit is not None:
        metadata["provisional_curve_audit"] = curve_audit
    (output / "material.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return load_material(output)

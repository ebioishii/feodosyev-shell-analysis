"""Explicit mapping for three pre-existing SS316 property CSV files.

These source files are not copied into the tracked examples directory.
"""

from pathlib import Path

import pandas as pd

from feodosyev_shell.materials.dataset import load_material

LEGACY_FILES = {
    "youngs_modulus": ("SS316_Youngs_Modulus.csv", "Youngs_Modulus_Pa", "Pa"),
    "yield_strength": ("SS316_yield_strength.csv", "Yield_Strength_Pa", "Pa"),
    "poisson_ratio": ("SS316_Poissons_ratio.csv", "Poisson's Ratio", "1"),
}


def import_legacy_ss316(
    source_dir: str | Path, output_dir: str | Path, *, source_note: str = "unknown"
):
    """Convert known columns into schema v1; preserve uncertain provenance as unknown."""
    import json

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
    metadata = {
        "schema_version": 1, "dataset_version": "local-import-1", "material": "SS316",
        "condition": "unknown", "source_status": "unverified", "source_note": source_note,
        "test_conditions": "unknown", "processing": "C to K; original values retained",
        "redistribution_permission": "unknown", "properties": properties, "curves": [],
    }
    (output / "material.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return load_material(output)

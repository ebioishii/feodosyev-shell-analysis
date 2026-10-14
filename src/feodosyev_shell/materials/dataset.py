"""Validated, unit-normalized material tables.

Schema version 1 stores temperature in K, stress and modulus in Pa, and
dimensionless strain. Each table has its own traceable metadata entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CURVE_KINDS = {"engineering", "true_total", "true_plastic"}
PROPERTY_NAMES = {"youngs_modulus", "yield_strength", "poisson_ratio", "thermal_expansion"}


def _finite_number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _table_path(root: Path, filename: str) -> Path:
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Data file must stay inside the dataset directory")
    if candidate.suffix.lower() != ".csv":
        raise ValueError("Data table must be CSV")
    return candidate


def _read_numeric(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    table = pd.read_csv(path)
    if list(table.columns) != list(columns) or table.empty:
        raise ValueError(f"{path.name} must have nonempty columns {columns}")
    for column in columns:
        try:
            table[column] = pd.to_numeric(table[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: nonnumeric {column}") from exc
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: nonfinite value")
    return table


@dataclass(frozen=True)
class Material:
    root: Path
    metadata: dict
    properties: dict[str, tuple[np.ndarray, np.ndarray]]
    curves: dict[float, tuple[np.ndarray, np.ndarray]]

    @property
    def curve_kind(self) -> str | None:
        return self.metadata.get("curve_kind")

    def property_at(self, name: str, temperature_k: float) -> float:
        if name not in self.properties:
            raise ValueError(f"Material property unavailable: {name}")
        temperature = _finite_number(temperature_k, "temperature_k")
        x, y = self.properties[name]
        if not x[0] <= temperature <= x[-1]:
            raise ValueError(f"{name}: temperature outside [{x[0]}, {x[-1]}] K")
        return float(np.interp(temperature, x, y))

    def completeness_report(self) -> dict:
        expected = ["youngs_modulus", "yield_strength", "poisson_ratio", "thermal_expansion"]
        missing = [name for name in expected if name not in self.properties]
        thermal_entry = next(
            (entry for entry in self.metadata.get("properties", [])
             if entry["name"] == "thermal_expansion"),
            None,
        )
        return {
            "material": self.metadata["material"],
            "available_properties": sorted(self.properties),
            "missing_properties": missing,
            "curve_temperatures_k": sorted(self.curves),
            "full_stress_strain_curves_available": bool(self.curves),
            "source_status": self.metadata.get("source_status", "unknown"),
            "condition": self.metadata.get("condition", "unknown"),
            "thermal_expansion_type": (
                thermal_entry.get("coefficient_type") if thermal_entry else None
            ),
            "thermal_reference_temperature_k": (
                thermal_entry.get("reference_temperature_k") if thermal_entry else None
            ),
            "provisional_curve_data_available": "provisional_curve_audit" in self.metadata,
            "provisional_curve_reason": self.metadata.get("provisional_curve_audit", {}).get(
                "reason"
            ),
        }


def load_material(directory: str | Path) -> Material:
    root = Path(directory).resolve()
    metadata = json.loads((root / "material.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or not metadata.get("material"):
        raise ValueError("Unsupported schema version or missing material name")
    if not metadata.get("dataset_version") or not metadata.get("condition"):
        raise ValueError("Dataset version and material condition are required")
    properties = {}
    for entry in metadata.get("properties", []):
        name = entry["name"]
        if name not in PROPERTY_NAMES or name in properties:
            raise ValueError(f"Unknown or duplicate property: {name}")
        unit = {"poisson_ratio": "1", "thermal_expansion": "1/K"}.get(name, "Pa")
        if entry.get("unit") != unit:
            raise ValueError(f"Incorrect internal unit for {name}")
        if name == "thermal_expansion":
            if entry.get("coefficient_type") not in {"mean", "instantaneous"}:
                raise ValueError("Thermal expansion coefficient type required")
            _finite_number(entry.get("reference_temperature_k"), "reference_temperature_k")
        table = _read_numeric(_table_path(root, entry["file"]), ("temperature_k", "value"))
        x = table.temperature_k.to_numpy(dtype=float)
        y = table.value.to_numpy(dtype=float)
        if np.any(x <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError(f"{name}: temperatures must be positive and strictly increasing")
        if name in {"youngs_modulus", "yield_strength"} and np.any(y <= 0):
            raise ValueError(f"{name}: values must be positive")
        if name == "poisson_ratio" and (np.any(y <= -1) or np.any(y >= 0.5)):
            raise ValueError("Poisson ratio outside isotropic stable range")
        properties[name] = (x, y)
    curves = {}
    curve_entries = metadata.get("curves", [])
    if curve_entries and metadata.get("curve_kind") not in CURVE_KINDS:
        raise ValueError("Unknown stress-strain curve kind")
    for entry in curve_entries:
        temperature = _finite_number(entry["temperature_k"], "curve temperature_k")
        if temperature <= 0 or temperature in curves:
            raise ValueError("Curve temperatures must be unique and positive")
        table = _read_numeric(_table_path(root, entry["file"]), ("strain", "stress_pa"))
        strain = table.strain.to_numpy(dtype=float)
        stress = table.stress_pa.to_numpy(dtype=float)
        if len(strain) < 2 or np.any(strain < 0) or np.any(np.diff(strain) <= 0):
            raise ValueError("Curve strain must be nonnegative and strictly increasing")
        if np.any(stress < 0):
            raise ValueError("Curve stress must be nonnegative")
        curves[temperature] = (strain, stress)
    return Material(root, metadata, properties, curves)

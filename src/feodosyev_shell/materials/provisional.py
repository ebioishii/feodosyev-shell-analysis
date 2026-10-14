"""Prepare partial engineering curves from a provisional digitized CSV.

Selection is a software policy, not verification of the source figure. No
averaging, smoothing, initial elastic reconstruction or necking inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feodosyev_shell.materials.dataset import Material, load_material
from feodosyev_shell.materials.legacy import inspect_provisional_curves

POLICY = "visible-unique-contiguous-v1"


def prepare_provisional_ss316(source_file: str | Path, output_dir: str | Path) -> Material:
    """Create a separate, repeatable dataset; preserve every selection decision.

    Only visible_curve points at unique strain coordinates are eligible. Runs
    split at every excluded point and require at least two nodes. All sources
    and DOI labels must agree. Existing outputs may only be refreshed from the
    same source hash and selection-policy version.
    """
    source, output = Path(source_file).resolve(), Path(output_dir).resolve()
    audit = inspect_provisional_curves(source)
    frame = pd.read_csv(source)
    if (frame.source.nunique() != 1 or frame.source_doi.nunique() != 1
            or not (frame.source_status == "digitized_provisional").all()):
        raise ValueError("Preparation requires one source and DOI with provisional status")
    if not set(frame.trace_quality).issubset(
        {"visible_curve", "visible_steep_segment", "bridged_occlusion"}
    ):
        raise ValueError("Unknown trace quality label")
    if (frame.temperature_C <= -273.15).any():
        raise ValueError("Temperature must be above absolute zero")
    if output == source.parent or source.is_relative_to(output):
        raise ValueError("Processed output must be separate from the raw source directory")
    if output.exists() and any(output.iterdir()):
        meta_file = output / "material.json"
        previous = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        if (previous.get("source_sha256") != audit["sha256"]
                or previous.get("selection_policy") != POLICY):
            raise ValueError("Output contains a different dataset; choose a new output directory")

    frame["source_csv_row"] = np.arange(2, len(frame) + 2)
    frame["selection_reason"] = "accepted_visible_unique"
    frame.loc[frame.trace_quality != "visible_curve", "selection_reason"] = "excluded_quality"
    repeated = frame.duplicated(["temperature_C", "engineering_strain"], keep=False)
    frame.loc[repeated, "selection_reason"] = "excluded_duplicate_strain"
    frame["segment_id"] = -1
    entries, tables = [], {}
    for index, (temperature, group) in enumerate(frame.groupby("temperature_C", sort=True)):
        group = group.sort_values("engineering_strain", kind="stable")
        runs, run = [], []
        for row in group.index:
            if frame.at[row, "selection_reason"] == "accepted_visible_unique":
                run.append(row)
            else:
                if run:
                    runs.append(run)
                run = []
        if run:
            runs.append(run)
        kept, intervals = [], []
        for run in runs:
            if len(run) < 2:
                frame.loc[run, "selection_reason"] = "excluded_isolated_point"
                continue
            frame.loc[run, "segment_id"] = len(intervals)
            kept.extend(run)
            intervals.append([float(frame.at[run[0], "engineering_strain"]),
                              float(frame.at[run[-1], "engineering_strain"])])
        if not kept:
            raise ValueError(f"No eligible continuous segment at {temperature} C")
        filename = f"curve_{index:02d}.csv"
        tables[filename] = pd.DataFrame({
            "strain": frame.loc[kept, "engineering_strain"].to_numpy(),
            "stress_pa": frame.loc[kept, "engineering_stress_MPa"].to_numpy() * 1e6,
        })
        entries.append({
            "temperature_k": float(temperature + 273.15), "file": filename,
            "strain_intervals": intervals, "uniform_deformation": False,
            "source_location": str(frame.source.iloc[0]),
        })
    if not entries:
        raise ValueError("No contiguous visible unique segments with at least two points")
    metadata = {
        "schema_version": 1, "dataset_version": f"{POLICY}-{audit['sha256'][:12]}",
        "material": "SS316_PROVISIONAL", "condition": "unknown",
        "source_status": "digitized_provisional", "partial_curves": True,
        "source_sha256": audit["sha256"], "source_file": source.name,
        "source": str(frame.source.iloc[0]), "source_doi": str(frame.source_doi.iloc[0]),
        "selection_policy": POLICY, "redistribution_permission": "unknown",
        "test_conditions": "unknown; source figure not independently verified",
        "processing": "MPa to Pa; C to K; retain eligible contiguous segments; gaps unavailable",
        "limitations": ["No initial elastic reconstruction", "Uniform deformation unconfirmed",
                        "May include post-necking engineering data", "Exploratory use only"],
        "properties": [], "curve_kind": "engineering", "curves": entries,
        "auxiliary_files": ["selection_audit.csv"],
        "selection_counts": {
            str(k): int(v) for k, v in frame.selection_reason.value_counts().items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    frame.to_csv(output / "selection_audit.csv", index=False)
    (output / "material.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return load_material(output)

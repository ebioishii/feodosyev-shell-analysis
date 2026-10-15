"""Pre-screen evidence for converting measured engineering curves to plastic data.

This is an inventory of missing evidence, never an approval of a material law.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from feodosyev_shell.materials.dataset import Material

EVIDENCE_FIELDS = (
    "uniform_limit_status", "uniform_limit_source", "uniform_limit_definition",
    "uniform_strain_limit", "uniform_limit_uncertainty", "uniform_limit_confidence",
    "initial_segment_status", "initial_segment_source",
    "test_conditions_status", "test_conditions_source",
    "elastic_compatibility_status", "elastic_compatibility_source",
    "yield_definition", "yield_definition_source",
)


def evidence_template(material: Material) -> dict:
    """Create an unverified worksheet; nulls must be filled from actual evidence."""
    return {
        "dataset_version": material.metadata["dataset_version"],
        "source_sha256": material.metadata.get("source_sha256"),
        "temperatures": [
            {"temperature_k": temperature, **dict.fromkeys(EVIDENCE_FIELDS)}
            for temperature in sorted(material.curves)
        ],
    }


def _indexed_evidence(material: Material, evidence: Mapping | None) -> dict[float, dict]:
    if evidence is None:
        return {}
    if not isinstance(evidence, Mapping):
        raise ValueError("Evidence must be an object")
    if evidence.get("dataset_version") != material.metadata["dataset_version"]:
        raise ValueError("Evidence dataset_version does not match material")
    if evidence.get("source_sha256") != material.metadata.get("source_sha256"):
        raise ValueError("Evidence source_sha256 does not match material")
    records = evidence.get("temperatures")
    if not isinstance(records, list):
        raise ValueError("Evidence temperatures must be a list")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each temperature evidence record must be an object")
        try:
            temperature = float(record["temperature_k"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Evidence temperature_k must be numeric") from exc
        if not math.isfinite(temperature) or temperature not in material.curves:
            raise ValueError("Evidence temperature_k is not a material curve temperature")
        if temperature in indexed:
            raise ValueError("Duplicate evidence temperature_k")
        indexed[temperature] = record
    return indexed


def audit_true_plastic_readiness(material: Material, evidence: Mapping | None = None) -> dict:
    """List objective blockers; a clear pre-screen still needs human source review."""
    indexed = _indexed_evidence(material, evidence)
    rows = []
    for temperature, (strain, _) in sorted(material.curves.items()):
        record = indexed.get(temperature, {})
        blockers = []
        if material.curve_kind != "engineering":
            blockers.append("engineering_curve_definition_required")
        if material.metadata.get("source_status") in {
            "synthetic", "digitized_provisional", "unverified", "unknown", None
        }:
            blockers.append("source_experiment_not_verified")
        condition = material.metadata.get("condition")
        if (not isinstance(condition, str) or not condition.strip()
                or condition.lower() == "unknown"):
            blockers.append("material_condition_unknown")
        if strain[0] > 0:
            blockers.append("initial_zero_strain_region_missing")
        if (record.get("initial_segment_status") != "documented"
                or not record.get("initial_segment_source")):
            blockers.append("initial_segment_evidence_missing")
        if (record.get("test_conditions_status") != "documented"
                or not record.get("test_conditions_source")):
            blockers.append("test_conditions_evidence_missing")
        if (record.get("elastic_compatibility_status") != "documented"
                or not record.get("elastic_compatibility_source")):
            blockers.append("compatible_elastic_modulus_evidence_missing")
        try:
            modulus = material.property_at("youngs_modulus", temperature)
        except ValueError:
            modulus = None
        if modulus is None:
            blockers.append("elastic_modulus_value_missing_at_temperature")
        if not record.get("yield_definition") or not record.get("yield_definition_source"):
            blockers.append("yield_definition_evidence_missing")
        cutoff = record.get("uniform_strain_limit")
        if (record.get("uniform_limit_status") != "documented"
                or not record.get("uniform_limit_source")
                or not record.get("uniform_limit_definition")):
            blockers.append("uniform_limit_evidence_missing")
        if cutoff is not None:
            try:
                cutoff = float(cutoff)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid uniform_strain_limit at {temperature} K") from exc
            if not math.isfinite(cutoff) or cutoff <= 0:
                raise ValueError(f"Invalid uniform_strain_limit at {temperature} K")
            # A documented cutoff only helps when an uninterrupted measured
            # interval covers every strain from zero through that cutoff.
            spans = material.strain_intervals(temperature)
            if not (spans[0][0] == 0 and spans[0][1] >= cutoff):
                blockers.append("curve_gap_or_range_before_uniform_limit")
        else:
            blockers.append("uniform_limit_value_missing")
        rows.append({
            "temperature_k": temperature,
            "curve_first_strain": float(strain[0]),
            "curve_last_strain": float(strain[-1]),
            "valid_intervals": [list(span) for span in material.strain_intervals(temperature)],
            "uniform_strain_limit": cutoff,
            "blockers": blockers,
            "software_precheck": "blocked" if blockers else "manual_review_required",
        })
    return {
        "material": material.metadata["material"],
        "dataset_version": material.metadata["dataset_version"],
        "source_status": material.metadata.get("source_status", "unknown"),
        "physical_validation_status": "unverified",
        "all_software_prechecks_clear": bool(rows) and all(not row["blockers"] for row in rows),
        "temperatures": rows,
    }

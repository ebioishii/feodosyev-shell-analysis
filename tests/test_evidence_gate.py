"""Synthetic-only checks for the Phase 3B-3 evidence pre-screen."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from feodosyev_shell.materials import load_material
from feodosyev_shell.materials.evidence_gate import (
    audit_true_plastic_readiness,
    evidence_template,
)
from scripts.audit_phase3b3_evidence import main

SYNTHETIC = load_material(Path(__file__).resolve().parents[1] / "examples" / "synthetic")


def _filled_worksheet(material):
    worksheet = evidence_template(material)
    for record in worksheet["temperatures"]:
        record.update({
            "uniform_limit_status": "documented",
            "uniform_limit_source": "synthetic exercise record",
            "uniform_limit_definition": "synthetic engineering uniform elongation",
            "uniform_strain_limit": 0.004,
            "initial_segment_status": "documented",
            "initial_segment_source": "synthetic exercise record",
            "test_conditions_status": "documented",
            "test_conditions_source": "synthetic exercise record",
            "elastic_compatibility_status": "documented",
            "elastic_compatibility_source": "synthetic exercise record",
            "yield_definition": "explicit synthetic initial yield",
            "yield_definition_source": "synthetic exercise record",
        })
    return worksheet


def test_provisional_or_synthetic_sources_never_pass_software_prescreen():
    evidence = _filled_worksheet(SYNTHETIC)
    report = audit_true_plastic_readiness(SYNTHETIC, evidence)
    assert not report["all_software_prechecks_clear"]
    assert report["physical_validation_status"] == "unverified"
    assert all("source_experiment_not_verified" in row["blockers"]
               for row in report["temperatures"])


def test_documented_compatibility_does_not_substitute_for_modulus_values():
    candidate = replace(SYNTHETIC, properties={})
    report = audit_true_plastic_readiness(candidate, _filled_worksheet(candidate))
    assert all("elastic_modulus_value_missing_at_temperature" in row["blockers"]
               for row in report["temperatures"])


def test_even_complete_synthetic_simulation_needs_manual_physical_review():
    # This re-labels synthetic test data only to exercise the software gate.
    candidate = replace(SYNTHETIC, metadata={
        **SYNTHETIC.metadata, "source_status": "experimental_candidate",
        "condition": "documented synthetic exercise condition",
    })
    report = audit_true_plastic_readiness(candidate, _filled_worksheet(candidate))
    assert report["all_software_prechecks_clear"]
    assert all(row["software_precheck"] == "manual_review_required"
               for row in report["temperatures"])
    assert report["physical_validation_status"] == "unverified"


def test_gap_before_claimed_uniform_cutoff_blocks_curve():
    candidate = replace(SYNTHETIC, metadata={
        **SYNTHETIC.metadata,
        "source_status": "experimental_candidate",
        "condition": "documented synthetic exercise condition",
        "curves": [
            {**entry, "strain_intervals": [[0, 0.001], [0.003, 0.005]]}
            for entry in SYNTHETIC.metadata["curves"]
        ],
    })
    report = audit_true_plastic_readiness(candidate, _filled_worksheet(candidate))
    assert all("curve_gap_or_range_before_uniform_limit" in row["blockers"]
               for row in report["temperatures"])


def test_missing_initial_points_and_unverified_cutoff_block_conversion():
    shortened = {
        temperature: (strain[1:].copy(), stress[1:].copy())
        for temperature, (strain, stress) in SYNTHETIC.curves.items()
    }
    candidate = replace(SYNTHETIC, curves=shortened)
    report = audit_true_plastic_readiness(candidate)
    for row in report["temperatures"]:
        assert "initial_zero_strain_region_missing" in row["blockers"]
        assert "uniform_limit_evidence_missing" in row["blockers"]
        assert "uniform_limit_value_missing" in row["blockers"]


def test_evidence_must_match_dataset_and_curve_temperatures():
    evidence = _filled_worksheet(SYNTHETIC)
    evidence["dataset_version"] = "wrong-version"
    with pytest.raises(ValueError, match="dataset_version"):
        audit_true_plastic_readiness(SYNTHETIC, evidence)
    evidence = _filled_worksheet(SYNTHETIC)
    evidence["temperatures"].append({**evidence["temperatures"][0]})
    with pytest.raises(ValueError, match="Duplicate"):
        audit_true_plastic_readiness(SYNTHETIC, evidence)
    evidence = _filled_worksheet(SYNTHETIC)
    evidence["temperatures"][0]["uniform_strain_limit"] = np.nan
    with pytest.raises(ValueError, match="Invalid uniform_strain_limit"):
        audit_true_plastic_readiness(SYNTHETIC, evidence)


def test_rerunning_cli_preserves_researchers_existing_worksheet(tmp_path, monkeypatch):
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", [
        "audit_phase3b3_evidence.py", "--material", str(SYNTHETIC.root),
        "--output", str(output),
    ])
    main()
    worksheet = output / "evidence_template.json"
    entered = json.loads(worksheet.read_text(encoding="utf-8"))
    entered["temperatures"][0]["uniform_limit_source"] = "researcher note"
    worksheet.write_text(json.dumps(entered), encoding="utf-8")
    main()
    assert json.loads(worksheet.read_text(encoding="utf-8")) == entered

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from feodosyev_shell.materials import load_material
from feodosyev_shell.materials.legacy import import_legacy_ss316, inspect_provisional_curves

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic"


def test_properties_and_completeness():
    material = load_material(EXAMPLE)
    assert material.property_at("youngs_modulus", 350) == pytest.approx(190e9)
    assert material.completeness_report()["missing_properties"] == []
    with pytest.raises(ValueError, match="outside"):
        material.property_at("youngs_modulus", 500)


def test_duplicate_and_nonfinite_rows_rejected(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "material")
    path = tmp_path / "material" / "youngs_modulus.csv"
    path.write_text("temperature_k,value\n300,2e11\n300,NaN\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_material(path.parent)


def test_missing_property_only_blocks_that_property(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "material")
    path = tmp_path / "material" / "material.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["properties"] = [p for p in data["properties"] if p["name"] != "thermal_expansion"]
    path.write_text(json.dumps(data), encoding="utf-8")
    material = load_material(path.parent)
    assert "thermal_expansion" in material.completeness_report()["missing_properties"]
    assert np.isfinite(material.property_at("youngs_modulus", 350))
    with pytest.raises(ValueError, match="unavailable"):
        material.property_at("thermal_expansion", 350)


@pytest.mark.parametrize("case", ["nonzero_start", "gap", "provisional"])
def test_partial_curve_metadata_cannot_claim_full_curves(tmp_path, case):
    shutil.copytree(EXAMPLE, tmp_path / "material")
    root = tmp_path / "material"
    path = root / "material.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if case == "nonzero_start":
        for curve in metadata["curves"]:
            table = root / curve["file"]
            table.write_text("strain,stress_pa\n0.001,200000000\n0.003,240000000\n"
                             if curve["temperature_k"] == 300 else
                             "strain,stress_pa\n0.001,180000000\n0.003,210000000\n",
                             encoding="utf-8")
    elif case == "gap":
        metadata["curves"][0]["strain_intervals"] = [[0, 0.001], [0.003, 0.005]]
    else:
        metadata["source_status"] = "digitized_provisional"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="partial_curves"):
        load_material(root)
    metadata["partial_curves"] = True
    path.write_text(json.dumps(metadata), encoding="utf-8")
    report = load_material(root).completeness_report()
    assert report["curve_data_available"] is True
    assert report["full_stress_strain_curves_available"] is False


def test_legacy_mapping(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SS316_Youngs_Modulus.csv").write_text(
        "Temperature_C,Youngs_Modulus_Pa\n25,2e11\n125,1.8e11\n", encoding="utf-8")
    (source / "SS316_yield_strength.csv").write_text(
        "Temperature_C,Yield_Strength_Pa\n25,2e8\n125,1.8e8\n", encoding="utf-8")
    (source / "SS316_Poissons_ratio.csv").write_text(
        "Temperature_C,Poisson's Ratio\n25,0.3\n125,0.31\n", encoding="utf-8")
    material = import_legacy_ss316(source, tmp_path / "processed")
    assert material.property_at("youngs_modulus", 298.15) == pytest.approx(2e11)
    assert material.completeness_report()["source_status"] == "unverified"
    assert material.completeness_report()["full_stress_strain_curves_available"] is False


def test_new_thermal_table_and_provisional_curve_audit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SS316_Youngs_Modulus.csv").write_text(
        "Temperature_C,Youngs_Modulus_Pa\n100,2e11\n315,1.8e11\n", encoding="utf-8")
    (source / "SS316_yield_strength.csv").write_text(
        "Temperature_C,Yield_Strength_Pa\n100,2e8\n315,1.8e8\n", encoding="utf-8")
    (source / "SS316_Poissons_ratio.csv").write_text(
        "Temperature_C,Poisson's Ratio\n100,0.3\n315,0.31\n", encoding="utf-8")
    (source / "SS316_mean_thermal_expansion.csv").write_text(
        "reference_temperature_C,temperature_C,mean_alpha_per_K\n"
        "0,100,1.59e-5\n0,315,1.62e-5\n", encoding="utf-8")
    (source / "SS316_plastic_StressStrain.csv").write_text(
        "temperature_C,engineering_strain,engineering_stress_MPa,"
        "source_status,trace_quality,source,source_doi,pixel_x,pixel_y\n"
        "20,0.01,200,digitized_provisional,visible_curve,Figure 2,doi,10,20\n"
        "20,0.01,205,digitized_provisional,visible_curve,Figure 2,doi,10,19\n",
        encoding="utf-8",
    )
    material = import_legacy_ss316(source, tmp_path / "processed")
    report = material.completeness_report()
    assert report["missing_properties"] == []
    assert report["thermal_expansion_type"] == "mean"
    assert report["thermal_reference_temperature_k"] == 273.15
    assert material.property_at("thermal_expansion", 373.15) == pytest.approx(1.59e-5)
    assert material.metadata["properties"][-1]["coefficient_type"] == "mean"
    assert material.metadata["properties"][-1]["reference_temperature_k"] == 273.15
    assert report["provisional_curve_data_available"] is True
    assert report["full_stress_strain_curves_available"] is False
    audit = inspect_provisional_curves(source / "SS316_plastic_StressStrain.csv")
    assert audit["by_temperature"][0]["duplicate_strain_rows"] == 1
    assert audit["imported_as_model_curves"] is False

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from feodosyev_shell.materials import load_material
from feodosyev_shell.materials.legacy import import_legacy_ss316

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

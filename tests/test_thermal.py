import json
import shutil
from pathlib import Path

import pytest

from feodosyev_shell.constitutive import mechanical_hoop_strain
from feodosyev_shell.materials import (
    accumulated_thermal_strain,
    load_material,
    thermal_strain,
)
from feodosyev_shell.materials.legacy import import_legacy_ss316

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic"


def _mean_material(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    for filename, value_column, value in (
        ("SS316_Youngs_Modulus.csv", "Youngs_Modulus_Pa", 2e11),
        ("SS316_yield_strength.csv", "Yield_Strength_Pa", 2e8),
        ("SS316_Poissons_ratio.csv", "Poisson's Ratio", 0.3),
    ):
        (directory / filename).write_text(
            f"Temperature_C,{value_column}\n100,{value}\n315,{value}\n", encoding="utf-8"
        )
    (directory / "SS316_mean_thermal_expansion.csv").write_text(
        "reference_temperature_C,temperature_C,mean_alpha_per_K\n"
        "0,100,1.59e-5\n0,315,1.62e-5\n",
        encoding="utf-8",
    )
    return import_legacy_ss316(directory, tmp_path / "processed")


def test_beta_nodes_and_interpolation_before_reference_subtraction(tmp_path):
    material = _mean_material(tmp_path)
    beta_100 = 1.59e-5 * 100
    beta_315 = 1.62e-5 * 315
    beta_200 = beta_100 + (beta_315 - beta_100) * 100 / 215
    assert accumulated_thermal_strain(material, 273.15) == 0
    assert accumulated_thermal_strain(material, 373.15) == pytest.approx(beta_100)
    assert accumulated_thermal_strain(material, 588.15) == pytest.approx(beta_315)
    assert accumulated_thermal_strain(material, 473.15) == pytest.approx(beta_200)
    assert thermal_strain(material, 588.15, 373.15) == pytest.approx(beta_315 - beta_100)
    assert thermal_strain(material, 373.15, 588.15) == pytest.approx(beta_100 - beta_315)
    assert thermal_strain(material, 473.15, 473.15) == 0


def test_unsupported_temperature_and_coefficient_kind(tmp_path):
    material = _mean_material(tmp_path)
    with pytest.raises(ValueError, match="outside measured"):
        accumulated_thermal_strain(material, 293.15)
    with pytest.raises(ValueError, match="outside measured"):
        thermal_strain(material, 588.15, 293.15)
    with pytest.raises(ValueError, match="positive finite"):
        accumulated_thermal_strain(material, float("nan"))
    with pytest.raises(ValueError, match="mean expansion"):
        accumulated_thermal_strain(load_material(EXAMPLE), 300)


def test_missing_thermal_data_is_explicit(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "material")
    path = tmp_path / "material" / "material.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["properties"] = [
        entry for entry in metadata["properties"] if entry["name"] != "thermal_expansion"
    ]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        accumulated_thermal_strain(load_material(path.parent), 300)


def test_small_strain_shell_kinematics_only():
    assert mechanical_hoop_strain(0.0002, 0.1, 0.0015) == pytest.approx(0.0005)
    with pytest.raises(ValueError, match="positive"):
        mechanical_hoop_strain(0.0002, 0, 0.0015)
    with pytest.raises(ValueError, match="finite"):
        mechanical_hoop_strain(float("inf"), 0.1, 0.0015)

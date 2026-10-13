from pathlib import Path

import numpy as np
import pytest

from feodosyev_shell.constitutive import convert_curve, evaluate_curve
from feodosyev_shell.materials import load_material

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic"


@pytest.fixture
def material():
    return load_material(EXAMPLE)


def test_exact_nodes_and_temperature_midpoint(material):
    assert evaluate_curve(material, 300, 0.001, kind="engineering") == pytest.approx(200e6)
    assert evaluate_curve(material, 400, 0.003, kind="engineering") == pytest.approx(210e6)
    assert evaluate_curve(material, 350, 0.001, kind="engineering") == pytest.approx(190e6)
    assert evaluate_curve(material, 350, 0.002, kind="engineering") == pytest.approx(207.5e6)
    assert evaluate_curve(material, 350, 0, kind="engineering") == 0
    assert evaluate_curve(material, 350, 0.005, kind="engineering") == pytest.approx(245e6)


def test_outside_range_or_wrong_kind(material):
    for temperature, strain in ((299, 0.001), (350, 0.006)):
        with pytest.raises(ValueError):
            evaluate_curve(material, temperature, strain, kind="engineering")
    with pytest.raises(ValueError, match="definition"):
        evaluate_curve(material, 350, 0.001, kind="true_total")


def test_conversion_known_values():
    strain, stress = convert_curve([0, 0.1], [0, 100], from_kind="engineering",
                                   to_kind="true_total", uniform_deformation=True)
    np.testing.assert_allclose(strain, [0, np.log(1.1)], rtol=1e-8)
    np.testing.assert_allclose(stress, [0, 110], rtol=1e-8)
    plastic, stress = convert_curve([0.02, 0.04], [100, 120], from_kind="engineering",
                                    to_kind="true_plastic", youngs_modulus_pa=1e4,
                                    uniform_deformation=True)
    np.testing.assert_allclose(plastic, np.log1p([0.02, 0.04]) - stress / 1e4, rtol=1e-8)


def test_conversion_rejects_unconfirmed_or_inconsistent_data():
    with pytest.raises(ValueError, match="Uniform"):
        convert_curve([0, 0.1], [0, 100], from_kind="engineering", to_kind="true_total")
    with pytest.raises(ValueError, match="Negative plastic"):
        convert_curve([0, 0.01], [0, 1000], from_kind="engineering",
                      to_kind="true_plastic", youngs_modulus_pa=100,
                      uniform_deformation=True)

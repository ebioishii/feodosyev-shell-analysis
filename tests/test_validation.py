from pathlib import Path

import numpy as np

from feodosyev_shell.materials.dataset import Material
from scripts.validate_phase2 import cross_validate


def test_loto_methods_share_supported_points_and_match_linear_temperature_law():
    temperatures = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    strain = np.array([0.0, 0.1, 0.2, 0.3])
    curves = {temperature: (strain, 1e6 * (20 + 0.2 * temperature + 50 * strain))
              for temperature in temperatures}
    entries = [{"temperature_k": temperature, "strain_intervals": [[0, 0.1], [0.2, 0.3]]}
               for temperature in temperatures]
    material = Material(Path("."), {"curves": entries}, {}, curves)
    samples, summary = cross_validate(material)
    np.testing.assert_allclose(sorted(summary.temperature_c),
                               [400 - 273.15, 500 - 273.15, 600 - 273.15, 700 - 273.15])
    assert (samples.pchip_bracket_excursion_mpa == 0).all()
    np.testing.assert_allclose(samples.linear_error_mpa, 0, atol=1e-10)
    np.testing.assert_allclose(samples.pchip_error_mpa, 0, atol=1e-10)


def test_loto_rejects_unavailable_common_strain_points():
    temperatures = [300.0, 400.0, 500.0, 600.0]
    strain = np.array([0.0, 0.1, 0.2, 0.3])
    curves = {t: (strain, 1e6 * (t / 10 + 100 * strain)) for t in temperatures}
    entries = [{"temperature_k": t, "strain_intervals": [[0, 0.1], [0.2, 0.3]]}
               for t in temperatures]
    entries[0]["strain_intervals"] = [[0, 0.1]]
    material = Material(Path("."), {"curves": entries}, {}, curves)
    samples, _ = cross_validate(material)
    assert (samples[samples.temperature_c == 400 - 273.15].engineering_strain <= 0.1).all()

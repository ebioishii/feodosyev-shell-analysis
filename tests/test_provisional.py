import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from feodosyev_shell.constitutive import curve_strain_intervals, evaluate_curve
from feodosyev_shell.materials import load_material
from feodosyev_shell.materials.legacy import PLASTIC_COLUMNS
from feodosyev_shell.materials.provisional import prepare_provisional_ss316
from feodosyev_shell.results import write_manifest
from feodosyev_shell.visualization import plot_curves_2d, plot_surface_3d


@pytest.fixture
def source(tmp_path):
    rows = []
    for temperature, offset in [(20, 0), (120, 100)]:
        for x, quality in [(0.01, 'visible_curve'), (0.02, 'visible_curve'),
                           (0.03, 'bridged_occlusion'), (0.04, 'visible_curve'),
                           (0.05, 'visible_curve'), (0.06, 'visible_curve'),
                           (0.06, 'visible_steep_segment'), (0.07, 'visible_curve')]:
            rows.append([temperature, x, 1000 * x + offset, 'digitized_provisional',
                         quality, 'synthetic figure', 'synthetic-doi', x * 100, 1])
    path = tmp_path / 'raw.csv'
    pd.DataFrame(rows, columns=PLASTIC_COLUMNS).to_csv(path, index=False)
    return path


def prepare(source):
    return prepare_provisional_ss316(source, source.parent / 'processed')


def test_selection_and_repeatability(source):
    original = source.read_bytes()
    material = prepare(source)
    assert material.strain_intervals(293.15) == [(0.01, 0.02), (0.04, 0.05)]
    assert material.properties == {}
    report = material.completeness_report()
    assert report['provisional_curve_data_available']
    assert not report['full_stress_strain_curves_available']
    audit = pd.read_csv(material.root / 'selection_audit.csv')
    assert len(audit) == 16
    assert (audit.selection_reason == 'excluded_duplicate_strain').sum() == 4
    assert (audit.selection_reason == 'excluded_isolated_point').sum() == 2
    before = {p.name: p.read_bytes() for p in material.root.iterdir()}
    prepare(source)
    assert before == {p.name: p.read_bytes() for p in material.root.iterdir()}
    assert source.read_bytes() == original
    manifest = write_manifest(material, source.parent / 'manifest.json', settings={})
    assert manifest['sha256']['selection_audit.csv'] == hashlib.sha256(
        (material.root / 'selection_audit.csv').read_bytes()).hexdigest()


def test_interpolation_and_gaps(source):
    material = prepare(source)
    with pytest.raises(ValueError, match='allow_provisional'):
        evaluate_curve(material, 343.15, 0.015, kind='engineering')
    query = lambda t, x: evaluate_curve(  # noqa: E731
        material, t, x, kind='engineering', allow_provisional=True)
    # Independent hand calculation: 15 MPa at 20 C, 115 MPa at 120 C.
    assert query(343.15, 0.015) == pytest.approx(65e6, rel=1e-8, abs=1e-6)
    assert curve_strain_intervals(material, 343.15) == [(0.01, 0.02), (0.04, 0.05)]
    for t, (x, y) in material.curves.items():
        np.testing.assert_allclose(query(t, x), y, rtol=1e-8, atol=1e-6)
    for t in [293.15, 343.15, 393.15]:
        for x in [0, 0.03, 0.06, [0.01, 0.03], np.nan]:
            with pytest.raises(ValueError):
                query(t, x)
    for t in [290, 400, np.nan]:
        with pytest.raises(ValueError):
            query(t, 0.01)


def test_charts_preserve_gaps_even_on_coarse_grid(source):
    material = prepare(source)
    with pytest.raises(ValueError, match='allow_provisional'):
        plot_surface_3d(material)
    figure = plot_curves_2d(material, [293.15, 343.15], allow_provisional=True)
    assert len(figure.data) == 4
    assert 'PROVISIONAL' in figure.layout.title.text
    surface = plot_surface_3d(material, n_temperature=2, n_strain=2,
                              allow_provisional=True).data[0]
    x = np.asarray(surface.x) / 100
    assert np.any((x > 0.02) & (x < 0.04))
    assert np.isnan(np.asarray(surface.z)[:, (x > 0.02) & (x < 0.04)]).all()


@pytest.mark.parametrize('intervals', [[], [[0.01, 0.03]],
                                      [[0.01, 0.04], [0.02, 0.05]], [[0.01, 0.02]]])
def test_invalid_interval_metadata(source, intervals):
    material = prepare(source)
    metadata = material.metadata
    metadata['curves'][0]['strain_intervals'] = intervals
    (material.root / 'material.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError):
        load_material(material.root)


def test_source_mixing_and_existing_output_rejected(source):
    material = prepare(source)
    frame = pd.read_csv(source)
    frame.loc[0, 'source_doi'] = 'another-source'
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match='one source'):
        prepare(source)
    frame['source_doi'] = 'changed-source'
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match='different dataset'):
        prepare(source)
    assert load_material(material.root).metadata['source_doi'] == 'synthetic-doi'

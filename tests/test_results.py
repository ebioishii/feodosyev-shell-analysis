from pathlib import Path

from feodosyev_shell.materials import load_material
from feodosyev_shell.results import write_manifest


def test_manifest_hashes_are_repeatable(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "synthetic"
    material = load_material(example)
    first = write_manifest(material, tmp_path / "a.json", settings={"temperature_k": 350})
    second = write_manifest(material, tmp_path / "b.json", settings={"temperature_k": 350})
    assert first["sha256"] == second["sha256"]
    assert first["units"]["stress"] == "Pa"
    assert len(first["sha256"]["material.json"]) == 64
    assert first["git_commit"] is not None
    assert isinstance(first["git_dirty"], bool)

"""Write enough context to reproduce a numerical result."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from feodosyev_shell.materials.dataset import Material


def _git_output(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_manifest(material: Material, output_file: str | Path, *, settings: dict) -> dict:
    """Record dataset bytes, source revision, method, units and dependency versions."""
    files = [material.root / "material.json"]
    files += [material.root / entry["file"] for entry in material.metadata.get("properties", [])]
    files += [material.root / entry["file"] for entry in material.metadata.get("curves", [])]
    hashes = {
        path.relative_to(material.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))
    }
    repo = Path(__file__).resolve().parents[3]
    status = _git_output(repo, "status", "--porcelain")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "material": material.metadata["material"],
        "dataset_version": material.metadata["dataset_version"],
        "sha256": hashes,
        "git_commit": _git_output(repo, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "settings": settings,
        "units": {"temperature": "K", "stress": "Pa", "strain": "1"},
        "interpolation": "piecewise linear in strain, then temperature; no extrapolation",
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "plotly", "matplotlib")},
    }
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

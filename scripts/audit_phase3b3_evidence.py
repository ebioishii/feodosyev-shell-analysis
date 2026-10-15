"""Write a local-only Phase 3B-3 evidence worksheet and blocker report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from feodosyev_shell.materials import load_material
from feodosyev_shell.materials.evidence_gate import (
    audit_true_plastic_readiness,
    evidence_template,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material", type=Path, default=Path("data/processed/ss316_provisional_v1")
    )
    parser.add_argument("--evidence", type=Path, help="Optional completed evidence worksheet")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase3b3_evidence"))
    args = parser.parse_args()
    material = load_material(args.material)
    output = args.output.resolve()
    if output == material.root or output.is_relative_to(material.root):
        raise ValueError("Evidence output must be outside the material dataset directory")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else None
    output.mkdir(parents=True, exist_ok=True)
    template = evidence_template(material)
    template_path = output / "evidence_template.json"
    # Preserve a worksheet the researcher may have filled in after a prior run.
    if not template_path.exists():
        template_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    report = audit_true_plastic_readiness(material, evidence)
    report["material_metadata_sha256"] = hashlib.sha256(
        (material.root / "material.json").read_bytes()
    ).hexdigest()
    names = {"material.json"}
    names.update(entry["file"] for entry in material.metadata.get("properties", []))
    names.update(entry["file"] for entry in material.metadata.get("curves", []))
    names.update(material.metadata.get("auxiliary_files", []))
    dataset_hashes = {}
    for name in sorted(names):
        candidate = (material.root / name).resolve()
        if not candidate.is_relative_to(material.root):
            raise ValueError("Dataset hash file must stay inside material directory")
        dataset_hashes[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    report["dataset_file_sha256"] = dataset_hashes
    report["evidence_sha256"] = (hashlib.sha256(args.evidence.read_bytes()).hexdigest()
                                 if args.evidence else None)
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    report["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    (output / "readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for row in report["temperatures"]:
        print(f"{row['temperature_k'] - 273.15:g} C: {row['software_precheck']}; "
              f"{len(row['blockers'])} blockers")
    print("Physical validation: unverified; no plastic model created")
    print(f"Saved local report in {output}")


if __name__ == "__main__":
    main()

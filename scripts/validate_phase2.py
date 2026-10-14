"""Local, traceable audit of digitized engineering curves; no material approval.

Example: uv run --locked python scripts/validate_phase2.py --source PRIVATE.csv
Output stays under outputs/, which is ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from feodosyev_shell.constitutive import curve_strain_intervals
from feodosyev_shell.materials import load_material

POLICY = "loto-prepeak-common-domain-v1"
COLORS = {20: "black", 350: "#ff00dd", 450: "#3636ff", 550: "#999000",
          650: "#008d9d", 750: "#00a723"}


def _supports(material, temperature_k: float, strain: float) -> bool:
    return any(left <= strain <= right
               for left, right in curve_strain_intervals(material, temperature_k))


def _at(material, temperature_k: float, strain: float) -> float:
    x, y = material.curves[temperature_k]
    return float(np.interp(strain, x, y) / 1e6)


def peak_table(material) -> pd.DataFrame:
    """Retained digitized maxima are proxies, not verified necking limits."""
    rows = []
    for temperature_k, (strain, stress) in sorted(material.curves.items()):
        at = int(np.argmax(stress))
        rows.append({
            "temperature_c": temperature_k - 273.15,
            "retained_peak_stress_mpa": float(stress[at] / 1e6),
            "retained_peak_strain": float(strain[at]),
            "retained_points": len(strain),
            "valid_segments": len(material.strain_intervals(temperature_k)),
            "uniform_strain_limit": "unverified",
            "peak_is_uniform_limit": False,
        })
    return pd.DataFrame(rows)


def _neighbors(temperatures: list[float], index: int) -> tuple[float, float, float] | None:
    if index == 0 or index == len(temperatures) - 1:
        return None
    target, low, high = temperatures[index], temperatures[index - 1], temperatures[index + 1]
    options = [t for t in temperatures if t not in {target, low, high}]
    extra = min(options, key=lambda t: (min(abs(t - low), abs(t - high)), abs(t - target)))
    return low, high, extra


def cross_validate(material) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare methods at the same retained target nodes within four valid prepeak spans.

    PCHIP uses the two bracketing temperatures and one nearest additional
    temperature. Missing strain intervals disqualify the point; no extrapolation.
    """
    temperatures = sorted(material.curves)
    peaks = {t: material.curves[t][0][np.argmax(material.curves[t][1])]
             for t in temperatures}
    rows = []
    for index, target in enumerate(temperatures):
        neighbor = _neighbors(temperatures, index)
        if neighbor is None:
            continue
        low, high, extra = neighbor
        training = sorted([low, high, extra])
        # A missing source temperature between selected knots cannot be skipped.
        if any(t not in training and t != target
               for t in temperatures if training[0] <= t <= training[-1]):
            raise ValueError("PCHIP selection would skip an available temperature")
        x_target, y_target = material.curves[target]
        cutoff = min(peaks[t] for t in [target, *training])
        for strain, observed_pa in zip(x_target, y_target, strict=True):
            if strain > cutoff or not all(
                _supports(material, t, float(strain)) for t in training
            ):
                continue
            bracket = np.array([_at(material, low, float(strain)),
                                _at(material, high, float(strain))])
            fraction = (target - low) / (high - low)
            linear = float(bracket[0] + fraction * (bracket[1] - bracket[0]))
            pchip = float(PchipInterpolator(training, [_at(material, t, float(strain))
                                                       for t in training])(target))
            rows.append({
                "temperature_c": target - 273.15, "train_low_c": low - 273.15,
                "train_high_c": high - 273.15, "pchip_extra_c": extra - 273.15,
                "engineering_strain": float(strain),
                "observed_mpa": float(observed_pa / 1e6),
                "linear_mpa": linear, "pchip_mpa": pchip,
                "linear_error_mpa": linear - observed_pa / 1e6,
                "pchip_error_mpa": pchip - observed_pa / 1e6,
                "pchip_bracket_excursion_mpa": float(max(
                    bracket.min() - pchip, pchip - bracket.max(), 0.0)),
            })
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise ValueError("No common valid prepeak samples for cross-validation")
    summaries = []
    for temperature, group in samples.groupby("temperature_c", sort=True):
        item = {"temperature_c": temperature, "points": len(group),
                "strain_min": group.engineering_strain.min(),
                "strain_max": group.engineering_strain.max()}
        for method in ("linear", "pchip"):
            error = group[f"{method}_error_mpa"].to_numpy()
            item[f"{method}_mae_mpa"] = float(np.mean(np.abs(error)))
            item[f"{method}_rmse_mpa"] = float(np.sqrt(np.mean(error**2)))
            item[f"{method}_max_abs_error_mpa"] = float(np.max(np.abs(error)))
        item["pchip_max_bracket_excursion_mpa"] = group.pchip_bracket_excursion_mpa.max()
        summaries.append(item)
    return samples, pd.DataFrame(summaries)


def _source_color_mask(image: np.ndarray, temperature_c: int) -> np.ndarray:
    rgb = image[..., :3].astype(float)
    if rgb.max() <= 1:
        rgb *= 255
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if temperature_c == 20:
        mask = (red < 100) & (green < 100) & (blue < 100)
    elif temperature_c == 350:
        mask = (red > 180) & (green < 130) & (blue > 130)
    elif temperature_c == 450:
        mask = (blue > 180) & (red > 100) & (red < 170) & (green < 100)
    elif temperature_c == 550:
        mask = (red > 80) & (green > 65) & (blue < 100) & (red > blue * 1.4)
    elif temperature_c == 650:
        mask = (red < 105) & (green > 80) & (blue > 90)
    else:
        mask = (red < 105) & (green > 90) & (blue < 130)
    yy, xx = np.indices(mask.shape)
    return mask & (xx >= 120) & (xx <= 440) & (yy >= 60) & (yy <= 350)


def figure_overlay(material, crop: Path, output: Path) -> pd.DataFrame:
    """Overlay onto publisher page 5 crop with visually calibrated printed axes.

    Pixel distance measures agreement to a rendered *printed* line, not error
    against author raw instrument data. Calibration is approximate.
    """
    image = mpimg.imread(crop)
    if image.shape[:2] != (440, 520):
        raise ValueError("Expected the page-5 Fig. 2(a) crop at 180 dpi, [190:630,500:1020]")
    axis = {"x_zero_px": 120, "x_per_strain_px": 500,
            "y_at_150mpa_px": 358, "y_per_mpa_px": 0.633}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for panel in axes:
        panel.imshow(image)
        panel.set_xlim(75, 490)
        panel.set_ylim(385, 30)
        panel.axis("off")
    axes[0].set_title("Lv et al. Fig. 2(a), publisher PDF")
    axes[1].set_title("Retained digitized points over original")
    rows = []
    for temperature_k, (strain, stress) in sorted(material.curves.items()):
        temperature = round(temperature_k - 273.15)
        x = axis["x_zero_px"] + axis["x_per_strain_px"] * strain
        y = axis["y_at_150mpa_px"] - axis["y_per_mpa_px"] * (stress / 1e6 - 150)
        axes[1].scatter(x[::3], y[::3], s=8, edgecolors=COLORS[temperature], marker="o",
                        facecolors="none", linewidths=0.65, label=f"{temperature} °C")
        mask = _source_color_mask(image, temperature)
        positions = np.column_stack(np.where(mask))
        if len(positions) == 0:
            raise ValueError(f"Could not isolate {temperature} C line in publisher figure")
        distances, _ = cKDTree(positions).query(np.column_stack([y, x]))
        rows.append({"temperature_c": temperature,
                     "rendered_line_median_distance_px": float(np.median(distances)),
                     "rendered_line_p90_distance_px": float(np.percentile(distances, 90)),
                     "rendered_line_fraction_within_5px": float(np.mean(distances <= 5)),
                     "calibration": json.dumps(axis)})
    fig.savefig(output / "fig2a_overlay.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def _common_intervals(material, temperatures: list[float]) -> list[tuple[float, float]]:
    intervals = material.strain_intervals(temperatures[0])
    for temperature in temperatures[1:]:
        intervals = [(max(a, c), min(b, d))
                     for a, b in intervals
                     for c, d in material.strain_intervals(temperature)
                     if max(a, c) <= min(b, d)]
    return intervals


def plot_errors(samples: pd.DataFrame, material, output: Path) -> None:
    temperatures = sorted(samples.temperature_c.unique())
    fig, axes = plt.subplots(len(temperatures), 1, figsize=(9, 8), sharex=True,
                             layout="constrained")
    for axis, temperature in zip(axes, temperatures, strict=True):
        part = samples[samples.temperature_c == temperature]
        first = part.iloc[0]
        sources = [temperature + 273.15, first.train_low_c + 273.15,
                   first.train_high_c + 273.15, first.pchip_extra_c + 273.15]
        for index, (left, right) in enumerate(_common_intervals(material, sources)):
            segment = part[(part.engineering_strain >= left)
                           & (part.engineering_strain <= right)]
            if segment.empty:
                continue
            axis.plot(segment.engineering_strain * 100, segment.linear_error_mpa,
                      label="linear" if index == 0 else None,
                      color="tab:blue", linewidth=1)
            axis.plot(segment.engineering_strain * 100, segment.pchip_error_mpa,
                      label="PCHIP" if index == 0 else None,
                      color="tab:orange", linewidth=1)
        axis.axhline(0, color="0.6", linewidth=0.6)
        axis.set_ylabel(f"{temperature:g} °C\nMPa")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Engineering strain (%)")
    fig.savefig(output / "loto_error_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/ss316_provisional_v1"))
    parser.add_argument("--figure-crop", type=Path,
                        default=Path("outputs/phase2_validation/fig2a_crop.png"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/phase2_validation"))
    args = parser.parse_args()
    material = load_material(args.dataset)
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_sha != material.metadata.get("source_sha256"):
        raise ValueError("Source CSV SHA-256 differs from processed dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    peaks = peak_table(material)
    samples, summary = cross_validate(material)
    peaks.to_csv(args.output / "c1_peak_table.csv", index=False)
    samples.to_csv(args.output / "loto_error_samples.csv", index=False)
    summary.to_csv(args.output / "loto_summary.csv", index=False)
    plot_errors(samples, material, args.output)
    overlay = figure_overlay(material, args.figure_crop, args.output)
    overlay.to_csv(args.output / "fig2a_overlay_residuals.csv", index=False)
    (args.output / "validation_manifest.json").write_text(json.dumps({
        "source_csv_sha256": source_sha,
        "processed_dataset_version": material.metadata["dataset_version"],
        "processed_selection_policy": material.metadata["selection_policy"],
        "figure_crop_sha256": hashlib.sha256(args.figure_crop.read_bytes()).hexdigest(),
        "method": POLICY,
        "comparison": "retained target nodes; 4-curve common valid spans; before all 4 peaks",
        "pchip_knots": "bracketing temperatures plus one nearest additional temperature",
        "figure_overlay": "visually calibrated PDF page 5 raster at 180 dpi; pixel residual only",
        "physical_validation": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(overlay.to_string(index=False))


if __name__ == "__main__":
    main()

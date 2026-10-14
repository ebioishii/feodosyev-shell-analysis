"""Interactive plots that preserve excluded strain gaps and dataset status."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from feodosyev_shell.constitutive import curve_strain_intervals, evaluate_curve
from feodosyev_shell.constitutive.curves import _allow_curve_use
from feodosyev_shell.materials.dataset import Material


def _title(material: Material) -> str:
    prefix = "PROVISIONAL / 暫用、部分曲線 — " if material.is_provisional else ""
    return f"{prefix}{material.metadata['material']}：{material.curve_kind}"


def plot_curves_2d(
    material: Material, temperatures_k: list[float], *, allow_provisional: bool = False
) -> go.Figure:
    _allow_curve_use(material, allow_provisional)
    if not material.curves:
        raise ValueError("No curves to plot")
    figure = go.Figure()
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2"]
    for index, temperature in enumerate(temperatures_k):
        for segment, (left, right) in enumerate(curve_strain_intervals(material, temperature)):
            if temperature in material.curves:
                x, _ = material.curves[temperature]
                strain = x[(x >= left) & (x <= right)]
                mode = "lines+markers"
            else:
                strain = np.linspace(left, right, 100)
                mode = "lines"
            stress = evaluate_curve(material, temperature, strain, kind=material.curve_kind,
                                    allow_provisional=allow_provisional)
            figure.add_trace(go.Scatter(
                x=strain * 100, y=stress / 1e6, mode=mode,
                name=f"{temperature - 273.15:g} °C", legendgroup=str(index),
                showlegend=segment == 0, line={"color": colors[index % len(colors)]},
                connectgaps=False,
            ))
    figure.update_layout(xaxis_title=f"{material.curve_kind} 應變 (%)",
                         yaxis_title="應力 (MPa)", title=_title(material))
    return figure


def plot_surface_3d(
    material: Material, *, n_temperature: int = 30, n_strain: int = 80,
    allow_provisional: bool = False,
) -> go.Figure:
    _allow_curve_use(material, allow_provisional)
    if len(material.curves) < 2 or n_temperature < 2 or n_strain < 2:
        raise ValueError("3D surface needs two temperatures and at least two grid points per axis")
    knots = sorted(material.curves)
    minimum = max(curve[0][0] for curve in material.curves.values())
    maximum = min(curve[0][-1] for curve in material.curves.values())
    if minimum >= maximum:
        raise ValueError("No shared strain interval across all curves")
    # Insert a midpoint in every gap, including gaps narrower than the regular grid.
    extra = []
    for temperature in knots:
        intervals = material.strain_intervals(temperature)
        extra.extend(value for pair in intervals for value in pair)
        extra.extend((a[1] + b[0]) / 2 for a, b in zip(intervals, intervals[1:], strict=False))
    strains = np.unique(np.r_[np.linspace(minimum, maximum, n_strain), extra])
    strains = strains[(strains >= minimum) & (strains <= maximum)]
    temperatures = np.unique(np.r_[np.linspace(knots[0], knots[-1], n_temperature), knots])
    stresses = np.full((len(temperatures), len(strains)), np.nan)
    for index, temperature in enumerate(temperatures):
        mask = np.zeros(len(strains), dtype=bool)
        for left, right in curve_strain_intervals(material, temperature):
            mask |= (strains >= left) & (strains <= right)
        if mask.any():
            stresses[index, mask] = evaluate_curve(
                material, temperature, strains[mask], kind=material.curve_kind,
                allow_provisional=allow_provisional,
            )
    if not np.isfinite(stresses).any():
        raise ValueError("No supported strain samples for the surface")
    figure = go.Figure(go.Surface(
        x=strains * 100,
        y=np.broadcast_to((temperatures - 273.15)[:, None], stresses.shape),
        z=stresses / 1e6, connectgaps=False,
        hovertemplate="應變 %{x:.3f}%<br>溫度 %{y:.1f} °C<br>應力 %{z:.2f} MPa<extra></extra>",
    ))
    figure.update_layout(title=_title(material),
                         scene=dict(xaxis_title=f"{material.curve_kind} 應變 (%)",
                                    yaxis_title="溫度 (°C)", zaxis_title="應力 (MPa)"))
    return figure

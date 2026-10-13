"""Interactive plots generated from validated curve evaluations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from feodosyev_shell.constitutive import evaluate_curve
from feodosyev_shell.materials.dataset import Material


def plot_curves_2d(material: Material, temperatures_k: list[float]) -> go.Figure:
    if not material.curves:
        raise ValueError("No curves to plot")
    figure = go.Figure()
    for temperature in temperatures_k:
        knots = sorted(material.curves)
        if temperature in material.curves:
            strain, stress = material.curves[temperature]
            mode = "lines+markers"
        else:
            lower = max((t for t in knots if t < temperature), default=None)
            upper = min((t for t in knots if t > temperature), default=None)
            if lower is None or upper is None:
                raise ValueError("Temperature outside available curves")
            left, right = material.curves[lower][0], material.curves[upper][0]
            strain = np.linspace(max(left[0], right[0]), min(left[-1], right[-1]), 100)
            stress = evaluate_curve(material, temperature, strain, kind=material.curve_kind)
            mode = "lines"
        figure.add_trace(go.Scatter(x=strain * 100, y=stress / 1e6, mode=mode,
                                    name=f"{temperature - 273.15:g} °C"))
    figure.update_layout(xaxis_title="應變 (%)", yaxis_title="應力 (MPa)",
                         title=f"{material.metadata['material']}：{material.curve_kind}")
    return figure


def plot_surface_3d(
    material: Material, *, n_temperature: int = 30, n_strain: int = 80
) -> go.Figure:
    if len(material.curves) < 2 or n_temperature < 2 or n_strain < 2:
        raise ValueError("3D surface needs two temperatures and at least two grid points per axis")
    knots = sorted(material.curves)
    minimum = max(curve[0][0] for curve in material.curves.values())
    maximum = min(curve[0][-1] for curve in material.curves.values())
    if minimum >= maximum:
        raise ValueError("No shared strain interval across all curves")
    temperatures = np.linspace(knots[0], knots[-1], n_temperature)
    strains = np.linspace(minimum, maximum, n_strain)
    stresses = np.array([evaluate_curve(material, t, strains, kind=material.curve_kind)
                         for t in temperatures])
    figure = go.Figure(go.Surface(x=strains * 100, y=(temperatures - 273.15)[:, None]
                                  * np.ones((1, n_strain)), z=stresses / 1e6,
                                  hovertemplate=(
                                      "應變 %{x:.3f}%<br>溫度 %{y:.1f} °C<br>"
                                      "應力 %{z:.2f} MPa<extra></extra>"
                                  )))
    figure.update_layout(title=f"{material.metadata['material']}：{material.curve_kind}",
                         scene=dict(xaxis_title="應變 (%)", yaxis_title="溫度 (°C)",
                                    zaxis_title="應力 (MPa)"))
    return figure

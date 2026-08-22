"""Plotly figures shared by reports and the Streamlit interface."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from modelens.domain.beam_theory import cantilever_mode_shape
from modelens.domain.entities import AnalysisResult, ComparisonResult, ModalMode


def signal_figure(result: AnalysisResult, point_index: int = -1) -> go.Figure:
    """Plot raw and cleaned displacement at one beam station."""
    point = point_index % result.signal.cleaned_displacement.shape[1]
    figure = go.Figure()
    figure.add_scatter(
        x=result.signal.time_s,
        y=result.signal.raw_displacement[:, point],
        name="Raw",
        opacity=0.55,
    )
    figure.add_scatter(
        x=result.signal.time_s,
        y=result.signal.cleaned_displacement[:, point],
        name="Cleaned",
    )
    figure.update_layout(
        title=f"Displacement at station {point + 1}",
        xaxis_title="Time [s]",
        yaxis_title=f"Displacement [{result.signal.unit}]",
        template="plotly_white",
    )
    return figure


def psd_figure(result: AnalysisResult) -> go.Figure:
    """Plot PSDs of the retained POD temporal coordinates."""
    figure = go.Figure()
    for component, power in enumerate(result.modal.component_psd, start=1):
        figure.add_scatter(
            x=result.modal.frequencies_hz,
            y=power,
            name=f"POD {component}",
        )
    for mode in result.modal.modes:
        figure.add_vline(x=mode.frequency_hz, line_dash="dot", line_color="#ef8354")
    figure.update_layout(
        title="Modal-coordinate power spectra",
        xaxis_title="Frequency [Hz]",
        yaxis_title="PSD [signal²/Hz]",
        yaxis_type="log",
        template="plotly_white",
    )
    return figure


def scree_figure(result: AnalysisResult) -> go.Figure:
    """Plot retained POD energy fractions."""
    indices = np.arange(1, result.modal.explained_energy.size + 1)
    figure = go.Figure(
        go.Bar(x=indices, y=100.0 * result.modal.explained_energy, marker_color="#35618f")
    )
    figure.update_layout(
        title="POD explained energy",
        xaxis_title="Component",
        yaxis_title="Energy [%]",
        template="plotly_white",
    )
    return figure


def mode_shape_figure(result: AnalysisResult, mode: ModalMode) -> go.Figure:
    """Compare a measured mode shape with its cantilever counterpart."""
    measured = mode.shape / max(float(np.max(np.abs(mode.shape))), 1e-12)
    figure = go.Figure()
    figure.add_scatter(
        x=result.signal.position_normalized,
        y=measured,
        mode="lines+markers",
        name="Measured",
    )
    if mode.index <= 4:
        theoretical = cantilever_mode_shape(result.signal.position_normalized, mode.index)
        if float(np.dot(theoretical, measured)) < 0.0:
            theoretical = -theoretical
        figure.add_scatter(
            x=result.signal.position_normalized,
            y=theoretical,
            mode="lines",
            name="Euler–Bernoulli",
        )
    figure.update_layout(
        title=f"Mode {mode.index}: {mode.frequency_hz:.3f} Hz",
        xaxis_title="Normalized beam position",
        yaxis_title="Normalized displacement",
        template="plotly_white",
    )
    return figure


def animated_mode_figure(result: AnalysisResult, mode: ModalMode) -> go.Figure:
    """Animate a normalized measured mode shape through one oscillation."""
    spatial = result.signal.position_normalized
    shape = mode.shape / max(float(np.max(np.abs(mode.shape))), 1e-12)
    phases = np.linspace(0.0, 2.0 * np.pi, 25)
    frames = [
        go.Frame(
            data=[go.Scatter(x=spatial, y=shape * np.sin(phase), mode="lines+markers")],
            name=str(index),
        )
        for index, phase in enumerate(phases)
    ]
    figure = go.Figure(data=frames[0].data, frames=frames)
    figure.update_layout(
        title=f"Animated shape — mode {mode.index}",
        xaxis={"title": "Normalized beam position", "range": [0.0, 1.0]},
        yaxis={"title": "Normalized displacement", "range": [-1.1, 1.1]},
        template="plotly_white",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 50, "redraw": True}}],
                    }
                ],
            }
        ],
    )
    return figure


def twin_figure(result: AnalysisResult) -> go.Figure:
    """Plot measured and fitted theoretical frequencies."""
    if result.twin is None:
        return go.Figure().update_layout(title="No calibrated twin is available")
    measured = [mode.frequency_hz for mode in result.modal.modes[: result.twin.residuals_hz.size]]
    indices = np.arange(1, len(measured) + 1)
    figure = go.Figure()
    figure.add_scatter(x=indices, y=measured, mode="markers", name="Measured")
    figure.add_scatter(
        x=indices,
        y=result.twin.theoretical_frequencies_hz,
        mode="lines+markers",
        name="Fitted twin",
    )
    figure.update_layout(
        title="Measured vs fitted modal frequencies",
        xaxis_title="Mode",
        yaxis_title="Frequency [Hz]",
        template="plotly_white",
    )
    return figure


def mac_figure(result: ComparisonResult) -> go.Figure:
    """Plot the pairwise Modal Assurance Criterion matrix."""
    figure = go.Figure(
        go.Heatmap(
            z=result.mac_matrix,
            zmin=0.0,
            zmax=1.0,
            colorscale="Blues",
            colorbar={"title": "MAC"},
        )
    )
    figure.update_layout(
        title="Modal Assurance Criterion",
        xaxis_title="Modified mode",
        yaxis_title="Baseline mode",
        template="plotly_white",
    )
    return figure

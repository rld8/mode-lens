"""Restricted physical twin exploration page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modelens.adapters.plotly_presenter import twin_figure
from modelens.domain.beam_theory import cantilever_frequencies_hz, combined_parameter
from modelens.domain.entities import BeamGeometry, Material
from modelens.interfaces.streamlit_app.components import require_analysis, scope_notice

st.title("Digital twin")
result = require_analysis()
scope_notice()

if result.twin is None:
    st.info(
        "This experiment has no geometry/material measurements, so no physical twin was fitted."
    )
    st.stop()

st.plotly_chart(twin_figure(result), use_container_width=True)
st.metric(result.twin.parameter_name, f"{result.twin.fitted_value:.6g}")
if result.twin.confidence_interval is not None:
    st.caption(
        f"Approximate 95% interval: {result.twin.confidence_interval[0]:.6g}–"
        f"{result.twin.confidence_interval[1]:.6g}"
    )
for warning in result.twin.warnings:
    st.warning(warning)

st.subheader("Counterfactual explorer")
col1, col2, col3 = st.columns(3)
with col1:
    length = st.slider("Length [mm]", 100.0, 600.0, 300.0, 1.0) / 1000.0
    width = st.slider("Width [mm]", 5.0, 80.0, 25.0, 0.5) / 1000.0
with col2:
    thickness = st.slider("Thickness [mm]", 0.5, 10.0, 3.0, 0.1) / 1000.0
    density = st.slider("Density [kg/m³]", 200.0, 8000.0, 1180.0, 10.0)
with col3:
    young_gpa = st.slider("Young's modulus [GPa]", 0.1, 210.0, 0.5, 0.1)

geometry = BeamGeometry(length, width, thickness)
material = Material("Counterfactual", young_gpa * 1e9, density)
frequencies = cantilever_frequencies_hz(geometry, combined_parameter(geometry, material), 4)
st.dataframe(
    pd.DataFrame({"Mode": np.arange(1, 5), "Predicted frequency [Hz]": frequencies}),
    hide_index=True,
    use_container_width=True,
)

base = float(frequencies[0])
perturbations = [
    BeamGeometry(length * 1.01, width, thickness),
    BeamGeometry(length, width * 1.01, thickness),
    BeamGeometry(length, width, thickness * 1.01),
]
changes = [
    100.0
    * (
        cantilever_frequencies_hz(
            perturbed,
            combined_parameter(perturbed, material),
            1,
        )[0]
        / base
        - 1.0
    )
    for perturbed in perturbations
]
denser = Material("Density +1%", young_gpa * 1e9, density * 1.01)
stiffer = Material("E +1%", young_gpa * 1.01e9, density)
changes.extend(
    [
        100.0
        * (
            cantilever_frequencies_hz(
                geometry,
                combined_parameter(geometry, changed_material),
                1,
            )[0]
            / base
            - 1.0
        )
        for changed_material in (denser, stiffer)
    ]
)
sensitivity = pd.DataFrame(
    {
        "Parameter +1%": ["Length", "Width", "Thickness", "Density", "Young's modulus"],
        "First-frequency change [%]": changes,
    }
)
chart = px.bar(sensitivity, x="Parameter +1%", y="First-frequency change [%]")
chart.update_layout(title=f"Local analytical sensitivity around f₁={base:.2f} Hz")
st.plotly_chart(chart, use_container_width=True)

"""Measured modes and spectral evidence page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from modelens.adapters.jinja_report import JinjaReportRenderer
from modelens.adapters.plotly_presenter import (
    animated_mode_figure,
    mode_shape_figure,
    psd_figure,
    scree_figure,
    signal_figure,
)
from modelens.interfaces.streamlit_app.components import modes_table, require_analysis
from modelens.interfaces.streamlit_app.state import result_json_bytes

st.title("Modal analysis")
result = require_analysis()

left, right = st.columns(2)
with left:
    st.plotly_chart(signal_figure(result), use_container_width=True)
with right:
    st.plotly_chart(scree_figure(result), use_container_width=True)
st.plotly_chart(psd_figure(result), use_container_width=True)
modes_table(result)

mode_by_index = {mode.index: mode for mode in result.modal.modes}
if not mode_by_index:
    st.warning("No modal mode passed the identification checks for this capture.")
    st.stop()

selected_index = st.selectbox(
    "Inspect mode",
    options=list(mode_by_index),
    format_func=lambda index: f"Mode {index} — {mode_by_index[index].frequency_hz:.3f} Hz",
    key="modal_mode_index",
)
selected = mode_by_index[selected_index]
shape_column, animation_column = st.columns(2)
with shape_column:
    st.plotly_chart(mode_shape_figure(result, selected), use_container_width=True)
with animation_column:
    st.plotly_chart(animated_mode_figure(result, selected), use_container_width=True)

if selected.damping_ratio is None:
    st.info("Damping is reported as not identifiable because its decay checks did not pass.")

st.subheader("Explicit export")
json_column, report_column = st.columns(2)
with json_column:
    st.download_button(
        "Download complete result.json",
        data=result_json_bytes(result),
        file_name=f"modelens_{result.run_id}_result.json",
        mime="application/json",
        use_container_width=True,
    )
with report_column, tempfile.TemporaryDirectory(prefix="modelens-report-") as directory:
    report_path = JinjaReportRenderer().render_analysis(
        result, Path(directory) / "analysis_report.html"
    )
    st.download_button(
        "Download self-contained HTML report",
        data=report_path.read_bytes(),
        file_name=f"modelens_{result.run_id}_report.html",
        mime="text/html",
        use_container_width=True,
    )

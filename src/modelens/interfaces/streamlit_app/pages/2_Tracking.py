"""Tracking audit page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modelens.adapters.opencv_tracking import render_tracking_overlay
from modelens.interfaces.streamlit_app.components import require_analysis
from modelens.interfaces.streamlit_app.state import VIDEO_BYTES_KEY, VIDEO_NAME_KEY

st.title("Tracking")
result = require_analysis()

video_bytes = st.session_state.get(VIDEO_BYTES_KEY)
if isinstance(video_bytes, bytes):
    overlay_key = f"modelens_overlay_{result.run_id}"
    if st.button("Render tracking overlay", use_container_width=True):
        with st.spinner("Rendering an auditable overlay…"):
            suffix = Path(str(st.session_state.get(VIDEO_NAME_KEY, "capture.mp4"))).suffix
            with tempfile.TemporaryDirectory(prefix="modelens-overlay-") as directory:
                input_path = Path(directory) / f"source{suffix or '.mp4'}"
                output_path = Path(directory) / "tracking_overlay.mp4"
                input_path.write_bytes(video_bytes)
                render_tracking_overlay(str(input_path), str(output_path), result.tracking)
                st.session_state[overlay_key] = output_path.read_bytes()
    rendered = st.session_state.get(overlay_key)
    st.video(rendered if isinstance(rendered, bytes) else video_bytes)

valid_fraction = np.mean(result.tracking.valid, axis=0)
confidence = np.mean(result.tracking.confidence, axis=0)
summary = pd.DataFrame(
    {
        "Station": np.arange(1, valid_fraction.size + 1),
        "Valid observations [%]": 100.0 * valid_fraction,
        "Mean confidence": confidence,
    }
)
st.caption(f"Tracker: {result.tracking.method}")
st.dataframe(summary, hide_index=True, use_container_width=True)

station = st.slider(
    "Station", 1, result.signal.position_normalized.size, result.signal.position_normalized.size
)
figure = go.Figure()
figure.add_scatter(
    x=result.tracking.time_s,
    y=result.tracking.positions_px[:, station - 1, 1],
    mode="lines",
    name="y position",
)
figure.update_layout(
    xaxis_title="Time [s]",
    yaxis_title="Image y [px]",
    template="plotly_white",
)
st.plotly_chart(figure, use_container_width=True)

heatmap = px.imshow(
    result.signal.cleaned_displacement.T,
    x=result.signal.time_s,
    y=result.signal.position_normalized,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    origin="lower",
    labels={"x": "Time [s]", "y": "Normalized position", "color": result.signal.unit},
)
heatmap.update_layout(title="Cleaned displacement: time × beam position")
st.plotly_chart(heatmap, use_container_width=True)

signals = {"time_s": result.signal.time_s}
for point in range(result.signal.cleaned_displacement.shape[1]):
    signals[f"raw_{point:02d}_{result.signal.unit}"] = result.signal.raw_displacement[:, point]
    signals[f"clean_{point:02d}_{result.signal.unit}"] = result.signal.cleaned_displacement[
        :, point
    ]
st.download_button(
    "Download signals.csv",
    data=pd.DataFrame(signals).to_csv(index=False).encode(),
    file_name=f"modelens_{result.run_id}_signals.csv",
    mime="text/csv",
)

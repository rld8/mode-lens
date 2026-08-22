"""Experiment input and capture-quality page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from modelens.config import ExperimentConfig, GeometryConfig, MaterialConfig, load_config
from modelens.domain.errors import ModeLensError
from modelens.interfaces.streamlit_app.components import quality_table, scope_notice
from modelens.interfaces.streamlit_app.state import (
    analyse_path,
    analyse_upload,
    current_analysis,
    repository_root,
)

st.title("Experiment")
st.write(
    "Start with the reproducible cantilever demo, or upload a short capture and provide the "
    "beam axis in pixel coordinates. The support is the first point; the free end is the second."
)
scope_notice()

root = repository_root()
base_config = load_config(root / "configs/demo_cantilever.yaml")
demo_path = root / "data/samples/cantilever_baseline.mp4"

demo_column, upload_column = st.columns(2)
with demo_column:
    st.subheader("Bundled demo")
    st.video(str(demo_path))
    run_demo = st.button(
        "Load and analyse cantilever demo", type="primary", use_container_width=True
    )

with upload_column:
    st.subheader("Your capture")
    upload = st.file_uploader("MP4, MOV or AVI", type=["mp4", "mov", "avi"])
    col_a, col_b = st.columns(2)
    with col_a:
        start_x = st.number_input("Support x [px]", min_value=0.0, value=144.0)
        start_y = st.number_input("Support y [px]", min_value=0.0, value=360.0)
    with col_b:
        end_x = st.number_input("Free-end x [px]", min_value=0.0, value=1140.0)
        end_y = st.number_input("Free-end y [px]", min_value=0.0, value=360.0)
    use_scale = st.checkbox("Use an in-plane length scale")
    metres_per_pixel = (
        st.number_input(
            "Scale [metres/pixel]",
            min_value=0.000001,
            value=0.0003012,
            format="%.7f",
        )
        if use_scale
        else None
    )
    tracker = st.selectbox(
        "Tracker",
        options=["lucas_kanade", "contour"],
        help=(
            "Lucas–Kanade suits textured real specimens. Contour suits a bright "
            "specimen on a dark background."
        ),
    )
    with st.expander("Optional region of interest"):
        use_roi = st.checkbox("Restrict quality and contour tracking to a rectangle")
        roi_col_a, roi_col_b = st.columns(2)
        with roi_col_a:
            roi_x = st.number_input("ROI x [px]", min_value=0, value=90)
            roi_y = st.number_input("ROI y [px]", min_value=0, value=190)
        with roi_col_b:
            roi_width = st.number_input("ROI width [px]", min_value=1, value=1140)
            roi_height = st.number_input("ROI height [px]", min_value=1, value=380)
    with st.expander("Optional physical twin measurements"):
        use_twin = st.checkbox("Fit a restricted Euler-Bernoulli twin")
        length_mm = st.number_input("Free length [mm]", min_value=1.0, value=300.0)
        width_mm = st.number_input("Width [mm]", min_value=0.1, value=25.0)
        thickness_mm = st.number_input("Thickness [mm]", min_value=0.1, value=3.0)
        young_gpa = st.number_input("Young's modulus [GPa]", min_value=0.001, value=0.5)
        density = st.number_input("Density [kg/m³]", min_value=1.0, value=1180.0)
    analyse_uploaded = st.button(
        "Analyse uploaded capture", disabled=upload is None, use_container_width=True
    )

if run_demo:
    try:
        with st.spinner("Tracking the beam and identifying modes…"):
            analyse_path(demo_path, base_config)
        st.success("Demo analysis completed. Explore Tracking, Modal analysis and Digital twin.")
    except (ModeLensError, ValueError, OSError) as exc:
        st.error(str(exc))

if analyse_uploaded and upload is not None:
    geometry = (
        GeometryConfig(
            length_m=length_mm / 1000.0,
            width_m=width_mm / 1000.0,
            thickness_m=thickness_mm / 1000.0,
        )
        if use_twin
        else None
    )
    material = (
        MaterialConfig(
            name="User-specified",
            young_modulus_pa=young_gpa * 1e9,
            density_kg_m3=density,
        )
        if use_twin
        else None
    )
    experiment = ExperimentConfig(
        boundary_condition="cantilever",
        roi=(roi_x, roi_y, roi_width, roi_height) if use_roi else None,
        axis_start_px=(start_x, start_y),
        axis_end_px=(end_x, end_y),
        meters_per_pixel=metres_per_pixel,
        geometry=geometry,
        material=material,
    )
    config = base_config.model_copy(
        update={
            "experiment": experiment,
            "tracking": base_config.tracking.model_copy(update={"method": tracker}),
        }
    )
    try:
        with st.spinner("Analysing locally; the uploaded file is not retained…"):
            analyse_upload(upload.getvalue(), Path(upload.name).suffix.lower(), config)
        st.success("Capture analysed. Review every warning before interpreting the modes.")
    except (ModeLensError, ValueError, OSError) as exc:
        st.error(str(exc))

result = current_analysis()
if result is not None:
    st.divider()
    first, second, third, fourth = st.columns(4)
    first.metric("FPS", f"{result.metadata.fps_effective:.2f}")
    second.metric("Duration", f"{result.metadata.duration_s:.2f} s")
    third.metric("Resolution", f"{result.metadata.width_px}×{result.metadata.height_px}")
    fourth.metric("Quality", f"{result.quality.score:.0f}/100")
    quality_table(result)
    for warning in result.warnings:
        st.warning(warning)

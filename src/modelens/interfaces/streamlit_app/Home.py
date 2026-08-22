"""ModeLens Streamlit entry point."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="ModeLens",
    page_icon="〽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HERE = Path(__file__).resolve().parent
pages = [
    st.Page(str(HERE / "pages/1_Experiment.py"), title="Experiment", icon="🎥"),
    st.Page(str(HERE / "pages/2_Tracking.py"), title="Tracking", icon="🎯"),
    st.Page(str(HERE / "pages/3_Modal_Analysis.py"), title="Modal analysis", icon="〽️"),
    st.Page(str(HERE / "pages/4_Digital_Twin.py"), title="Digital twin", icon="📐"),
    st.Page(str(HERE / "pages/5_Comparison.py"), title="Comparison", icon="↔️"),
]

st.logo(str(HERE.parents[3] / "assets/logo.svg"), size="large")
navigation = st.navigation(pages)

st.sidebar.caption("Local-first · explainable · reproducible")
st.sidebar.info("Uploaded captures are analysed locally and kept only in session memory.")
navigation.run()

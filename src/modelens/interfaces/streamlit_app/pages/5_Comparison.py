"""Baseline/modified comparison and explicit exports."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from modelens.adapters.jinja_report import JinjaReportRenderer
from modelens.adapters.plotly_presenter import mac_figure
from modelens.application.compare_experiments import CompareExperiments
from modelens.application.dto import analysis_result_from_dict
from modelens.domain.entities import ComparisonResult
from modelens.interfaces.streamlit_app.components import scope_notice
from modelens.interfaces.streamlit_app.state import COMPARISON_KEY

st.title("Comparison")
scope_notice()
st.write("Upload two complete `result.json` exports produced with compatible station layouts.")

baseline_file = st.file_uploader("Baseline result.json", type=["json"], key="baseline")
modified_file = st.file_uploader("Modified result.json", type=["json"], key="modified")
if st.button("Compare experiments", disabled=baseline_file is None or modified_file is None):
    try:
        baseline_payload = json.loads(baseline_file.getvalue())
        modified_payload = json.loads(modified_file.getvalue())
        result = CompareExperiments().execute(
            analysis_result_from_dict(baseline_payload),
            analysis_result_from_dict(modified_payload),
        )
        st.session_state[COMPARISON_KEY] = result
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        st.error(f"The result files are invalid or incompatible: {exc}")

comparison = st.session_state.get(COMPARISON_KEY)
if isinstance(comparison, ComparisonResult):
    st.plotly_chart(mac_figure(comparison), use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Baseline mode": match.baseline_index,
                    "Modified mode": match.modified_index,
                    "MAC": round(match.mac, 4),
                    "Δf/f [%]": round(100.0 * match.relative_frequency_change, 3),
                    "Δ damping": match.damping_change,
                    "Label": match.label.value,
                    "Reason": match.reason,
                }
                for match in comparison.matches
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    for warning in comparison.warnings:
        st.warning(warning)
    if comparison.unmatched_baseline or comparison.unmatched_modified:
        st.info(
            f"Unmatched baseline modes: {comparison.unmatched_baseline or 'none'}; "
            f"unmatched modified modes: {comparison.unmatched_modified or 'none'}."
        )
    with tempfile.TemporaryDirectory(prefix="modelens-report-") as directory:
        path = JinjaReportRenderer().render_comparison(
            comparison, Path(directory) / "comparison_report.html"
        )
        st.download_button(
            "Download self-contained HTML report",
            data=path.read_bytes(),
            file_name="modelens_comparison_report.html",
            mime="text/html",
        )

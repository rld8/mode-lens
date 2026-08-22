"""Reusable Streamlit presentation components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modelens.domain.entities import AnalysisResult


def scope_notice() -> None:
    """Show the scientific-scope notice beside affected results."""
    st.warning(
        "Educational and controlled experiment only. ModeLens does not certify structural "
        "integrity and must not be used for safety decisions."
    )


def require_analysis() -> AnalysisResult:
    """Stop a page cleanly when no experiment has been analysed."""
    from modelens.interfaces.streamlit_app.state import current_analysis

    result = current_analysis()
    if result is None:
        st.info("Run the bundled demo or analyse a capture on the Experiment page first.")
        st.stop()
    return result


def modes_table(result: AnalysisResult) -> None:
    """Display modal values without replacing missing estimates by zero."""
    rows = []
    for mode in result.modal.modes:
        rows.append(
            {
                "Mode": mode.index,
                "Frequency [Hz]": round(mode.frequency_hz, 4),
                "Frequency CI [Hz]": "not identifiable"
                if mode.frequency_ci_hz is None
                else f"{mode.frequency_ci_hz[0]:.3f}–{mode.frequency_ci_hz[1]:.3f}",
                "Damping ratio": "not identifiable"
                if mode.damping_ratio is None
                else round(mode.damping_ratio, 5),
                "POD energy [%]": round(100.0 * mode.energy_fraction, 2),
                "Prominence": round(mode.prominence, 3),
                "Flags": ", ".join(mode.quality_flags),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def quality_table(result: AnalysisResult) -> None:
    """Display each measured capture-quality indicator."""
    rows = [
        {
            "Check": metric.name,
            "Value": f"{metric.value:.4g} {metric.unit}",
            "Threshold": metric.threshold,
            "Result": "Pass" if metric.passed else "Warning",
            "Meaning": metric.explanation,
        }
        for metric in result.quality.metrics
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

"""自动诊断页面。"""

import streamlit as st
from components import render_diagnostics


def render(runs, diagnostics):
    """渲染自动诊断 tab。"""
    st.subheader("规则引擎诊断")
    run_names = [r.name for r in runs]
    render_diagnostics(diagnostics, run_names)

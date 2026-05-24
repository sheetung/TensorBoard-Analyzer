"""配置参数页面。"""

import streamlit as st
from components import render_config_table


def render(runs):
    """渲染配置参数 tab。"""
    st.subheader("训练配置对比")
    for run in runs:
        with st.expander(f"📁 {run.name}", expanded=len(runs) == 1):
            if run.config:
                render_config_table(run.config)
            else:
                st.info("无配置信息（缺少 cfgs.pkl）")

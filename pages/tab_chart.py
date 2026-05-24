"""对比曲线页面。"""

import streamlit as st
from components import render_comparison_chart


def render(runs):
    """渲染对比曲线 tab。"""
    all_metric_keys = set()
    for run in runs:
        all_metric_keys.update(run.scalars.keys())
    available_metrics = sorted(all_metric_keys)

    if not available_metrics:
        st.warning("无训练指标数据")
        return

    default_metrics = available_metrics
    selected_metrics = st.multiselect(
        "选择要对比的指标",
        options=available_metrics,
        default=[m for m in default_metrics if m in available_metrics],
    )

    # 概览：reward 对比
    if "Train/mean_reward" in available_metrics:
        st.subheader("Reward 总览")
        fig = render_comparison_chart(runs, "Train/mean_reward")
        st.plotly_chart(fig, use_container_width=True)

    # 逐指标对比
    for metric in selected_metrics:
        if metric == "Train/mean_reward":
            continue
        fig = render_comparison_chart(runs, metric)
        st.plotly_chart(fig, use_container_width=True)

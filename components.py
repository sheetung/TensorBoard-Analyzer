"""共享渲染组件：图表、配置表、诊断展示。"""

import streamlit as st
import plotly.graph_objects as go


def render_comparison_chart(runs, metric_key):
    """渲染单个指标的对比折线图。"""
    fig = go.Figure()
    for run in runs:
        data = run.scalars.get(metric_key, {})
        if data.get("steps") and data.get("values"):
            fig.add_trace(go.Scatter(
                x=data["steps"],
                y=data["values"],
                mode="lines",
                name=run.name[:40],
                opacity=0.8,
            ))
    fig.update_layout(
        title=metric_key,
        xaxis_title="Iteration",
        yaxis_title="Value",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(font=dict(size=10)),
    )
    return fig


def render_config_table(config):
    """渲染配置参数表格。"""
    if not config:
        st.info("无配置信息")
        return

    for cfg_name, cfg_data in config.items():
        if not cfg_data or not isinstance(cfg_data, dict):
            continue
        with st.expander(f"📋 {cfg_name}", expanded=False):
            rows = []
            for k, v in cfg_data.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        rows.append({"参数": f"{k}.{kk}", "值": str(vv)})
                elif isinstance(v, list):
                    rows.append({"参数": k, "值": str(v)})
                else:
                    rows.append({"参数": k, "值": str(v)})
            st.table(rows)


def render_diagnostics(diagnostics_list, run_names):
    """渲染诊断结果。"""
    for diag, name in zip(diagnostics_list, run_names):
        st.subheader(f"🔍 {name}")

        if diag.get("issues"):
            st.error("**检测到的问题：**")
            for issue in diag["issues"]:
                st.write(f"- {issue}")

        if diag.get("suggestions"):
            st.warning("**调参建议：**")
            for s in diag["suggestions"]:
                st.write(f"- {s}")

        if not diag.get("issues"):
            st.success(diag.get("summary", "无问题"))

        if diag.get("metrics"):
            m = diag["metrics"]
            cols = st.columns(3)
            cols[0].metric("最终 Reward", f"{m.get('final_reward', 0):.4f}")
            cols[1].metric("峰值 Reward", f"{m.get('peak_reward', 0):.4f}")
            cols[2].metric("最后10轮 Reward", f"{m.get('final_reward_10', 0):.4f}")

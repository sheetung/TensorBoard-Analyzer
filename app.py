"""
TensorBoard Analyzer - 训练分析与 AI 诊断工具
基于 Streamlit 的 Web UI，支持 TensorBoard 数据对比和大模型调参建议。
"""

import os
import sys
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyzer import scan_log_dirs, load_run, compute_diagnostics, KEY_METRICS
from llm_advisor import query_llm, RECOMMENDED_MODELS

# 页面配置
st.set_page_config(
    page_title="TensorBoard Analyzer",
    page_icon="📊",
    layout="wide",
)

# 初始化 session state
if "llm_config" not in st.session_state:
    st.session_state.llm_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key": "",
        "base_url": "",
    }
if "runs" not in st.session_state:
    st.session_state.runs = []
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = []


def find_logs_dir():
    """查找 logs 目录，优先在 Flare 项目下查找。"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "Flare", "logs"),
        os.path.join(os.path.dirname(__file__), "logs"),
        "logs",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return os.path.join(os.path.dirname(__file__), "..", "Flare", "logs")


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
            # 扁平化嵌套 dict
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


def main():
    logs_dir = find_logs_dir()

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.title("📊 TensorBoard Analyzer")
        st.caption("训练对比分析 + AI 诊断")

        # 日志目录选择
        st.subheader("日志目录")
        logs_path = st.text_input("logs 路径", value=logs_dir, key="logs_path")

        available_runs = scan_log_dirs(logs_path)

        if not available_runs:
            st.warning("未找到训练日志")
            st.stop()

        # 显示可用训练列表
        run_names = [os.path.basename(p) for p in available_runs]
        selected = st.multiselect(
            "选择要对比的训练",
            options=run_names,
            default=run_names[:2] if len(run_names) >= 2 else run_names[:1],
            key="selected_runs",
        )

        # 加载按钮
        if st.button("🔄 加载数据", type="primary"):
            st.session_state.runs = []
            st.session_state.diagnostics = []
            for name in selected:
                path = os.path.join(logs_path, name)
                run = load_run(path)
                st.session_state.runs.append(run)
                diag = compute_diagnostics(run)
                st.session_state.diagnostics.append(diag)

        st.divider()

        # LLM 设置
        st.subheader("🤖 AI 诊断设置")
        with st.expander("LLM 配置", expanded=False):
            provider = st.selectbox(
                "Provider",
                ["anthropic", "openai", "deepseek", "ollama"],
                index=["anthropic", "openai", "deepseek", "ollama"].index(
                    st.session_state.llm_config["provider"]
                ),
                key="llm_provider",
            )
            recommended = RECOMMENDED_MODELS.get(provider, [])
            model_options = recommended + ["自定义"]
            # 尝试匹配当前模型到推荐列表
            current_model = st.session_state.llm_config["model"]
            # 去掉已有的 provider 前缀再匹配
            clean_model = current_model
            for pfx in ("anthropic/", "openai/", "deepseek/", "ollama/"):
                if clean_model.startswith(pfx):
                    clean_model = clean_model[len(pfx):]
            if clean_model in model_options:
                default_idx = model_options.index(clean_model)
            else:
                default_idx = len(model_options) - 1  # "自定义"
            model_choice = st.selectbox(
                "Model",
                model_options,
                index=default_idx,
                key="llm_model_choice",
            )
            if model_choice == "自定义":
                model = st.text_input(
                    "模型名称",
                    value=clean_model,
                    key="llm_model_custom",
                    placeholder="如 deepseek-chat",
                )
            else:
                model = model_choice
            api_key = st.text_input(
                "API Key",
                value=st.session_state.llm_config["api_key"],
                type="password",
                key="llm_api_key",
            )
            base_url = st.text_input(
                "Base URL（可选，Ollama 等自定义地址）",
                value=st.session_state.llm_config["base_url"],
                key="llm_base_url",
            )

            if st.button("保存 LLM 配置"):
                st.session_state.llm_config = {
                    "provider": provider,
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                }
                st.success("已保存")

    # ========== 主区域 ==========
    if not st.session_state.runs:
        st.info("👈 在左侧选择训练目录并点击「加载数据」")
        st.stop()

    runs = st.session_state.runs
    run_names = [r.name for r in runs]

    # Tab 布局
    tab_chart, tab_config, tab_diag, tab_ai = st.tabs(
        ["📈 对比曲线", "⚙️ 配置参数", "🔍 自动诊断", "🤖 AI 分析"]
    )

    # Tab 1: 对比曲线
    with tab_chart:
        # 选择要显示的指标
        available_metrics = []
        for key in KEY_METRICS:
            for run in runs:
                if key in run.scalars:
                    available_metrics.append(key)
                    break

        if not available_metrics:
            st.warning("无训练指标数据")
        else:
            default_metrics = [
                "Train/mean_reward",
                "Loss/value_function",
                "Loss/surrogate",
                "Episode/rew_crash",
                "Episode/rew_cable_angle_safety",
            ]
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
                    continue  # 已在上面显示
                fig = render_comparison_chart(runs, metric)
                st.plotly_chart(fig, use_container_width=True)

    # Tab 2: 配置参数
    with tab_config:
        st.subheader("训练配置对比")
        for run in runs:
            with st.expander(f"📁 {run.name}", expanded=len(runs) == 1):
                if run.config:
                    render_config_table(run.config)
                else:
                    st.info("无配置信息（缺少 cfgs.pkl）")

    # Tab 3: 自动诊断
    with tab_diag:
        st.subheader("规则引擎诊断")
        render_diagnostics(st.session_state.diagnostics, run_names)

    # Tab 4: AI 分析
    with tab_ai:
        st.subheader("🤖 AI 调参建议")

        if not st.session_state.llm_config.get("api_key"):
            st.warning("请先在左侧「AI 诊断设置」中配置 API Key")
        else:
            if st.button("🧠 开始 AI 分析", type="primary"):
                with st.spinner("正在分析训练数据并调用大模型..."):
                    # 构建对比上下文
                    comparison = ""
                    if len(runs) > 1:
                        comparison = (
                            f"本次分析对比了 {len(runs)} 次训练：\n"
                            + "\n".join(
                                f"- {r.name}（{r.total_iters} 轮）" for r in runs
                            )
                        )

                    result = query_llm(
                        runs=runs,
                        diagnostics_list=st.session_state.diagnostics,
                        config=st.session_state.llm_config,
                        comparison_context=comparison,
                    )

                st.markdown(result)

                with st.expander("📋 查看发送给 LLM 的上下文", expanded=False):
                    from llm_advisor import build_prompt
                    prompt = build_prompt(
                        runs,
                        st.session_state.diagnostics,
                        comparison,
                    )
                    st.code(prompt, language="text")


if __name__ == "__main__":
    main()

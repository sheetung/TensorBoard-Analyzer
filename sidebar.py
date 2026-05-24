"""侧边栏逻辑：日志扫描、LLM 配置 UI。"""

import os
import re
import streamlit as st
from core.llm_advisor import PROVIDER_PRESETS

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

_LLM_ENV_KEYS = ["LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "USER_PROMPT"]


def find_logs_dir():
    """查找 logs 目录，优先使用 .env 中的 LOGS_DIR 配置。"""
    env_dir = os.environ.get("LOGS_DIR", "")
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)
    candidates = [
        os.path.join(os.path.dirname(__file__), "logs"),
        os.path.join(os.path.dirname(__file__), "..", "logs"),
        "logs",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return os.path.join(os.path.dirname(__file__), "logs")


def get_llm_config():
    """从环境变量读取 LLM 配置。"""
    return {
        "provider": os.environ.get("LLM_PROVIDER", "deepseek"),
        "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "base_url": os.environ.get("LLM_BASE_URL", ""),
        "user_prompt": os.environ.get("USER_PROMPT", ""),
    }


def save_llm_config_to_env(config):
    """将 LLM 配置写回 .env 文件并更新当前进程环境变量。"""
    if not os.path.exists(ENV_FILE):
        return
    with open(ENV_FILE, "r") as f:
        content = f.read()
    mapping = {
        "LLM_PROVIDER": config["provider"],
        "LLM_MODEL": config["model"],
        "LLM_API_KEY": config["api_key"],
        "LLM_BASE_URL": config["base_url"],
        "USER_PROMPT": config["user_prompt"],
    }
    for key, value in mapping.items():
        pattern = rf'^{key}=".*"$'
        replacement = f'{key}="{value}"'
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content += f"\n{replacement}"
        os.environ[key] = value
    with open(ENV_FILE, "w") as f:
        f.write(content)


def render_sidebar(logs_dir):
    """渲染侧边栏，返回 (logs_path, selected_run_names)。"""
    with st.sidebar:
        st.title("📊 TensorBoard Analyzer")
        st.caption("训练对比分析 + AI 诊断")

        # 日志目录选择
        st.subheader("日志目录")
        logs_candidates = []
        for root_dir in [
            os.path.join(os.path.dirname(__file__), ".."),
            os.path.dirname(__file__),
            ".",
        ]:
            candidate = os.path.join(root_dir, "logs")
            if os.path.isdir(candidate):
                logs_candidates.append(os.path.abspath(candidate))
        if logs_dir not in logs_candidates:
            logs_candidates.insert(0, logs_dir)
        logs_candidates.append("自定义路径")
        logs_choice = st.selectbox("logs 路径", logs_candidates, key="logs_path")
        if logs_choice == "自定义路径":
            logs_path = st.text_input("输入自定义路径", key="logs_path_custom")
        else:
            logs_path = logs_choice

        from core.analyzer import scan_log_dirs
        available_runs = scan_log_dirs(logs_path)

        if not available_runs:
            st.warning("未找到训练日志")
            st.stop()

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
                from core.analyzer import load_run, compute_diagnostics
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
                ["deepseek", "openai", "anthropic", "ollama"],
                index=["deepseek", "openai", "anthropic", "ollama"].index(
                    st.session_state.llm_config["provider"]
                ),
                key="llm_provider",
            )
            preset = PROVIDER_PRESETS.get(provider, {})
            model_options = preset.get("models", []) + ["自定义"]
            current_model = st.session_state.llm_config["model"]
            if current_model in model_options:
                default_idx = model_options.index(current_model)
            else:
                default_idx = len(model_options) - 1
            model_choice = st.selectbox(
                "Model",
                model_options,
                index=default_idx,
                key="llm_model_choice",
            )
            if model_choice == "自定义":
                model = st.text_input(
                    "模型名称",
                    value=current_model,
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
            default_base_url = preset.get("base_url", "")
            base_url = st.text_input(
                "Base URL（留空使用默认地址）",
                value=st.session_state.llm_config.get("base_url", "") or default_base_url,
                key="llm_base_url",
            )

            user_prompt = st.text_area(
                "用户自定义提示（可选）",
                value=os.environ.get("USER_PROMPT", ""),
                placeholder="描述你的项目背景、训练目标、特殊约束等，帮助 AI 给出更精准的建议",
                key="llm_user_prompt",
                height=100,
            )

            if st.button("保存 LLM 配置"):
                new_config = {
                    "provider": provider,
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "user_prompt": user_prompt,
                }
                st.session_state.llm_config = new_config
                save_llm_config_to_env(new_config)
                st.toast("已保存", icon="✅")

    return logs_path, selected

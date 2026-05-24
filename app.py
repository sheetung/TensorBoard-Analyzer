"""
TensorBoard Analyzer - 训练分析与 AI 诊断工具
基于 Streamlit 的 Web UI，支持 TensorBoard 数据对比和大模型调参建议。
"""

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from sidebar import find_logs_dir, get_llm_config, render_sidebar
from tabs import tab_chart, tab_config, tab_diag, tab_ai, tab_chat

# 页面配置
st.set_page_config(
    page_title="TensorBoard Analyzer",
    page_icon="📊",
    layout="wide",
)

# 聊天输入框固定在底部
st.markdown("""
<style>
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    right: 40px !important;
    z-index: 999 !important;
}
</style>
<script>
function adjustChatInput() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    const chatInput = document.querySelector('[data-testid="stChatInput"]');
    if (!sidebar || !chatInput) return;
    chatInput.style.left = sidebar.offsetWidth > 50 ? '340px' : '40px';
}
document.addEventListener('DOMContentLoaded', function() {
    adjustChatInput();
    const main = document.querySelector('[data-testid="stApp"]') || document.body;
    new MutationObserver(adjustChatInput).observe(main, {childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class']});
});
</script>
""", unsafe_allow_html=True)

# 初始化 session state
if "llm_config" not in st.session_state:
    st.session_state.llm_config = get_llm_config()
if "runs" not in st.session_state:
    st.session_state.runs = []
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_system_prompt" not in st.session_state:
    st.session_state.chat_system_prompt = ""
if "chat_context_id" not in st.session_state:
    st.session_state.chat_context_id = ""


def main():
    logs_dir = find_logs_dir()

    # 渲染侧边栏
    logs_path, selected = render_sidebar(logs_dir)

    # 主区域
    if not st.session_state.runs:
        st.info("👈 在左侧选择训练目录并点击「加载数据」")
        st.stop()

    runs = st.session_state.runs

    # Tab 布局
    tab_chart_ui, tab_config_ui, tab_diag_ui, tab_ai_ui, tab_chat_ui = st.tabs(
        ["📈 对比曲线", "⚙️ 配置参数", "🔍 自动诊断", "🤖 AI 分析", "💬 AI 对话"]
    )

    with tab_chart_ui:
        tab_chart.render(runs)

    with tab_config_ui:
        tab_config.render(runs)

    with tab_diag_ui:
        tab_diag.render(runs, st.session_state.diagnostics)

    with tab_ai_ui:
        tab_ai.render(runs, st.session_state.diagnostics, st.session_state.llm_config)

    with tab_chat_ui:
        tab_chat.render(runs, st.session_state.diagnostics, st.session_state.llm_config)


if __name__ == "__main__":
    main()

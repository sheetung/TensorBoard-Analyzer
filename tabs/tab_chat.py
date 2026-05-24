"""AI 对话页面。"""

import os
import streamlit as st
import streamlit.components.v1 as components
from core.ai_chat import init_chat_messages, chat_llm


def inject_fixed_chat_input_css():
    """固定聊天输入框到底部，并根据主内容区位置自适应侧边栏。"""

    st.markdown(
        """
        <style>
        /* 给底部留白，防止最后一条消息被输入框遮住 */
        .block-container {
            padding-bottom: 7rem !important;
        }

        /* 固定输入框，但 left/width 由 JS 动态设置 */
        [data-testid="stChatInput"] {
            position: fixed !important;
            bottom: 0.75rem !important;
            right: auto !important;
            z-index: 9999 !important;
            transform: none !important;
            background: var(--background-color) !important;
            padding: 0.5rem 0 0.75rem 0 !important;
        }

        [data-testid="stChatInput"] > div {
            max-width: none !important;
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        """
        <script>
        function updateChatInputPosition() {
            const doc = window.parent.document;

            const chatInput = doc.querySelector('[data-testid="stChatInput"]');
            if (!chatInput) return;

            const blockContainer =
                doc.querySelector('[data-testid="stAppViewContainer"] .block-container') ||
                doc.querySelector('section.main .block-container') ||
                doc.querySelector('main .block-container') ||
                doc.querySelector('.block-container');

            if (!blockContainer) return;

            const rect = blockContainer.getBoundingClientRect();

            chatInput.style.setProperty("left", rect.left + "px", "important");
            chatInput.style.setProperty("width", rect.width + "px", "important");
            chatInput.style.setProperty("right", "auto", "important");
            chatInput.style.setProperty("transform", "none", "important");
            chatInput.style.setProperty("bottom", "0.75rem", "important");
        }

        updateChatInputPosition();

        window.parent.addEventListener("resize", updateChatInputPosition);

        const observer = new MutationObserver(updateChatInputPosition);
        observer.observe(window.parent.document.body, {
            attributes: true,
            childList: true,
            subtree: true,
        });

        const timer = setInterval(updateChatInputPosition, 200);

        setTimeout(() => {
            clearInterval(timer);
            updateChatInputPosition();
        }, 6000);
        </script>
        """,
        height=0,
        width=0,
    )


def render(runs, diagnostics, llm_config):
    """渲染 AI 对话 tab。"""

    inject_fixed_chat_input_css()

    st.subheader("💬 AI 对话")

    provider = llm_config.get("provider", "")
    api_key = llm_config.get("api_key", "")

    if provider != "ollama" and not api_key:
        st.warning("请先在左侧「AI 诊断设置」中配置 API Key")
        return

    # 生成当前训练数据的上下文 ID
    current_context_id = "|".join(r.name for r in runs)

    # 检测训练数据是否更新，自动重置对话
    if st.session_state.chat_context_id and st.session_state.chat_context_id != current_context_id:
        st.session_state.chat_messages = []
        st.session_state.chat_system_prompt = ""
        st.session_state.chat_context_id = ""
        st.toast("训练数据已更新，对话已重置", icon="🔄")

    # 重置对话按钮
    if st.button("🔄 重置对话", key="reset_chat"):
        st.session_state.chat_messages = []
        st.session_state.chat_system_prompt = ""
        st.session_state.chat_context_id = ""
        st.rerun()

    # 渲染历史消息
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入
    if user_input := st.chat_input("输入你的问题，例如：为什么 reward 在第 500 轮后下降？"):
        # 首次对话时初始化 system prompt
        if not st.session_state.chat_context_id:
            system_prompt, messages = init_chat_messages(
                runs=runs,
                diagnostics_list=diagnostics,
                user_prompt=os.environ.get("USER_PROMPT", ""),
            )
            st.session_state.chat_system_prompt = system_prompt
            st.session_state.chat_messages = messages
            st.session_state.chat_context_id = current_context_id

        # 添加用户消息
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # 调用 LLM
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = chat_llm(
                    system_prompt=st.session_state.chat_system_prompt,
                    messages=st.session_state.chat_messages,
                    config=llm_config,
                )
            st.markdown(response)

        # 保存助手回复
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
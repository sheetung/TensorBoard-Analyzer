"""AI 对话页面。"""

import os
from html import escape

import streamlit as st
import streamlit.components.v1 as components

from core.ai_chat import init_chat_messages, chat_llm

def inject_fixed_chat_input_css():
    """
    固定聊天输入框到底部，并尽量做成 ChatGPT 风格。

    特点：
    1. 白色胶囊输入框
    2. 没有左侧 + 号
    3. 没有内部灰背景
    4. 发送按钮在输入框最右边
    5. 侧边栏展开/收起时跟随主内容区移动
    """

    st.markdown(
        """
        <style>
        .block-container {
            padding-bottom: 7rem !important;
        }

        /*
         * 最外层：唯一输入框外壳
         */
        [data-testid="stChatInput"] {
            position: fixed !important;
            bottom: 1.15rem !important;
            right: auto !important;
            z-index: 9999 !important;
            transform: none !important;

            height: 60px !important;
            min-height: 60px !important;
            max-height: 60px !important;

            background: #ffffff !important;
            border: 1px solid rgba(0, 0, 0, 0.08) !important;
            border-radius: 999px !important;
            box-shadow: none !important;

            padding: 0 56px 0 18px !important;
            margin: 0 !important;
            box-sizing: border-box !important;
            overflow: hidden !important;

            display: flex !important;
            align-items: center !important;

            will-change: left, width !important;
        }

        /*
         * 内部所有容器透明，不要出现第二层灰背景
         */
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] form,
        [data-testid="stChatInput"] form > div {
            width: 100% !important;
            height: 100% !important;
            min-height: 0 !important;
            max-height: none !important;

            padding: 0 !important;
            margin: 0 !important;

            background: transparent !important;
            border: none !important;
            box-shadow: none !important;

            display: flex !important;
            align-items: center !important;
            box-sizing: border-box !important;
        }

        /*
         * 文字输入区域：透明、无背景、无边框
         */
        [data-testid="stChatInput"] textarea {
            width: 100% !important;
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;

            line-height: 28px !important;
            padding: 0 !important;
            margin: 0 !important;

            background: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;

            resize: none !important;
            overflow-y: hidden !important;
            box-sizing: border-box !important;
            font-size: 1rem !important;
        }

        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] textarea:focus-visible {
            background: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /*
         * 发送按钮：贴在最右边
         */
        [data-testid="stChatInput"] button {
            position: absolute !important;
            right: 8px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;

            width: 38px !important;
            height: 38px !important;
            min-width: 38px !important;
            min-height: 38px !important;
            max-width: 38px !important;
            max-height: 38px !important;

            padding: 0 !important;
            margin: 0 !important;
            border-radius: 999px !important;

            display: flex !important;
            align-items: center !important;
            justify-content: center !important;

            box-sizing: border-box !important;
            border: none !important;
            box-shadow: none !important;
        }

        /*
         * 有内容时按钮
         */
        [data-testid="stChatInput"] button:not(:disabled) {
            background: #111111 !important;
            color: #ffffff !important;
        }

        [data-testid="stChatInput"] button:not(:disabled) svg {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }

        /*
         * 无内容时按钮
         */
        [data-testid="stChatInput"] button:disabled {
            background: #f1f3f5 !important;
            color: #9aa0aa !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button:disabled svg {
            color: #9aa0aa !important;
            fill: #9aa0aa !important;
            stroke: #9aa0aa !important;
        }

        /*
         * 用户消息：右侧气泡，内容右对齐
         */
        .user-message-row {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 0.85rem;
        }

        .user-message-box {
            max-width: 58%;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            text-align: right;
            word-break: break-word;
            line-height: 1.6;
        }

        /*
         * AI 回复正常显示
         */
        .assistant-message-block {
            margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        """
        <script>
        const doc = window.parent.document;

        let rafId = null;
        let lastLeft = null;
        let lastWidth = null;

        function getBlockContainer() {
            return (
                doc.querySelector('[data-testid="stAppViewContainer"] .block-container') ||
                doc.querySelector('section.main .block-container') ||
                doc.querySelector('main .block-container') ||
                doc.querySelector('.block-container')
            );
        }

        function updateChatInputPosition() {
            const chatInput = doc.querySelector('[data-testid="stChatInput"]');
            const blockContainer = getBlockContainer();

            if (!chatInput || !blockContainer) {
                return;
            }

            const rect = blockContainer.getBoundingClientRect();

            const maxWidth = 920;
            const sidePadding = 24;
            const minWidth = 360;

            const targetWidth = Math.min(
                maxWidth,
                Math.max(minWidth, rect.width - sidePadding * 2)
            );

            const targetLeft = rect.left + (rect.width - targetWidth) / 2;

            if (
                lastLeft !== null &&
                Math.abs(lastLeft - targetLeft) < 0.5 &&
                Math.abs(lastWidth - targetWidth) < 0.5
            ) {
                return;
            }

            lastLeft = targetLeft;
            lastWidth = targetWidth;

            chatInput.style.setProperty("left", targetLeft + "px", "important");
            chatInput.style.setProperty("width", targetWidth + "px", "important");
            chatInput.style.setProperty("right", "auto", "important");
            chatInput.style.setProperty("transform", "none", "important");
            chatInput.style.setProperty("bottom", "1.15rem", "important");
        }

        function smoothTrack(duration = 450) {
            const start = performance.now();

            function frame(now) {
                updateChatInputPosition();

                if (now - start < duration) {
                    rafId = window.parent.requestAnimationFrame(frame);
                } else {
                    updateChatInputPosition();
                    rafId = null;
                }
            }

            if (rafId) {
                window.parent.cancelAnimationFrame(rafId);
            }

            rafId = window.parent.requestAnimationFrame(frame);
        }

        smoothTrack(800);

        window.parent.addEventListener("resize", () => {
            smoothTrack(500);
        });

        const observer = new MutationObserver(() => {
            smoothTrack(500);
        });

        observer.observe(doc.body, {
            attributes: true,
            childList: true,
            subtree: true,
        });

        const blockContainer = getBlockContainer();

        if (blockContainer && "ResizeObserver" in window.parent) {
            const resizeObserver = new window.parent.ResizeObserver(() => {
                smoothTrack(300);
            });

            resizeObserver.observe(blockContainer);
        }
        </script>
        """,
        height=0,
        width=0,
    )


def render_message(role, content):
    """
    渲染对话消息。

    - AI 回复：正常显示，不分左右，不显示头像
    - 用户输入：显示在右边，内容右对齐，不显示头像
    - 不显示“用户”和“AI”文字
    """

    if role == "assistant":
        st.markdown('<div class="assistant-message-block">', unsafe_allow_html=True)
        st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)

    elif role == "user":
        safe_content = escape(content).replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="user-message-row">
                <div class="user-message-box">
                    {safe_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
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
    if (
        st.session_state.chat_context_id
        and st.session_state.chat_context_id != current_context_id
    ):
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
    # 前两条是初始化上下文消息，不展示给用户
    for idx, msg in enumerate(st.session_state.chat_messages):
        if idx < 2:
            continue

        render_message(msg["role"], msg["content"])

    # 聊天输入
    user_input = st.chat_input(
        "输入你的问题，例如：为什么 reward 在第 500 轮后下降？"
    )

    if user_input:
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
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # 用户输入显示在右侧
        render_message("user", user_input)

        # 调用 LLM
        with st.spinner("思考中..."):
            response = chat_llm(
                system_prompt=st.session_state.chat_system_prompt,
                messages=st.session_state.chat_messages,
                config=llm_config,
            )

        # AI 回复正常显示，不分左右
        render_message("assistant", response)

        # 保存助手回复
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
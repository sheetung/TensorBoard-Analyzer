"""AI 对话页面。"""

import os
from html import escape

import streamlit as st
import streamlit.components.v1 as components

from core.ai_chat import init_chat_messages, chat_llm


def inject_fixed_chat_input_css():
    """
    固定聊天输入框到底部。

    效果：
    1. 类似 ChatGPT，不占满整个页面
    2. 固定在浏览器底部
    3. 侧边栏展开/收起时，输入框跟随主内容区域移动
    4. 使用 requestAnimationFrame，减少动画卡顿
    """

    st.markdown(
        """
        <style>
        .block-container {
            padding-bottom: 8rem !important;
        }

        [data-testid="stChatInput"] {
            position: fixed !important;
            bottom: 1.25rem !important;
            right: auto !important;
            z-index: 9999 !important;
            transform: none !important;
            background: transparent !important;
            padding: 0 !important;
            will-change: left, width !important;
        }

        [data-testid="stChatInput"] > div {
            width: 100% !important;
            max-width: none !important;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 1.5rem !important;
            max-height: 48px !important;
            height: 48px !important;
            resize: none !important;
            overflow-y: auto !important;
        }

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

            const maxWidth = 600;
            const sidePadding = 24;
            const minWidth = 280;

            const targetWidth = Math.min(
                maxWidth,
                Math.max(minWidth, rect.width - sidePadding * 2)
            );

            const targetLeft = rect.left + (rect.width - targetWidth) / 2;

            /*
             * 避免每一帧都重复写 style。
             * 只有位置/宽度真的变化时才更新，减少卡顿。
             */
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
            chatInput.style.setProperty("bottom", "1.25rem", "important");
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

        /*
         * 首次加载
         */
        smoothTrack(800);

        /*
         * 浏览器尺寸变化
         */
        window.parent.addEventListener("resize", () => {
            smoothTrack(500);
        });

        /*
         * 监听 Streamlit DOM 变化。
         * 侧边栏展开/收起会触发 DOM/class/style 变化。
         * 不直接频繁更新，而是启动一小段 rAF 跟踪。
         */
        const observer = new MutationObserver(() => {
            smoothTrack(500);
        });

        observer.observe(doc.body, {
            attributes: true,
            childList: true,
            subtree: true,
        });

        /*
         * 监听主内容区尺寸变化。
         */
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

    约定：
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
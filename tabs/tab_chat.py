"""AI 对话页面。"""

import os
import streamlit as st
from core.ai_chat import init_chat_messages, chat_llm


def render(runs, diagnostics, llm_config):
    """渲染 AI 对话 tab。"""
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

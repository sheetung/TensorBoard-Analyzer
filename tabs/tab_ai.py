"""AI 分析页面。"""

import os
import streamlit as st
from core.llm_advisor import query_llm, build_prompt


def render(runs, diagnostics, llm_config):
    """渲染 AI 分析 tab。"""
    st.subheader("🤖 AI 调参建议")

    if not llm_config.get("api_key"):
        st.warning("请先在左侧「AI 诊断设置」中配置 API Key")
        return

    if st.button("🧠 开始 AI 分析", type="primary"):
        with st.spinner("正在分析训练数据并调用大模型..."):
            comparison = ""
            if len(runs) > 1:
                comparison = (
                    f"本次分析对比了 {len(runs)} 次训练：\n"
                    + "\n".join(f"- {r.name}（{r.total_iters} 轮）" for r in runs)
                )

            result = query_llm(
                runs=runs,
                diagnostics_list=diagnostics,
                config=llm_config,
                comparison_context=comparison,
                user_prompt=os.environ.get("USER_PROMPT", ""),
            )

        st.markdown(result)

        with st.expander("📋 查看发送给 LLM 的上下文", expanded=False):
            system_content, user_content = build_prompt(
                runs,
                diagnostics,
                comparison,
                os.environ.get("USER_PROMPT", ""),
            )
            st.code(f"[system]\n{system_content}\n\n[user]\n{user_content}", language="text")

"""
AI 对话模块。
支持多轮对话，首次自动注入训练数据上下文，
Anthropic 独立处理 system prompt 参数。
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .llm_advisor import (
    DEFAULT_CONFIG,
    PROVIDER_PRESETS,
    SYSTEM_PROMPT,
    build_prompt,
    _call_openai_compatible,
)


def build_training_context(runs, diagnostics_list, user_prompt=""):
    """
    将当前训练数据格式化为上下文文本。

    Returns:
        user_content: 包含训练数据的文本
    """
    comparison_context = ""
    if len(runs) > 1:
        comparison_context = (
            f"本次分析对比了 {len(runs)} 次训练：\n"
            + "\n".join(f"- {r.name}（{r.total_iters} 轮）" for r in runs)
        )

    _, user_content = build_prompt(
        runs, diagnostics_list, comparison_context, user_prompt
    )
    return user_content


def init_chat_messages(runs, diagnostics_list, user_prompt=""):
    """
    初始化多轮对话 messages。

    Returns:
        system_prompt: AI 角色设定 + 训练数据上下文（不展示给用户）
        messages: OpenAI messages 格式，包含训练数据上下文
    """
    training_context = build_training_context(runs, diagnostics_list, user_prompt)

    system_prompt = f"""{SYSTEM_PROMPT}

你是 TensorBoard Analyzer 的 AI 助手，专门分析强化学习训练数据。

以下是当前加载的训练数据上下文（你基于这些数据回答问题，不要在回复中重复展示这些原始数据）：

{training_context}"""

    messages = [
        {"role": "user", "content": "我已加载了训练数据，请基于这些数据回答我的问题。"},
        {
            "role": "assistant",
            "content": "好的，我已读取训练数据。你可以问我关于训练表现、参数配置、调参建议等问题。",
        },
    ]

    return system_prompt, messages


def chat_llm(system_prompt, messages, config=None):
    """
    调用 LLM 进行多轮对话。

    Args:
        system_prompt: 系统提示词（含训练数据上下文）
        messages: 消息历史（不含 system），格式 [{"role": "user"/"assistant", "content": "..."}]
        config: LLM 配置 dict

    Returns:
        assistant 回复文本
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    provider = cfg["provider"] or os.environ.get("LLM_PROVIDER", "deepseek")
    model = cfg["model"] or os.environ.get("LLM_MODEL", "deepseek-chat")
    api_key = cfg["api_key"] or os.environ.get("LLM_API_KEY", "")
    base_url = cfg["base_url"] or os.environ.get("LLM_BASE_URL", "")

    preset = PROVIDER_PRESETS.get(provider, {})
    if not base_url:
        base_url = preset.get("base_url", "")

    if not api_key and provider != "ollama":
        return "错误：请先在设置页面配置 API Key。"

    try:
        if provider == "anthropic":
            return _call_anthropic_with_system(model, api_key, system_prompt, messages, cfg["temperature"])
        else:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            return _call_openai_compatible(model, api_key, base_url, full_messages, cfg["temperature"])
    except Exception as e:
        return f"LLM 调用失败: {e}\n\n请检查 API Key 和网络连接。"


def _call_anthropic_with_system(model, api_key, system_prompt, messages, temperature):
    """
    调用 Anthropic Claude API，system prompt 单独传入。

    Anthropic API 的 system 参数与 messages 分离，
    不能把 system 作为普通 message 传入。
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=16384,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text

"""
AI 对话模块。

支持：
1. 多轮对话
2. 首次自动注入训练数据上下文
3. Anthropic 独立处理 system prompt 参数
4. 对话模式不复用 llm_advisor.py 中的一次性报告 SYSTEM_PROMPT
5. 自动清理一级标题，避免聊天气泡排版过大
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .llm_advisor import (
    DEFAULT_CONFIG,
    PROVIDER_PRESETS,
    build_prompt,
    _call_openai_compatible,
)


CHAT_SYSTEM_PROMPT = """你是 TensorBoard Analyzer 的 AI 对话助手，专门帮助用户分析强化学习训练数据。

你已经获得了当前加载的 TensorBoard 训练数据，包括：
1. TensorBoard scalar 指标
2. cfgs.pkl 配置参数
3. 规则诊断结果
4. 多个训练 run 的对比信息
5. 用户补充的项目背景

回答要求：
- 用中文回答
- 这是连续对话，不是一次性分析报告
- 用户问什么就答什么，不要每次重复完整报告
- 不要以“本次训练对比内容”“问题诊断”“调参建议”等大标题开头
- 默认不要使用一级标题
- 不要输出 Markdown 一级标题，也就是不要使用以 "# " 开头的标题
- 如果需要分段，最多使用三级标题，例如“### 主要原因”
- 回答要像正常聊天一样自然
- 建议要具体，尽量结合 reward、loss、noise、crash、安全约束、配置参数回答
- 如果用户问“为什么”，要说明现象、可能原因和验证方式
- 如果用户问“怎么改”，要给出优先级、参数修改方向和预期效果
- 如果用户问“哪个 run 更好”，要结合已有数据进行对比
"""


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
        runs,
        diagnostics_list,
        comparison_context,
        user_prompt,
    )

    return user_content


def init_chat_messages(runs, diagnostics_list, user_prompt=""):
    """
    初始化多轮对话 messages。

    Returns:
        system_prompt: AI 角色设定 + 训练数据上下文，不展示给用户
        messages: OpenAI messages 格式，不包含 system
    """
    training_context = build_training_context(
        runs=runs,
        diagnostics_list=diagnostics_list,
        user_prompt=user_prompt,
    )

    system_prompt = f"""{CHAT_SYSTEM_PROMPT}

以下是当前加载的训练数据上下文。
你需要基于这些数据回答用户后续问题，但不要在回复中重复展示完整原始数据。

{training_context}
"""

    messages = [
        {
            "role": "user",
            "content": "我已加载训练数据，请基于这些数据回答我的问题。",
        },
        {
            "role": "assistant",
            "content": "好的，我已读取当前训练数据。你可以直接问我具体问题。",
        },
    ]

    return system_prompt, messages


def clean_chat_response(text: str) -> str:
    """
    清理 AI 对话回复格式。

    主要作用：
    1. 防止模型输出过大的一级标题
    2. 把 # 标题自动降级为 ### 标题
    """
    if not text:
        return text

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        # 只处理一级标题：# xxx
        # 不处理 ## / ###，避免破坏正常小标题
        if stripped.startswith("# ") and not stripped.startswith("## "):
            line = indent + "### " + stripped[2:].strip()

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def chat_llm(system_prompt, messages, config=None):
    """
    调用 LLM 进行多轮对话。

    Args:
        system_prompt: 系统提示词，含训练数据上下文
        messages: 消息历史，不含 system
        config: LLM 配置 dict

    Returns:
        assistant 回复文本
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    provider = cfg["provider"] or os.environ.get("LLM_PROVIDER", "deepseek")
    model = cfg["model"] or os.environ.get("LLM_MODEL", "deepseek-chat")
    api_key = cfg["api_key"] or os.environ.get("LLM_API_KEY", "")
    base_url = cfg["base_url"] or os.environ.get("LLM_BASE_URL", "")
    temperature = cfg.get("temperature", 0.3)

    preset = PROVIDER_PRESETS.get(provider, {})
    if not base_url:
        base_url = preset.get("base_url", "")

    if not api_key and provider != "ollama":
        return "错误：请先在设置页面配置 API Key。"

    try:
        if provider == "anthropic":
            result = _call_anthropic_with_system(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
            )
            return clean_chat_response(result)

        full_messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ] + messages

        result = _call_openai_compatible(
            model=model,
            api_key=api_key,
            base_url=base_url,
            messages=full_messages,
            temperature=temperature,
        )

        return clean_chat_response(result)

    except Exception as e:
        return f"LLM 调用失败: {e}\n\n请检查 API Key、Base URL、模型名称和网络连接。"


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
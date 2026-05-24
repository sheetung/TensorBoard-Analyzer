"""
LLM 诊断模块。
使用 OpenAI SDK 调用 OpenAI 兼容接口（OpenAI / DeepSeek / Ollama 等），
使用 Anthropic SDK 调用 Claude，根据训练指标和配置生成调参建议。
"""

import os
from dotenv import load_dotenv

load_dotenv()


# 默认配置
DEFAULT_CONFIG = {
    "provider": "deepseek",         # deepseek / openai / anthropic / ollama
    "model": "deepseek-v4-flash",
    "api_key": "",
    "base_url": "",                 # 自定义 API 地址（Ollama 等）
    "temperature": 0.3,
}

# 各 provider 推荐的模型名和默认 base_url
PROVIDER_PRESETS = {
    "deepseek": {
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "base_url": "https://api.openai.com/v1",
    },
    "anthropic": {
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250414"],
        "base_url": "",
    },
    "ollama": {
        "models": ["llama3", "qwen2", "mistral"],
        "base_url": "http://localhost:11434/v1",
    },
}

DEFAULT_SYSTEM_PROMPT = """你是一个强化学习训练专家，专门分析 PPO 等 RL 算法的训练表现。

你的任务是：
1. 根据提供的训练指标和配置参数，诊断训练中的问题
2. 给出具体、可操作的调参建议
3. 按优先级排列建议，说明每个建议的原因和预期效果

输出格式要求：
- 用中文回答
- 分为"## 问题诊断"和"## 调参建议"两个章节
- 问题诊断：简要列出当前训练的主要问题
- 调参建议：按优先级排列，最多 5 条，每条格式如下：
  1. **参数名** — 当前值 → 建议值
     原因和预期效果

不要在每条建议前重复"参数名 → 当前值 → 建议值 → 预期效果"这个标题。"""

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


def format_config_for_llm(config):
    """将 cfgs.pkl 中的配置格式化为可读文本。"""
    if not config:
        return "（无配置信息）"

    lines = []
    for cfg_name, cfg_data in config.items():
        if not cfg_data:
            continue
        lines.append(f"\n### {cfg_name}")
        if isinstance(cfg_data, dict):
            for k, v in cfg_data.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for kk, vv in v.items():
                        lines.append(f"    {kk}: {vv}")
                else:
                    lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {cfg_data}")
    return "\n".join(lines)


def format_diagnostics_for_llm(diagnostics):
    """将诊断结果格式化为 LLM 可读文本。"""
    return ""


def _phase_stats(vals, n):
    """计算某一段的均值和标准差。"""
    import math
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def format_run_for_llm(run):
    """将单次训练运行格式化为上下文。"""
    lines = [f"## 训练运行: {run.name}"]
    lines.append(f"总迭代数: {run.total_iters}")

    for key, values in run.scalars.items():
        if not values.get("values"):
            continue
        vals = values["values"]
        n = len(vals)
        lines.append(f"\n### {key}")

        # 基础统计
        lines.append(f"  起始值: {vals[0]:.6f}")
        lines.append(f"  最终值: {vals[-1]:.6f}")
        lines.append(f"  峰值: {max(vals):.6f}（第{vals.index(max(vals))}轮）")

        # 分阶段统计：前1/4、中1/2、后1/4
        q = max(n // 4, 1)
        early_mean, early_std = _phase_stats(vals[:q], q)
        mid_mean, mid_std = _phase_stats(vals[q:-q] if n > 2 * q else vals, max(len(vals[q:-q] if n > 2 * q else vals), 1))
        late_mean, late_std = _phase_stats(vals[-q:], q)

        lines.append(f"  前1/4均值: {early_mean:.6f}（标准差: {early_std:.6f}）")
        lines.append(f"  中段均值: {mid_mean:.6f}（标准差: {mid_std:.6f}）")
        lines.append(f"  后1/4均值: {late_mean:.6f}（标准差: {late_std:.6f}）")

    return "\n".join(lines)


def build_prompt(runs, diagnostics_list, comparison_context="", user_prompt=""):
    """构建 LLM prompt，返回 (system_content, user_content)。"""
    system_content = SYSTEM_PROMPT
    parts = []

    if user_prompt.strip():
        parts.append(f"## 用户补充说明\n{user_prompt.strip()}")

    if comparison_context:
        parts.append(comparison_context)

    for run, diag in zip(runs, diagnostics_list):
        parts.append(format_run_for_llm(run))
        parts.append(format_config_for_llm(run.config))
        parts.append(format_diagnostics_for_llm(diag))
        parts.append("\n---\n")

    if len(runs) > 1:
        parts.append("请对比以上多次训练，分析差异原因，并给出综合调参建议。")

    return system_content, "\n".join(parts)


def query_llm(runs, diagnostics_list, config=None, comparison_context="", user_prompt=""):
    """
    调用 LLM 进行训练诊断。

    Args:
        runs: RunData 列表
        diagnostics_list: 对应的诊断结果列表
        config: LLM 配置 dict（provider, model, api_key, base_url）
        comparison_context: 额外的对比上下文
        user_prompt: 用户自定义补充提示

    Returns:
        LLM 生成的建议文本
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    provider = cfg["provider"] or os.environ.get("LLM_PROVIDER", "deepseek")
    model = cfg["model"] or os.environ.get("LLM_MODEL", "deepseek-chat")
    api_key = cfg["api_key"] or os.environ.get("LLM_API_KEY", "")
    base_url = cfg["base_url"] or os.environ.get("LLM_BASE_URL", "")

    # 从 provider 预设中取默认 base_url（用户未自定义时）
    preset = PROVIDER_PRESETS.get(provider, {})
    if not base_url:
        base_url = preset.get("base_url", "")

    if not api_key and provider != "ollama":
        return "错误：请先在设置页面配置 API Key。"

    system_content, user_content = build_prompt(runs, diagnostics_list, comparison_context, user_prompt)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        if provider == "anthropic":
            return _call_anthropic(model, api_key, messages, cfg["temperature"])
        else:
            # OpenAI / DeepSeek / Ollama 都走 OpenAI 兼容接口
            return _call_openai_compatible(model, api_key, base_url, messages, cfg["temperature"])
    except Exception as e:
        return f"LLM 调用失败: {e}\n\n请检查 API Key 和网络连接。"


def _call_openai_compatible(model, api_key, base_url, messages, temperature):
    """调用 OpenAI 兼容接口（OpenAI / DeepSeek / Ollama）。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=16384,
    )
    return response.choices[0].message.content


def _call_anthropic(model, api_key, messages, temperature):
    """调用 Anthropic Claude API。"""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=16384,
        temperature=temperature,
        messages=messages,
    )
    return response.content[0].text

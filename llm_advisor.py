"""
LLM 诊断模块。
支持多种大模型 provider（Anthropic / OpenAI / DeepSeek / Ollama 等），
根据训练指标和配置生成调参建议。
"""

import os
import json
from litellm import completion


# 默认配置
DEFAULT_CONFIG = {
    "provider": "anthropic",        # anthropic / openai / deepseek / ollama
    "model": "claude-sonnet-4-20250514",
    "api_key": "",
    "base_url": "",                 # 自定义 API 地址（Ollama 等）
    "temperature": 0.3,
}

SYSTEM_PROMPT = """你是一个强化学习训练专家，专门分析 PPO 算法在四旋翼无人机-缆绳-负载系统上的训练表现。

你的任务是：
1. 根据提供的训练指标和配置参数，诊断训练中的问题
2. 给出具体、可操作的调参建议
3. 按优先级排列建议，说明每个建议的原因和预期效果

输出格式要求：
- 用中文回答
- 分为"问题诊断"和"调参建议"两部分
- 每条建议给出：参数名 → 当前值 → 建议值 → 预期效果
- 最多给出 5 条建议，按优先级排列"""


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
    lines = []
    if diagnostics.get("issues"):
        lines.append("### 已检测到的问题")
        for issue in diagnostics["issues"]:
            lines.append(f"- {issue}")
    if diagnostics.get("suggestions"):
        lines.append("\n### 规则引擎建议")
        for s in diagnostics["suggestions"]:
            lines.append(f"- {s}")
    return "\n".join(lines)


def format_run_for_llm(run):
    """将单次训练运行格式化为上下文。"""
    lines = [f"## 训练运行: {run.name}"]
    lines.append(f"总迭代数: {run.total_iters}")

    # 关键指标摘要
    for key, values in run.scalars.items():
        if not values.get("values"):
            continue
        vals = values["values"]
        lines.append(f"\n### {key}")
        lines.append(f"  起始值: {vals[0]:.6f}")
        lines.append(f"  最终值: {vals[-1]:.6f}")
        lines.append(f"  峰值: {max(vals):.6f}")
        lines.append(f"  最后100轮均值: {sum(vals[-100:])/min(len(vals),100):.6f}")

    return "\n".join(lines)


def build_prompt(runs, diagnostics_list, comparison_context=""):
    """构建完整的 LLM prompt。"""
    parts = [SYSTEM_PROMPT, "\n---\n"]

    if comparison_context:
        parts.append(comparison_context)

    for run, diag in zip(runs, diagnostics_list):
        parts.append(format_run_for_llm(run))
        parts.append(format_config_for_llm(run.config))
        parts.append(format_diagnostics_for_llm(diag))
        parts.append("\n---\n")

    if len(runs) > 1:
        parts.append("请对比以上多次训练，分析差异原因，并给出综合调参建议。")

    return "\n".join(parts)


def query_llm(runs, diagnostics_list, config=None, comparison_context=""):
    """
    调用 LLM 进行训练诊断。

    Args:
        runs: RunData 列表
        diagnostics_list: 对应的诊断结果列表
        config: LLM 配置 dict（provider, model, api_key, base_url）
        comparison_context: 额外的对比上下文

    Returns:
        LLM 生成的建议文本
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # 从环境变量 fallback
    api_key = cfg["api_key"] or os.environ.get("LLM_API_KEY", "")
    base_url = cfg["base_url"] or os.environ.get("LLM_BASE_URL", "")

    if not api_key:
        return "错误：请先在设置页面配置 API Key。"

    messages = [
        {
            "role": "user",
            "content": build_prompt(runs, diagnostics_list, comparison_context),
        }
    ]

    kwargs = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": 2048,
    }

    # Ollama / 自定义 base_url
    if base_url:
        kwargs["api_base"] = base_url

    # 根据 provider 设置 api_key 的环境变量
    provider = cfg["provider"]
    if provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif provider in ("openai", "deepseek"):
        os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            kwargs["api_base"] = base_url

    try:
        response = completion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 调用失败: {e}\n\n请检查 API Key 和网络连接。"

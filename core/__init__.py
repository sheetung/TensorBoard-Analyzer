"""核心模块：数据加载、LLM 调用、AI 对话。"""

from .analyzer import scan_log_dirs, load_run, compute_diagnostics
from .llm_advisor import (
    DEFAULT_CONFIG,
    PROVIDER_PRESETS,
    SYSTEM_PROMPT,
    build_prompt,
    query_llm,
    _call_openai_compatible,
)
from .ai_chat import init_chat_messages, chat_llm

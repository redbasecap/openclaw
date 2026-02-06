"""Routing matrix that maps (TaskType, ModelTier) → model name."""

from __future__ import annotations

from model_router.enums import ModelTier, TaskType

# Each entry maps a TaskType to a dict of {ModelTier: model_name}.
MODEL_CONFIG: dict[TaskType, dict[ModelTier, str]] = {
    TaskType.BRAIN: {
        ModelTier.BEST: "claude-opus-4-5-latest",
        ModelTier.COST_SAVING: "kimi-k2-5",
    },
    TaskType.HEARTBEAT: {
        ModelTier.BEST: "claude-haiku-latest",
        ModelTier.COST_SAVING: "claude-haiku-latest",
    },
    TaskType.CODING: {
        ModelTier.BEST: "codex-gpt-5-2",
        ModelTier.COST_SAVING: "minimax-2-1",
    },
    TaskType.WEB_SEARCH: {
        ModelTier.BEST: "claude-opus-4-5-latest",
        ModelTier.COST_SAVING: "deepseek-v3",
    },
    TaskType.CONTENT: {
        ModelTier.BEST: "claude-opus-4-5-latest",
        ModelTier.COST_SAVING: "kimi-k2-5",
    },
    TaskType.VOICE: {
        ModelTier.BEST: "gpt-4o-realtime",
        ModelTier.COST_SAVING: "gpt-4o-realtime",
    },
    TaskType.IMAGE_UNDERSTANDING: {
        ModelTier.BEST: "claude-opus-4-5-latest",
        ModelTier.COST_SAVING: "gemini-2-5-flash",
    },
}

# Maps model-name prefixes to provider keys used by _get_provider.
PROVIDER_PREFIX_MAP: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "codex": "openai",
    "o1": "openai",
    "deepseek": "deepseek",
    "kimi": "moonshot",
    "gemini": "google",
    "minimax": "minimax",
}

# Environment variable names for each provider's API key.
PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "google": "GOOGLE_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

"""Concrete provider implementations."""

from model_router.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    async def chat(self, message: str, model: str, **kwargs: object) -> str:
        # In production, call the Anthropic API via httpx / anthropic SDK.
        raise NotImplementedError("AnthropicProvider.chat requires a real API client")


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI GPT / Codex / Realtime models."""

    async def chat(self, message: str, model: str, **kwargs: object) -> str:
        raise NotImplementedError("OpenAIProvider.chat requires a real API client")


class DeepSeekProvider(BaseProvider):
    """Provider for DeepSeek models."""

    async def chat(self, message: str, model: str, **kwargs: object) -> str:
        raise NotImplementedError("DeepSeekProvider.chat requires a real API client")


class MoonshotProvider(BaseProvider):
    """Provider for Moonshot / Kimi models."""

    async def chat(self, message: str, model: str, **kwargs: object) -> str:
        raise NotImplementedError("MoonshotProvider.chat requires a real API client")


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models."""

    async def chat(self, message: str, model: str, **kwargs: object) -> str:
        raise NotImplementedError("GoogleProvider.chat requires a real API client")


class MiniMaxProvider(BaseProvider):
    """Provider for MiniMax models."""

    async def chat(self, message: str, model: str, **kwargs: object) -> str:
        raise NotImplementedError("MiniMaxProvider.chat requires a real API client")


# Registry used by ModelRouter._get_provider to resolve provider keys.
PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "moonshot": MoonshotProvider,
    "google": GoogleProvider,
    "minimax": MiniMaxProvider,
}

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "DeepSeekProvider",
    "GoogleProvider",
    "MiniMaxProvider",
    "MoonshotProvider",
    "OpenAIProvider",
    "PROVIDER_CLASSES",
]

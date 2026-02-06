"""Core ModelRouter – classifies tasks, picks a tier, and dispatches to providers."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from model_router.config import MODEL_CONFIG, PROVIDER_ENV_KEYS, PROVIDER_PREFIX_MAP
from model_router.enums import ModelTier, TaskType
from model_router.providers import PROVIDER_CLASSES
from model_router.providers.base import BaseProvider
from model_router.stats import RouterStats

logger = logging.getLogger(__name__)

# Keyword patterns used by _classify_task (checked in order, first match wins).
_TASK_KEYWORDS: list[tuple[TaskType, re.Pattern[str]]] = [
    (TaskType.CODING, re.compile(r"\b(code|programm|bug|python|function|class|def |import )\b", re.I)),
    (TaskType.WEB_SEARCH, re.compile(r"\b(suche|search|web|google|browse|find online)\b", re.I)),
    (TaskType.IMAGE_UNDERSTANDING, re.compile(r"\b(bild|image|foto|screenshot|picture|photo)\b", re.I)),
    (TaskType.CONTENT, re.compile(r"\b(schreib|content|blog|article|essay|write)\b", re.I)),
    (TaskType.HEARTBEAT, re.compile(r"\b(status|ping|health|alive|uptime)\b", re.I)),
]

# Words that signal a hard/complex request → prefer BEST tier.
_COMPLEXITY_PATTERN = re.compile(r"\b(complex|difficult|hard|advanced|challenging)\b", re.I)

# Threshold (in characters) above which we prefer the BEST tier.
_LENGTH_THRESHOLD = 2000


class ModelRouter:
    """Selects the cheapest appropriate AI model based on task type.

    Parameters
    ----------
    default_tier:
        Tier to use when no specific rule applies.
    api_keys:
        Mapping of provider key (e.g. ``"anthropic"``) to API key string.
        Missing keys are read from environment variables.
    """

    def __init__(
        self,
        default_tier: ModelTier = ModelTier.COST_SAVING,
        api_keys: Optional[dict[str, str]] = None,
    ) -> None:
        self.default_tier = default_tier
        self._api_keys: dict[str, str] = dict(api_keys or {})
        self._providers: dict[str, BaseProvider] = {}
        self.stats = RouterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        message: str,
        task_hint: Optional[TaskType] = None,
        force_tier: Optional[ModelTier] = None,
    ) -> str:
        """Classify *message*, select a model, call the provider, and return the response.

        If the cost-saving model fails, the router automatically retries with
        the best-tier model as a fallback.
        """
        task = task_hint or self._classify_task(message)
        tier = force_tier or self._select_tier(task, message)
        model = MODEL_CONFIG[task][tier]

        logger.info("Routing task=%s tier=%s model=%s", task.value, tier.value, model)

        try:
            return await self._call_model(model, message, task)
        except Exception:
            # Fallback: retry with the BEST tier model when cost-saving fails.
            if tier is not ModelTier.BEST:
                best_model = MODEL_CONFIG[task][ModelTier.BEST]
                logger.warning(
                    "Cost-saving model %s failed; falling back to %s",
                    model,
                    best_model,
                )
                self.stats.record_error(model)
                return await self._call_model(best_model, message, task)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_task(self, message: str) -> TaskType:
        """Determine the task type from message keywords."""
        for task_type, pattern in _TASK_KEYWORDS:
            if pattern.search(message):
                return task_type
        return TaskType.BRAIN

    def _select_tier(self, task: TaskType, message: str) -> ModelTier:
        """Choose BEST or COST_SAVING based on the task and message."""
        if task is TaskType.VOICE:
            return ModelTier.BEST
        if task is TaskType.HEARTBEAT:
            return ModelTier.COST_SAVING
        if len(message) > _LENGTH_THRESHOLD or _COMPLEXITY_PATTERN.search(message):
            return ModelTier.BEST
        return self.default_tier

    def _get_provider(self, model_name: str) -> BaseProvider:
        """Resolve *model_name* to a cached provider instance."""
        provider_key = self._resolve_provider_key(model_name)

        if provider_key not in self._providers:
            api_key = self._api_keys.get(provider_key) or os.environ.get(
                PROVIDER_ENV_KEYS.get(provider_key, ""),
                "",
            )
            cls = PROVIDER_CLASSES.get(provider_key)
            if cls is None:
                raise ValueError(f"No provider registered for key '{provider_key}'")
            self._providers[provider_key] = cls(api_key=api_key)

        return self._providers[provider_key]

    async def _call_model(self, model: str, message: str, task: TaskType) -> str:
        """Call the provider and record stats."""
        provider = self._get_provider(model)
        input_tokens = RouterStats.estimate_tokens(message)

        with RouterStats.timer() as t:
            result = await provider.chat(message, model)

        output_tokens = RouterStats.estimate_tokens(result)
        self.stats.record_call(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=t.elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider_key(model_name: str) -> str:
        """Map a model name to its provider key via prefix matching."""
        for prefix, key in PROVIDER_PREFIX_MAP.items():
            if model_name.startswith(prefix):
                return key
        raise ValueError(f"Cannot resolve provider for model '{model_name}'")

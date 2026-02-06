"""Tests for ModelRouter – classification, tier selection, and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from model_router.enums import ModelTier, TaskType
from model_router.router import ModelRouter


@pytest.fixture()
def router() -> ModelRouter:
    return ModelRouter(default_tier=ModelTier.COST_SAVING, api_keys={"anthropic": "test-key"})


# ── _classify_task ──────────────────────────────────────────────


class TestClassifyTask:
    def test_coding_keywords(self, router: ModelRouter) -> None:
        assert router._classify_task("Fix the Python bug") == TaskType.CODING
        assert router._classify_task("write code for me") == TaskType.CODING

    def test_web_search_keywords(self, router: ModelRouter) -> None:
        assert router._classify_task("search the web for AI") == TaskType.WEB_SEARCH
        assert router._classify_task("google quantum computing") == TaskType.WEB_SEARCH

    def test_image_keywords(self, router: ModelRouter) -> None:
        assert router._classify_task("describe this image") == TaskType.IMAGE_UNDERSTANDING
        assert router._classify_task("look at this screenshot") == TaskType.IMAGE_UNDERSTANDING

    def test_content_keywords(self, router: ModelRouter) -> None:
        assert router._classify_task("write a blog post") == TaskType.CONTENT
        assert router._classify_task("create content for me") == TaskType.CONTENT

    def test_heartbeat_keywords(self, router: ModelRouter) -> None:
        assert router._classify_task("ping") == TaskType.HEARTBEAT
        assert router._classify_task("health check status") == TaskType.HEARTBEAT

    def test_default_is_brain(self, router: ModelRouter) -> None:
        assert router._classify_task("explain quantum physics") == TaskType.BRAIN
        assert router._classify_task("hello there") == TaskType.BRAIN


# ── _select_tier ────────────────────────────────────────────────


class TestSelectTier:
    def test_voice_always_best(self, router: ModelRouter) -> None:
        assert router._select_tier(TaskType.VOICE, "anything") == ModelTier.BEST

    def test_heartbeat_always_cost_saving(self, router: ModelRouter) -> None:
        assert router._select_tier(TaskType.HEARTBEAT, "ping") == ModelTier.COST_SAVING

    def test_long_message_upgrades_to_best(self, router: ModelRouter) -> None:
        long_msg = "a" * 2001
        assert router._select_tier(TaskType.BRAIN, long_msg) == ModelTier.BEST

    def test_complex_keyword_upgrades_to_best(self, router: ModelRouter) -> None:
        assert router._select_tier(TaskType.BRAIN, "this is complex") == ModelTier.BEST
        assert router._select_tier(TaskType.BRAIN, "very difficult task") == ModelTier.BEST

    def test_default_tier_used_otherwise(self, router: ModelRouter) -> None:
        assert router._select_tier(TaskType.BRAIN, "hello") == ModelTier.COST_SAVING

    def test_best_default_tier(self) -> None:
        r = ModelRouter(default_tier=ModelTier.BEST)
        assert r._select_tier(TaskType.BRAIN, "hello") == ModelTier.BEST


# ── _resolve_provider_key ──────────────────────────────────────


class TestResolveProviderKey:
    def test_claude_maps_to_anthropic(self) -> None:
        assert ModelRouter._resolve_provider_key("claude-opus-4-5-latest") == "anthropic"

    def test_gpt_maps_to_openai(self) -> None:
        assert ModelRouter._resolve_provider_key("gpt-4o-realtime") == "openai"

    def test_codex_maps_to_openai(self) -> None:
        assert ModelRouter._resolve_provider_key("codex-gpt-5-2") == "openai"

    def test_deepseek_maps_to_deepseek(self) -> None:
        assert ModelRouter._resolve_provider_key("deepseek-v3") == "deepseek"

    def test_kimi_maps_to_moonshot(self) -> None:
        assert ModelRouter._resolve_provider_key("kimi-k2-5") == "moonshot"

    def test_gemini_maps_to_google(self) -> None:
        assert ModelRouter._resolve_provider_key("gemini-2-5-flash") == "google"

    def test_minimax_maps_to_minimax(self) -> None:
        assert ModelRouter._resolve_provider_key("minimax-2-1") == "minimax"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot resolve provider"):
            ModelRouter._resolve_provider_key("unknown-model")


# ── route (integration) ────────────────────────────────────────


class TestRoute:
    @pytest.mark.asyncio
    async def test_route_calls_provider(self, router: ModelRouter) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "Hello from mock"

        with patch.object(router, "_get_provider", return_value=mock_provider):
            result = await router.route("ping", task_hint=TaskType.HEARTBEAT)

        assert result == "Hello from mock"
        mock_provider.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_fallback_on_failure(self, router: ModelRouter) -> None:
        """When cost-saving model fails, the router should retry with BEST."""
        mock_provider = AsyncMock()
        mock_provider.chat.side_effect = [RuntimeError("fail"), "Fallback OK"]

        with patch.object(router, "_get_provider", return_value=mock_provider):
            result = await router.route("hello", task_hint=TaskType.BRAIN)

        assert result == "Fallback OK"
        assert mock_provider.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_route_records_stats(self, router: ModelRouter) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "ok"

        with patch.object(router, "_get_provider", return_value=mock_provider):
            await router.route("status check", task_hint=TaskType.HEARTBEAT)

        stats = router.stats.summary()
        assert "claude-haiku-latest" in stats

    @pytest.mark.asyncio
    async def test_force_tier(self, router: ModelRouter) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "best result"

        with patch.object(router, "_get_provider", return_value=mock_provider):
            result = await router.route(
                "hello",
                task_hint=TaskType.BRAIN,
                force_tier=ModelTier.BEST,
            )

        assert result == "best result"
        # BEST tier for BRAIN → claude-opus-4-5-latest
        mock_provider.chat.assert_awaited_once_with("hello", "claude-opus-4-5-latest")

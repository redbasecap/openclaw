"""Tests for the routing configuration matrix."""

from model_router.config import MODEL_CONFIG, PROVIDER_PREFIX_MAP
from model_router.enums import ModelTier, TaskType


def test_all_task_types_have_config() -> None:
    for task in TaskType:
        assert task in MODEL_CONFIG, f"Missing config for {task}"


def test_all_tiers_present_per_task() -> None:
    for task, tiers in MODEL_CONFIG.items():
        assert ModelTier.BEST in tiers, f"Missing BEST for {task}"
        assert ModelTier.COST_SAVING in tiers, f"Missing COST_SAVING for {task}"


def test_voice_same_model_both_tiers() -> None:
    voice = MODEL_CONFIG[TaskType.VOICE]
    assert voice[ModelTier.BEST] == voice[ModelTier.COST_SAVING]


def test_heartbeat_same_model_both_tiers() -> None:
    hb = MODEL_CONFIG[TaskType.HEARTBEAT]
    assert hb[ModelTier.BEST] == hb[ModelTier.COST_SAVING]


def test_provider_prefix_map_covers_all_models() -> None:
    """Every model in the config must resolve to a provider via prefix map."""
    for task, tiers in MODEL_CONFIG.items():
        for tier, model in tiers.items():
            matched = any(model.startswith(pfx) for pfx in PROVIDER_PREFIX_MAP)
            assert matched, f"Model '{model}' ({task}/{tier}) has no provider prefix"

"""Tests for enums module."""

from model_router.enums import ModelTier, TaskType


def test_task_type_values() -> None:
    assert TaskType.BRAIN.value == "brain"
    assert TaskType.HEARTBEAT.value == "heartbeat"
    assert TaskType.CODING.value == "coding"
    assert TaskType.WEB_SEARCH.value == "web_search"
    assert TaskType.CONTENT.value == "content"
    assert TaskType.VOICE.value == "voice"
    assert TaskType.IMAGE_UNDERSTANDING.value == "image_understanding"


def test_model_tier_values() -> None:
    assert ModelTier.BEST.value == "best"
    assert ModelTier.COST_SAVING.value == "cost_saving"


def test_task_type_is_str_enum() -> None:
    assert isinstance(TaskType.BRAIN, str)


def test_model_tier_is_str_enum() -> None:
    assert isinstance(ModelTier.BEST, str)

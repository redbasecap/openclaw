"""Enumerations for task types and model tiers."""

from enum import Enum


class TaskType(str, Enum):
    """Categories of tasks the router can classify messages into."""

    BRAIN = "brain"
    HEARTBEAT = "heartbeat"
    CODING = "coding"
    WEB_SEARCH = "web_search"
    CONTENT = "content"
    VOICE = "voice"
    IMAGE_UNDERSTANDING = "image_understanding"


class ModelTier(str, Enum):
    """Quality/cost tiers for model selection."""

    BEST = "best"
    COST_SAVING = "cost_saving"

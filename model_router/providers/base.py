"""Abstract base class for all AI model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Every concrete provider must implement the async ``chat`` method."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @abstractmethod
    async def chat(self, message: str, model: str, **kwargs: object) -> str:
        """Send *message* to the given *model* and return the response text."""

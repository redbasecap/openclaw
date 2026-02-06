#!/usr/bin/env python3
"""Usage demonstration for the model router.

Run with:
    python -m model_router.example
"""

from __future__ import annotations

import asyncio

from model_router.enums import ModelTier, TaskType
from model_router.router import ModelRouter


async def main() -> None:
    # Initialise the router (cost-saving by default).
    router = ModelRouter(
        default_tier=ModelTier.COST_SAVING,
        api_keys={
            "anthropic": "sk-ant-...",
            "openai": "sk-...",
        },
    )

    # --- Show classification examples ---
    examples = [
        "Can you fix this Python bug in my function?",
        "Search the web for the latest AI news",
        "Describe what's in this image",
        "Write a blog post about async programming",
        "ping – are you alive?",
        "Explain quantum entanglement to me",
    ]

    for msg in examples:
        task = router._classify_task(msg)
        tier = router._select_tier(task, msg)
        print(f"Message : {msg!r}")
        print(f"  Task  : {task.value}")
        print(f"  Tier  : {tier.value}")
        print()

    # --- Force a specific task + tier ---
    print("Forced routing (CODING / BEST):")
    try:
        result = await router.route(
            "Refactor this module",
            task_hint=TaskType.CODING,
            force_tier=ModelTier.BEST,
        )
        print(f"  Result: {result}")
    except NotImplementedError:
        print("  (provider not wired – expected in demo mode)")

    # --- Print stats ---
    print("\nRouter stats:")
    print(router.stats.summary())


if __name__ == "__main__":
    asyncio.run(main())

"""Tests for RouterStats."""

from model_router.stats import RouterStats


def test_record_and_summary() -> None:
    stats = RouterStats()
    stats.record_call("model-a", input_tokens=10, output_tokens=20, latency_s=0.5)
    stats.record_call("model-a", input_tokens=5, output_tokens=15, latency_s=0.3)
    stats.record_error("model-b")

    summary = stats.summary()

    assert summary["model-a"]["calls"] == 2
    assert summary["model-a"]["total_input_tokens"] == 15
    assert summary["model-a"]["total_output_tokens"] == 35
    assert summary["model-a"]["total_latency_s"] == 0.8

    assert summary["model-b"]["errors"] == 1
    assert summary["model-b"]["calls"] == 0


def test_estimate_tokens() -> None:
    assert RouterStats.estimate_tokens("hello world") >= 1
    assert RouterStats.estimate_tokens("") == 1  # minimum 1


def test_timer_context_manager() -> None:
    with RouterStats.timer() as t:
        _ = sum(range(100))
    assert t.elapsed >= 0.0

from pearl_gemm.sm120_telemetry import (
    TWO256,
    Sm120SearchTelemetry,
    format_sm120_rate,
    format_sm120_snapshot,
)


def _result(
    *,
    candidates: int,
    m: int,
    n: int,
    k: int,
    winners: int = 0,
    tail_d8: int = 0,
    tail_d12: int = 0,
    tail_d16: int = 0,
    tail_d20: int = 0,
    tail_d24: int = 0,
) -> dict[str, int | bool]:
    return {
        "candidates": candidates,
        "target_tests": candidates * (m // 16) * (n // 16),
        "valid_candidate_work": candidates * m * n * k,
        "winners": winners,
        "winner_overflow": False,
        "cuda_errors": 0,
        "b_cache_hits": 0,
        "b_cache_misses": 0,
        "tail_d8": tail_d8,
        "tail_d12": tail_d12,
        "tail_d16": tail_d16,
        "tail_d20": tail_d20,
        "tail_d24": tail_d24,
    }


def test_target_tests_and_krig_equivalent_hashrate() -> None:
    telemetry = Sm120SearchTelemetry(log_every=64)
    kwargs = dict(generation=7, m=32, n=32, k=4096, target=TWO256 - 1)

    first = telemetry.record(
        **kwargs,
        result=_result(candidates=2, m=32, n=32, k=4096),
        now_ns=0,
    )
    assert first.target_tests == 8
    assert first.total_valid_candidate_work == 2 * 32 * 32 * 4096
    assert first.generation_changed

    second = telemetry.record(
        **kwargs,
        result=_result(candidates=2, m=32, n=32, k=4096),
        now_ns=1_000_000_000,
    )
    expected_total_work = 4 * 32 * 32 * 4096
    assert second.total_target_tests == 16
    assert second.total_hashrate_hs == expected_total_work
    assert second.job_hashrate_hs == expected_total_work
    assert second.job_mean_target_hit_seconds == 1.0 / 16.0

    rate = format_sm120_rate(second)
    assert "hashrate_ths=" in rate
    assert "tiles_s=16.000" in rate


def test_generation_resets_job_window_not_total() -> None:
    telemetry = Sm120SearchTelemetry(log_every=64)
    result = _result(candidates=1, m=16, n=16, k=4096, tail_d8=1)
    telemetry.record(
        generation=1,
        m=16,
        n=16,
        k=4096,
        target=TWO256 - 1,
        result=result,
        now_ns=0,
    )
    changed = telemetry.record(
        generation=2,
        m=16,
        n=16,
        k=4096,
        target=TWO256 - 1,
        result={**result, "winners": 1},
        now_ns=1_000_000_000,
    )
    assert changed.generation_changed
    assert changed.job_target_tests == 1
    assert changed.total_target_tests == 2
    assert changed.winners == 1
    assert changed.tail_counts_total[8] == 2
    verbose = format_sm120_snapshot(changed)
    assert "generation=2" in verbose
    assert "total_tests=2" in verbose
    assert "winners=1" in verbose

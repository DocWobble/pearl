from pearl_gemm.sm120_telemetry import TWO256, Sm120SearchTelemetry, format_sm120_snapshot


def _result(candidates: int = 1, winners: int = 0) -> dict[str, int | bool]:
    return {
        "candidates": candidates,
        "valid_candidate_work": 123,
        "winners": winners,
        "winner_overflow": False,
        "cuda_errors": 0,
        "b_cache_hits": 1,
        "b_cache_misses": 0,
    }


def test_target_tests_count_real_16x16_tiles() -> None:
    telemetry = Sm120SearchTelemetry(log_every=64)
    first = telemetry.record(
        generation=7,
        m=32,
        n=32,
        k=4096,
        target=TWO256 - 1,
        result=_result(candidates=2),
        now_ns=0,
    )
    assert first.target_tests == 8
    assert first.job_target_tests == 8
    assert first.total_target_tests == 8
    assert first.job_expected_hits == 8.0
    assert first.generation_changed
    assert first.should_log

    second = telemetry.record(
        generation=7,
        m=32,
        n=32,
        k=4096,
        target=TWO256 - 1,
        result=_result(candidates=2),
        now_ns=1_000_000_000,
    )
    assert second.total_target_tests == 16
    assert second.total_target_tests_per_second == 16.0
    assert second.job_target_tests_per_second == 16.0
    assert second.job_mean_target_hit_seconds == 1.0 / 16.0
    assert not second.generation_changed
    assert not second.should_log


def test_generation_resets_job_window_but_not_total() -> None:
    telemetry = Sm120SearchTelemetry(log_every=64)
    telemetry.record(
        generation=1,
        m=16,
        n=16,
        k=4096,
        target=TWO256 - 1,
        result=_result(),
        now_ns=0,
    )
    changed = telemetry.record(
        generation=2,
        m=16,
        n=16,
        k=4096,
        target=TWO256 - 1,
        result=_result(winners=1),
        now_ns=1_000_000_000,
    )
    assert changed.generation_changed
    assert changed.job_target_tests == 1
    assert changed.total_target_tests == 2
    assert changed.winners == 1
    line = format_sm120_snapshot(changed)
    assert "generation=2" in line
    assert "total_tests=2" in line
    assert "winners=1" in line

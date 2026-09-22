"""Runtime accounting for the SM120 alpha search path.

The unit counted here is a completed Pearl jackpot target comparison.  One
candidate matrix can contain many independent 16x16 proof tiles, so the
candidate count alone is not a hashrate measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

TWO256 = 1 << 256


@dataclass(frozen=True)
class Sm120TelemetrySnapshot:
    generation: int
    call: int
    m: int
    n: int
    k: int
    candidates: int
    target_tests: int
    job_target_tests: int
    total_target_tests: int
    job_target_tests_per_second: float
    total_target_tests_per_second: float
    job_expected_hits: float
    total_expected_hits: float
    job_mean_target_hit_seconds: float | None
    total_mean_target_hit_seconds: float | None
    valid_candidate_work: int
    total_valid_candidate_work: int
    winners: int
    b_cache_hits: int
    b_cache_misses: int
    generation_changed: bool
    should_log: bool


class Sm120SearchTelemetry:
    def __init__(self, log_every: int = 64) -> None:
        if log_every < 1:
            raise ValueError("log_every must be >= 1")
        self.log_every = log_every
        self._start_ns: int | None = None
        self._job_start_ns: int | None = None
        self._generation: int | None = None
        self._calls = 0
        self._total_target_tests = 0
        self._job_target_tests = 0
        self._total_expected_hits = 0.0
        self._job_expected_hits = 0.0
        self._total_valid_candidate_work = 0

    @staticmethod
    def _rate(count: int, start_ns: int | None, now_ns: int) -> float:
        if start_ns is None or now_ns <= start_ns:
            return 0.0
        return count / ((now_ns - start_ns) / 1e9)

    @staticmethod
    def _mean_hit_seconds(expected_hits: float, start_ns: int | None, now_ns: int) -> float | None:
        if expected_hits <= 0.0 or start_ns is None or now_ns <= start_ns:
            return None
        return ((now_ns - start_ns) / 1e9) / expected_hits

    def record(
        self,
        *,
        generation: int,
        m: int,
        n: int,
        k: int,
        target: int,
        result: Mapping[str, Any],
        now_ns: int | None = None,
    ) -> Sm120TelemetrySnapshot:
        if m <= 0 or n <= 0 or k <= 0 or m % 16 or n % 16:
            raise ValueError("SM120 telemetry requires complete 16x16 proof-tile geometry")
        if target < 0 or target >= TWO256:
            raise ValueError("target must be a uint256 value")

        now = time.monotonic_ns() if now_ns is None else now_ns
        if self._start_ns is None:
            self._start_ns = now

        generation_changed = generation != self._generation
        if generation_changed:
            self._generation = generation
            self._job_start_ns = now
            self._job_target_tests = 0
            self._job_expected_hits = 0.0

        candidates = int(result.get("candidates", 0))
        if candidates < 0:
            raise ValueError("candidate count cannot be negative")
        target_tests = candidates * (m // 16) * (n // 16)
        expected_hits = target_tests * ((target + 1) / TWO256)
        valid_candidate_work = int(result.get("valid_candidate_work", 0))
        winners = int(result.get("winners", 0))
        b_cache_hits = int(result.get("b_cache_hits", 0))
        b_cache_misses = int(result.get("b_cache_misses", 0))

        self._calls += 1
        self._total_target_tests += target_tests
        self._job_target_tests += target_tests
        self._total_expected_hits += expected_hits
        self._job_expected_hits += expected_hits
        self._total_valid_candidate_work += valid_candidate_work

        should_log = (
            generation_changed
            or self._calls % self.log_every == 0
            or winners > 0
            or bool(result.get("winner_overflow", False))
            or int(result.get("cuda_errors", 0)) > 0
        )

        return Sm120TelemetrySnapshot(
            generation=generation,
            call=self._calls,
            m=m,
            n=n,
            k=k,
            candidates=candidates,
            target_tests=target_tests,
            job_target_tests=self._job_target_tests,
            total_target_tests=self._total_target_tests,
            job_target_tests_per_second=self._rate(self._job_target_tests, self._job_start_ns, now),
            total_target_tests_per_second=self._rate(self._total_target_tests, self._start_ns, now),
            job_expected_hits=self._job_expected_hits,
            total_expected_hits=self._total_expected_hits,
            job_mean_target_hit_seconds=self._mean_hit_seconds(self._job_expected_hits, self._job_start_ns, now),
            total_mean_target_hit_seconds=self._mean_hit_seconds(self._total_expected_hits, self._start_ns, now),
            valid_candidate_work=valid_candidate_work,
            total_valid_candidate_work=self._total_valid_candidate_work,
            winners=winners,
            b_cache_hits=b_cache_hits,
            b_cache_misses=b_cache_misses,
            generation_changed=generation_changed,
            should_log=should_log,
        )


def _fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def format_sm120_snapshot(snapshot: Sm120TelemetrySnapshot) -> str:
    """Format one grep-friendly alpha telemetry line."""
    return (
        "SM120_ALPHA "
        f"generation={snapshot.generation} call={snapshot.call} "
        f"m={snapshot.m} n={snapshot.n} k={snapshot.k} "
        f"candidates={snapshot.candidates} tests={snapshot.target_tests} "
        f"job_tests={snapshot.job_target_tests} total_tests={snapshot.total_target_tests} "
        f"job_tests_s={snapshot.job_target_tests_per_second:.3f} "
        f"total_tests_s={snapshot.total_target_tests_per_second:.3f} "
        f"job_expected_hits={snapshot.job_expected_hits:.9e} "
        f"job_mean_hit_s={_fmt_seconds(snapshot.job_mean_target_hit_seconds)} "
        f"total_expected_hits={snapshot.total_expected_hits:.9e} "
        f"total_mean_hit_s={_fmt_seconds(snapshot.total_mean_target_hit_seconds)} "
        f"valid_work={snapshot.valid_candidate_work} "
        f"total_valid_work={snapshot.total_valid_candidate_work} "
        f"b_cache_hit={snapshot.b_cache_hits} b_cache_miss={snapshot.b_cache_misses} "
        f"winners={snapshot.winners}"
    )

"""Runtime accounting for the SM120 throughput/search branch.

Two deliberately different units are reported:

* target tests/s: completed 16x16 jackpot BLAKE3 comparisons, useful for
  probability modelling.
* Krig-equivalent H/s: candidates*m*n*k per second.  The captured Krig
  Prometheus baseline exposes its public hashrate as tiles/s multiplied by the
  16*16*K work represented by each tile, so this is the apples-to-apples miner
  hashrate unit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Mapping

TWO256 = 1 << 256
TAIL_BITS = (8, 12, 16, 20, 24)


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
    job_expected_opens_per_second: float
    total_expected_opens_per_second: float
    job_mean_target_hit_seconds: float | None
    total_mean_target_hit_seconds: float | None
    valid_candidate_work: int
    job_valid_candidate_work: int
    total_valid_candidate_work: int
    job_hashrate_hs: float
    total_hashrate_hs: float
    rolling_hashrate_hs: float
    tail_counts_total: dict[int, int]
    tail_ratios_total: dict[int, float]
    winners: int
    b_cache_hits: int
    b_cache_misses: int
    generation_changed: bool
    should_log: bool

    @property
    def job_hashrate_ths(self) -> float:
        return self.job_hashrate_hs / 1e12

    @property
    def total_hashrate_ths(self) -> float:
        return self.total_hashrate_hs / 1e12

    @property
    def rolling_hashrate_ths(self) -> float:
        return self.rolling_hashrate_hs / 1e12


class Sm120SearchTelemetry:
    def __init__(self, log_every: int = 64, rolling_seconds: float = 30.0) -> None:
        if log_every < 1:
            raise ValueError("log_every must be >= 1")
        if rolling_seconds <= 0:
            raise ValueError("rolling_seconds must be > 0")
        self.log_every = log_every
        self.rolling_seconds = rolling_seconds
        self._start_ns: int | None = None
        self._job_start_ns: int | None = None
        self._generation: int | None = None
        self._calls = 0
        self._total_target_tests = 0
        self._job_target_tests = 0
        self._total_expected_hits = 0.0
        self._job_expected_hits = 0.0
        self._total_valid_candidate_work = 0
        self._job_valid_candidate_work = 0
        self._tail_counts_total = {bits: 0 for bits in TAIL_BITS}
        self._rolling: deque[tuple[int, int]] = deque()

    @staticmethod
    def _rate(count: int, start_ns: int | None, now_ns: int) -> float:
        if start_ns is None or now_ns <= start_ns:
            return 0.0
        return count / ((now_ns - start_ns) / 1e9)

    @staticmethod
    def _mean_hit_seconds(
        expected_hits: float, start_ns: int | None, now_ns: int
    ) -> float | None:
        if expected_hits <= 0.0 or start_ns is None or now_ns <= start_ns:
            return None
        return ((now_ns - start_ns) / 1e9) / expected_hits

    def _rolling_rate(self, now_ns: int) -> float:
        cutoff = now_ns - int(self.rolling_seconds * 1e9)
        while len(self._rolling) > 1 and self._rolling[1][0] <= cutoff:
            self._rolling.popleft()
        if len(self._rolling) < 2:
            return 0.0
        start_ns, start_work = self._rolling[0]
        end_ns, end_work = self._rolling[-1]
        if end_ns <= start_ns:
            return 0.0
        return (end_work - start_work) / ((end_ns - start_ns) / 1e9)

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
            self._job_valid_candidate_work = 0

        candidates = int(result.get("candidates", 0))
        if candidates < 0:
            raise ValueError("candidate count cannot be negative")

        computed_tests = candidates * (m // 16) * (n // 16)
        target_tests = int(result.get("target_tests", computed_tests))
        if target_tests != computed_tests:
            raise ValueError(
                f"backend target-test mismatch: backend={target_tests} computed={computed_tests}"
            )

        expected_hits = target_tests * ((target + 1) / TWO256)
        valid_candidate_work = int(result.get("valid_candidate_work", 0))
        expected_work = candidates * m * n * k
        if valid_candidate_work != expected_work:
            raise ValueError(
                f"backend work mismatch: backend={valid_candidate_work} computed={expected_work}"
            )

        winners = int(result.get("winners", 0))
        b_cache_hits = int(result.get("b_cache_hits", 0))
        b_cache_misses = int(result.get("b_cache_misses", 0))

        self._calls += 1
        self._total_target_tests += target_tests
        self._job_target_tests += target_tests
        self._total_expected_hits += expected_hits
        self._job_expected_hits += expected_hits
        self._total_valid_candidate_work += valid_candidate_work
        self._job_valid_candidate_work += valid_candidate_work

        for bits in TAIL_BITS:
            self._tail_counts_total[bits] += int(result.get(f"tail_d{bits}", 0))

        self._rolling.append((now, self._total_valid_candidate_work))
        rolling_hashrate = self._rolling_rate(now)

        tail_ratios: dict[int, float] = {}
        for bits in TAIL_BITS:
            expected = self._total_target_tests / (1 << bits)
            tail_ratios[bits] = (
                self._tail_counts_total[bits] / expected if expected > 0 else 0.0
            )

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
            job_target_tests_per_second=self._rate(
                self._job_target_tests, self._job_start_ns, now
            ),
            total_target_tests_per_second=self._rate(
                self._total_target_tests, self._start_ns, now
            ),
            job_expected_hits=self._job_expected_hits,
            total_expected_hits=self._total_expected_hits,
            job_expected_opens_per_second=self._rate(
                self._job_expected_hits, self._job_start_ns, now
            ),
            total_expected_opens_per_second=self._rate(
                self._total_expected_hits, self._start_ns, now
            ),
            job_mean_target_hit_seconds=self._mean_hit_seconds(
                self._job_expected_hits, self._job_start_ns, now
            ),
            total_mean_target_hit_seconds=self._mean_hit_seconds(
                self._total_expected_hits, self._start_ns, now
            ),
            valid_candidate_work=valid_candidate_work,
            job_valid_candidate_work=self._job_valid_candidate_work,
            total_valid_candidate_work=self._total_valid_candidate_work,
            job_hashrate_hs=self._rate(
                self._job_valid_candidate_work, self._job_start_ns, now
            ),
            total_hashrate_hs=self._rate(
                self._total_valid_candidate_work, self._start_ns, now
            ),
            rolling_hashrate_hs=rolling_hashrate,
            tail_counts_total=dict(self._tail_counts_total),
            tail_ratios_total=tail_ratios,
            winners=winners,
            b_cache_hits=b_cache_hits,
            b_cache_misses=b_cache_misses,
            generation_changed=generation_changed,
            should_log=should_log,
        )


def _fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def format_sm120_rate(snapshot: Sm120TelemetrySnapshot) -> str:
    """Human-first line using the same H/s dimensional accounting as Krig."""
    return (
        "SM120_RATE "
        f"hashrate_ths={snapshot.total_hashrate_ths:.6f} "
        f"rolling_ths={snapshot.rolling_hashrate_ths:.6f} "
        f"tiles_s={snapshot.total_target_tests_per_second:.0f} "
        f"expected_opens_s={snapshot.total_expected_opens_per_second:.3e} "
        f"tail16_ratio={snapshot.tail_ratios_total[16]:.3f} "
        f"winners={snapshot.winners}"
    )


def format_sm120_snapshot(snapshot: Sm120TelemetrySnapshot) -> str:
    """Verbose grep-friendly telemetry, enabled explicitly for debugging."""
    tail = " ".join(
        f"tail{bits}={snapshot.tail_counts_total[bits]} "
        f"tail{bits}_ratio={snapshot.tail_ratios_total[bits]:.6f}"
        for bits in TAIL_BITS
    )
    return (
        "SM120_ALPHA "
        f"generation={snapshot.generation} call={snapshot.call} "
        f"m={snapshot.m} n={snapshot.n} k={snapshot.k} "
        f"candidates={snapshot.candidates} tests={snapshot.target_tests} "
        f"job_tests={snapshot.job_target_tests} total_tests={snapshot.total_target_tests} "
        f"job_tests_s={snapshot.job_target_tests_per_second:.3f} "
        f"total_tests_s={snapshot.total_target_tests_per_second:.3f} "
        f"hashrate_hs={snapshot.total_hashrate_hs:.3f} "
        f"hashrate_ths={snapshot.total_hashrate_ths:.9f} "
        f"rolling_ths={snapshot.rolling_hashrate_ths:.9f} "
        f"job_ths={snapshot.job_hashrate_ths:.9f} "
        f"job_expected_hits={snapshot.job_expected_hits:.9e} "
        f"job_expected_opens_s={snapshot.job_expected_opens_per_second:.9e} "
        f"job_mean_hit_s={_fmt_seconds(snapshot.job_mean_target_hit_seconds)} "
        f"total_expected_hits={snapshot.total_expected_hits:.9e} "
        f"total_expected_opens_s={snapshot.total_expected_opens_per_second:.9e} "
        f"total_mean_hit_s={_fmt_seconds(snapshot.total_mean_target_hit_seconds)} "
        f"valid_work={snapshot.valid_candidate_work} "
        f"total_valid_work={snapshot.total_valid_candidate_work} "
        f"{tail} "
        f"b_cache_hit={snapshot.b_cache_hits} b_cache_miss={snapshot.b_cache_misses} "
        f"winners={snapshot.winners}"
    )

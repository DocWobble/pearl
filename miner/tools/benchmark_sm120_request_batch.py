#!/usr/bin/env python3
"""Benchmark vLLM request batch sizes using live SM120_RATE telemetry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


def discover_model(endpoint: str) -> str:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=30) as response:
        payload = json.loads(response.read())
    models = payload.get("data", [])
    if not models:
        raise RuntimeError("vLLM returned no models")
    return str(models[0]["id"])


def post_batch(endpoint: str, model: str, prompts: list[str], timeout: float) -> None:
    body = json.dumps({
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"completion returned HTTP {response.status}")
        response.read()


def prompt(stream_id: int, sequence: int, words: int) -> str:
    prefix = f"Pearl workload stream={stream_id:016x} sequence={sequence:016x}. "
    return prefix + ("pearl " * words)


def new_rates(path: Path, offset: int) -> tuple[int, list[float]]:
    if not path.exists():
        return offset, []
    with path.open("r", errors="replace") as stream:
        stream.seek(offset)
        text = stream.read()
        new_offset = stream.tell()
    values: list[float] = []
    for line in text.splitlines():
        if "SM120_RATE " not in line:
            continue
        fields = dict(
            token.split("=", 1)
            for token in line.split()
            if "=" in token
        )
        value = fields.get("rolling_ths")
        if value is not None and float(value) > 0:
            values.append(float(value))
    return new_offset, values


def run_window(
    endpoint: str,
    model: str,
    batch: int,
    seconds: float,
    stream_id: int,
    sequence: int,
    words: int,
    timeout: float,
) -> tuple[int, int]:
    deadline = time.monotonic() + seconds
    requests = 0
    while time.monotonic() < deadline:
        prompts = [
            prompt(stream_id, sequence + lane, words)
            for lane in range(batch)
        ]
        post_batch(endpoint, model, prompts, timeout)
        sequence += batch
        requests += 1
    return sequence, requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--prompt-words", type=int, default=1024)
    parser.add_argument("--miner-log", type=Path, default=Path(".sm120-miner.log"))
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    model = args.model or discover_model(args.endpoint)
    stream_id = int.from_bytes(os.urandom(8), "little")
    sequence = 0
    offset = args.miner_log.stat().st_size if args.miner_log.exists() else 0
    scores: list[tuple[float, int]] = []

    for batch in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        started = time.monotonic()
        sequence, requests = run_window(
            args.endpoint,
            model,
            batch,
            args.seconds,
            stream_id,
            sequence,
            args.prompt_words,
            args.timeout,
        )
        time.sleep(1.0)
        offset, rates = new_rates(args.miner_log, offset)
        if not rates:
            print(f"batch={batch}: no SM120_RATE samples", flush=True)
            continue
        median = statistics.median(rates)
        scores.append((median, batch))
        print(
            f"batch={batch} requests={requests} "
            f"elapsed={time.monotonic()-started:.1f}s "
            f"median_rolling_ths={median:.6f}",
            flush=True,
        )

    if not scores:
        raise SystemExit("no measurable batch configuration")
    best_rate, best_batch = max(scores)
    print(
        f"best_batch={best_batch} median_rolling_ths={best_rate:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

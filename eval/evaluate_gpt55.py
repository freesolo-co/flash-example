"""evaluate gpt-5.5 on the established flash-example holdout suite."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import evaluate_suite
import httpx

GATEWAY_BASE_URL = os.environ.get("GPT55_BASE_URL", "http://127.0.0.1:8317/v1")
MODEL = "gpt-5.5"
MAX_TOKENS = 8192
TEMPERATURE = 0.0


class GatewayCaseClient:
    def __init__(self, http_client: httpx.Client) -> None:
        self._http_client = http_client
        self.records: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request: Any) -> SimpleNamespace:
        retryable_statuses = {408, 429, 500, 502, 503, 504}
        for attempt in range(6):
            response = self._http_client.post("chat/completions", json=request)
            if not response.is_error:
                break
            if response.status_code not in retryable_statuses or attempt == 5:
                raise RuntimeError(
                    f"gateway request failed with HTTP {response.status_code}: {response.text}"
                )
            time.sleep(min(2**attempt, 30))
        payload = response.json()
        choice = payload["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        finish_reason = choice.get("finish_reason")
        self.records.append(
            {
                "request_messages": request["messages"],
                "response": content,
                "finish_reason": finish_reason,
            }
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )


def evaluate_case(
    http_client: httpx.Client,
    loaded: evaluate_suite.LoadedExample,
    row: dict[str, Any],
) -> dict[str, Any]:
    client = GatewayCaseClient(http_client)
    if loaded.profile.mode == "single":
        result = evaluate_suite.evaluate_single_case(client, loaded, MODEL, row)
    else:
        result = evaluate_suite.evaluate_multi_case(client, loaded, MODEL, row)
    return {
        "result": asdict(result),
        "turn_records": client.records,
    }


def evaluate_example(example_name: str, workers: int, base_url: str) -> dict[str, Any]:
    source_profile = evaluate_suite.PROFILES[example_name]
    profile = replace(source_profile, max_tokens=MAX_TOKENS)
    loaded = evaluate_suite.load_example(profile)
    with httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        timeout=httpx.Timeout(600.0),
        headers={"content-type": "application/json"},
    ) as http_client, ThreadPoolExecutor(max_workers=workers) as executor:
        cases = list(
            executor.map(
                lambda row: evaluate_case(http_client, loaded, row),
                loaded.rows,
            )
        )
    results = [case["result"] for case in cases]
    count = len(results)
    return {
        "example": example_name,
        "model": MODEL,
        "n": count,
        "mean_reward": sum(float(result["score"]) for result in results) / count,
        "component_name": profile.component_name,
        "component_rate": sum(bool(result["success"]) for result in results) / count,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "case_ids": [str(result["id"]) for result in results],
        "truncated_turns": sum(int(result["truncated_turns"]) for result in results),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=sorted(evaluate_suite.PROFILES), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-url", default=GATEWAY_BASE_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.dry_run:
        profile = replace(
            evaluate_suite.PROFILES[args.example], max_tokens=MAX_TOKENS
        )
        loaded = evaluate_suite.load_example(profile)
        payload = {
            "example": args.example,
            "model": MODEL,
            "n": len(loaded.rows),
            "case_ids": [str(row.get("id", "")) for row in loaded.rows],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
    else:
        payload = evaluate_example(args.example, args.workers, args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "cases"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

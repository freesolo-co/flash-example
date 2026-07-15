"""Multi-turn GRPO number guessing with strict JSON actions."""

from __future__ import annotations

import json
import random

from freesolo.datasets import TaskExample
from freesolo.environments import (
    EnvironmentEpisode,
    EnvironmentMultiTurn,
    EnvironmentStepResult,
    RewardResult,
)


def system_prompt(low: int, high: int) -> str:
    # keep the json example in a plain (non f-string) literal so its braces are
    # never parsed as str.format replacement fields
    return (
        f"Guess my secret integer between {low} and {high}. After each guess I reply "
        "higher, lower, or correct. Every reply must be exactly one JSON object like "
        '{"guess":42} with no other keys or text.'
    )


def parse_guess(text: str) -> int | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {"guess"}:
        return None
    guess = value["guess"]
    if isinstance(guess, bool) or not isinstance(guess, int):
        return None
    return guess


def build_dataset(
    num_examples: int = 24, low: int = 1, high: int = 100, max_turns: int = 7
) -> list[dict]:
    rng = random.Random(7)
    return [
        {
            "id": f"number-guess-{index:03d}",
            "input": f"Find the secret number from {low} through {high}.",
            "output": str(secret),
            "metadata": {
                "secret": secret,
                "low": low,
                "high": high,
                "max_turns": max_turns,
            },
        }
        for index, secret in enumerate(rng.randint(low, high) for _ in range(num_examples))
    ]


def metadata(example: TaskExample) -> tuple[int, int, int, int]:
    values = example.metadata or {}
    return (
        int(values["low"]),
        int(values["high"]),
        int(values["secret"]),
        int(values["max_turns"]),
    )


class StructuredNumberGuessEnvironment(EnvironmentMultiTurn):
    def __init__(self, num_examples: int = 24, max_turns: int = 7) -> None:
        self.dataset = build_dataset(num_examples=num_examples, max_turns=max_turns)

    def start_episode(
        self, example: TaskExample, prompt_text: str
    ) -> list[dict[str, str]]:
        _ = prompt_text
        low, high, _, _ = metadata(example)
        return [
            {"role": "system", "content": system_prompt(low, high)},
            {"role": "user", "content": str(example.input)},
        ]

    def max_episode_turns(self, example: TaskExample) -> int:
        return metadata(example)[3]

    def step_episode(
        self,
        example: TaskExample,
        messages: list[dict[str, str]],
        assistant_response: str,
    ) -> EnvironmentStepResult:
        _ = messages
        low, high, secret, _ = metadata(example)
        guess = parse_guess(assistant_response)
        if guess is None or not low <= guess <= high:
            reply = f"invalid: reply with one JSON integer guess from {low} through {high}"
            return EnvironmentStepResult(
                done=False, messages=({"role": "user", "content": reply},)
            )
        if guess == secret:
            return EnvironmentStepResult(
                done=True, messages=({"role": "user", "content": "correct"},)
            )
        reply = "higher" if guess < secret else "lower"
        return EnvironmentStepResult(
            done=False, messages=({"role": "user", "content": reply},)
        )

    def score_episode(
        self, example: TaskExample, episode: EnvironmentEpisode
    ) -> RewardResult:
        _ = example
        solved = any(
            message["role"] == "user" and str(message["content"]).strip() == "correct"
            for message in episode.messages
        )
        assistant_messages = [
            message for message in episode.messages if message["role"] == "assistant"
        ]
        valid = sum(parse_guess(str(message["content"])) is not None for message in assistant_messages)
        format_score = valid / max(1, len(assistant_messages))
        score = (0.9 if solved else 0.0) + 0.1 * format_score
        return RewardResult(
            score=score,
            success=solved,
            reason="secret found" if solved else "secret not found",
        )


def load_environment(**kwargs: object) -> StructuredNumberGuessEnvironment:
    return StructuredNumberGuessEnvironment(
        num_examples=int(kwargs.get("num_examples", 24)),
        max_turns=int(kwargs.get("max_turns", 7)),
    )

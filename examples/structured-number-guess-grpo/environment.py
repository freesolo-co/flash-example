"""Multi-turn number guessing with strict JSON actions."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from freesolo.datasets import TaskExample
from freesolo.environments import (
    EnvironmentEpisode,
    EnvironmentMultiTurn,
    EnvironmentStepResult,
    RewardResult,
)

_DATASET_PATH = Path(__file__).parent / "data" / "train.jsonl"
_RANGE_PATTERN = re.compile(r"from (\d+) through (\d+)\.")


def system_prompt(low: int, high: int) -> str:
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


def parse_bounded_guess(text: str, low: int, high: int) -> int | None:
    guess = parse_guess(text)
    return guess if guess is not None and low <= guess <= high else None


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


def load_distilled_dataset(path: str | Path = _DATASET_PATH) -> list[dict]:
    rows = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            match = _RANGE_PATTERN.search(str(row["input"]))
            if match is None:
                raise ValueError(f"could not parse range from input: {row['input']}")
            low, high = (int(value) for value in match.groups())
            messages = row["output"]["messages"]
            final_assistant = next(
                message for message in reversed(messages) if message["role"] == "assistant"
            )
            secret = parse_bounded_guess(str(final_assistant["content"]), low, high)
            if secret is None:
                raise ValueError("target transcript has no bounded final assistant guess")
            rows.append(
                {
                    "id": f"number-guess-distilled-{index:04d}",
                    "input": row["input"],
                    "output": row["output"],
                    "metadata": {
                        "secret": secret,
                        "low": low,
                        "high": high,
                        "max_turns": 7,
                    },
                }
            )
    return rows


def metadata(example: TaskExample) -> tuple[int, int, int, int]:
    values = example.metadata or {}
    return (
        int(values["low"]),
        int(values["high"]),
        int(values["secret"]),
        int(values["max_turns"]),
    )


class StructuredNumberGuessEnvironment(EnvironmentMultiTurn):
    def __init__(self, dataset_path: str | Path = _DATASET_PATH) -> None:
        self.dataset = load_distilled_dataset(dataset_path)

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
        guess = parse_bounded_guess(assistant_response, low, high)
        if guess is None:
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
        low, high, secret, _ = metadata(example)
        guesses = [
            parse_bounded_guess(str(message["content"]), low, high)
            for message in episode.messages
            if message["role"] == "assistant"
        ]
        solved = secret in guesses
        valid = sum(guess is not None for guess in guesses)
        format_score = valid / max(1, len(guesses))
        score = (0.9 if solved else 0.0) + 0.1 * format_score
        return RewardResult(
            score=score,
            success=solved,
            reason="secret found" if solved else "secret not found",
        )

    def sft_completion(self, example: TaskExample) -> list[dict[str, str]]:
        output = example.output
        if not isinstance(output, dict) or not isinstance(output.get("messages"), list):
            raise ValueError("number-guess SFT rows require output.messages")
        return [dict(message) for message in output["messages"]]


def load_environment(**kwargs: object) -> StructuredNumberGuessEnvironment:
    dataset_path = kwargs.get("dataset_path", _DATASET_PATH)
    environment = StructuredNumberGuessEnvironment(dataset_path=str(dataset_path))
    if "num_examples" in kwargs:
        environment.dataset = build_dataset(
            num_examples=int(kwargs["num_examples"]),
            low=int(kwargs.get("low", 1)),
            high=int(kwargs.get("high", 100)),
            max_turns=int(kwargs.get("max_turns", 7)),
        )
    return environment

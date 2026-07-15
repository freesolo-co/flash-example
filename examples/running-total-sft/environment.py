"""Deterministic multi-turn SFT environment for running totals."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from freesolo.datasets import TaskExample
from freesolo.environments import (
    EnvironmentEpisode,
    EnvironmentMultiTurn,
    EnvironmentStepResult,
    RewardResult,
)

SYSTEM_PROMPT = (
    "I will give you one integer at a time. After each number, reply with only "
    "the running total as a bare integer. Do not add words or punctuation."
)
_INTEGER = re.compile(r"^[+-]?\d+$")


def number_prompt(value: int) -> str:
    return f"Number: {value}"


def running_totals(numbers: Sequence[int]) -> list[int]:
    total = 0
    values = []
    for number in numbers:
        total += number
        values.append(total)
    return values


def gold_completion(numbers: Sequence[int]) -> list[dict[str, str]]:
    totals = running_totals(numbers)
    messages = [{"role": "assistant", "content": str(totals[0])}]
    for index in range(1, len(numbers)):
        messages.append({"role": "user", "content": number_prompt(numbers[index])})
        messages.append({"role": "assistant", "content": str(totals[index])})
    return messages


def numbers_for(example: TaskExample) -> list[int]:
    return [int(value) for value in (example.metadata or {})["numbers"]]


def assistant_replies(messages: Sequence[dict[str, str]]) -> list[str]:
    return [str(message["content"]) for message in messages if message["role"] == "assistant"]


def build_dataset(num_examples: int = 24, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(num_examples):
        count = 3 + index % 2
        numbers = [rng.randint(1, 9) for _ in range(count)]
        rows.append(
            {
                "id": f"running-total-{index:03d}",
                "input": "Numbers: " + " ".join(str(value) for value in numbers),
                "output": {"messages": gold_completion(numbers)},
                "metadata": {"numbers": numbers},
            }
        )
    return rows


class RunningTotalEnvironment(EnvironmentMultiTurn):
    def __init__(self, num_examples: int = 24, seed: int = 0) -> None:
        self.dataset = build_dataset(num_examples, seed)

    def start_episode(
        self, example: TaskExample, prompt_text: str
    ) -> list[dict[str, str]]:
        _ = prompt_text
        numbers = numbers_for(example)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": number_prompt(numbers[0])},
        ]

    def max_episode_turns(self, example: TaskExample) -> int:
        return len(numbers_for(example))

    def step_episode(
        self,
        example: TaskExample,
        messages: list[dict[str, str]],
        assistant_response: str,
    ) -> EnvironmentStepResult:
        _ = assistant_response
        numbers = numbers_for(example)
        answered = len(assistant_replies(messages))
        if answered >= len(numbers):
            return EnvironmentStepResult(done=True)
        return EnvironmentStepResult(
            done=False,
            messages=({"role": "user", "content": number_prompt(numbers[answered])},),
        )

    def score_episode(
        self, example: TaskExample, episode: EnvironmentEpisode
    ) -> RewardResult:
        expected = running_totals(numbers_for(example))
        replies = assistant_replies(episode.messages)
        correct = sum(
            bool(_INTEGER.fullmatch(reply.strip())) and int(reply.strip()) == total
            for reply, total in zip(replies, expected, strict=False)
        )
        score = correct / len(expected)
        return RewardResult(
            score=score,
            success=correct == len(expected),
            reason=f"{correct}/{len(expected)} running totals exact",
        )

    def sft_completion(self, example: TaskExample) -> list[dict[str, str]]:
        return gold_completion(numbers_for(example))


def load_environment(**kwargs: object) -> RunningTotalEnvironment:
    return RunningTotalEnvironment(
        num_examples=int(kwargs.get("num_examples", 24)),
        seed=int(kwargs.get("seed", 0)),
    )

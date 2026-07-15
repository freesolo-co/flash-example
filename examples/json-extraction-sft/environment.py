"""Single-turn SFT environment for strict customer-message JSON extraction."""

from __future__ import annotations

import json

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

SYSTEM_PROMPT = (
    "Extract the customer request as one JSON object with exactly these fields: "
    "category, priority, and summary. category must be billing, delivery, returns, "
    "account, or product. priority must be low, medium, or high. Keep summary short."
)

_CASES = (
    ("I was charged twice for order 18.", "billing", "high", "duplicate charge for order 18"),
    ("Can I change the email on my account?", "account", "low", "change account email"),
    ("My package was due yesterday and is still missing.", "delivery", "high", "overdue package is missing"),
    ("The blue mug arrived cracked.", "product", "medium", "blue mug arrived cracked"),
    ("I want to return shoes that do not fit.", "returns", "medium", "return shoes that do not fit"),
    ("Where can I download last month's invoice?", "billing", "low", "download last month's invoice"),
    ("Tracking has not updated for three days.", "delivery", "medium", "tracking has not updated"),
    ("Please reset my locked account.", "account", "high", "reset locked account"),
)


def label(category: str, priority: str, summary: str) -> dict[str, str]:
    return {"category": category, "priority": priority, "summary": summary}


def encode_label(value: dict[str, str]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def parse_label(text: str) -> dict[str, str] | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {"category", "priority", "summary"}:
        return None
    if not all(isinstance(item, str) for item in value.values()):
        return None
    return value


def build_dataset() -> list[dict]:
    rows = []
    for index in range(24):
        message, category, priority, summary = _CASES[index % len(_CASES)]
        expected = label(category, priority, summary)
        rows.append(
            {
                "id": f"json-{index:03d}",
                "input": message,
                "output": encode_label(expected),
                "metadata": {"expected": expected},
            }
        )
    return rows


class JsonExtractionEnvironment(EnvironmentSingleTurn):
    def __init__(self) -> None:
        self.dataset = build_dataset()

    def build_prompt_messages(
        self, example: TaskExample, prompt_text: str
    ) -> list[dict[str, str]]:
        _ = prompt_text
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(example.input)},
        ]

    def sft_completion(self, example: TaskExample) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": str(example.output)}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        predicted = parse_label(response_text)
        expected = (example.metadata or {}).get("expected")
        correct = predicted == expected
        return RewardResult(
            score=1.0 if correct else 0.0,
            success=correct,
            reason="exact JSON label match" if correct else "JSON label did not match",
        )


def load_environment(**kwargs: object) -> JsonExtractionEnvironment:
    _ = kwargs
    return JsonExtractionEnvironment()

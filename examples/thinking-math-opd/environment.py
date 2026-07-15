"""Single-turn OPD environment for synthetic grade-school arithmetic."""

from __future__ import annotations

import re

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

SYSTEM_PROMPT = (
    "Solve the word problem carefully. End your response with exactly one final line "
    "in the form Answer: <integer>."
)
_FINAL_ANSWER = re.compile(r"(?:^|\n)Answer:\s*(-?\d+)\s*$", re.IGNORECASE)


def extract_answer(text: str) -> int | None:
    match = _FINAL_ANSWER.search(text.strip())
    return int(match.group(1)) if match else None


def build_dataset() -> list[dict]:
    rows = []
    for index in range(24):
        first = 4 + index
        added = 2 + index % 6
        if index % 3 == 0:
            question = (
                f"Mina has {first} marbles and receives {added} more. "
                "How many marbles does she have now?"
            )
            answer = first + added
        elif index % 3 == 1:
            total = first + added + 5
            question = (
                f"A shelf holds {total} books. {added} are borrowed. "
                "How many books remain?"
            )
            answer = total - added
        else:
            groups = 2 + index % 4
            each = 3 + index % 5
            question = (
                f"There are {groups} baskets with {each} apples in each basket. "
                "How many apples are there altogether?"
            )
            answer = groups * each
        rows.append(
            {
                "id": f"math-{index:03d}",
                "input": question,
                "output": f"Answer: {answer}",
                "metadata": {"answer": answer},
            }
        )
    return rows


class ThinkingMathEnvironment(EnvironmentSingleTurn):
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

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        predicted = extract_answer(response_text)
        expected = int((example.metadata or {})["answer"])
        correct = predicted == expected
        return RewardResult(
            score=1.0 if correct else 0.0,
            success=correct,
            reason="exact final number" if correct else "final number was missing or incorrect",
        )


def load_environment(**kwargs: object) -> ThinkingMathEnvironment:
    _ = kwargs
    return ThinkingMathEnvironment()

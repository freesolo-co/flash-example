"""Single-turn GRPO environment for nested boolean expressions."""

from __future__ import annotations

import random
import re

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

SYSTEM_PROMPT = (
    "Please solve the following boolean logic problem. You may reason inside "
    "<think>...</think>. Evaluate the expression and end with exactly one final "
    "answer: <answer>True</answer> or <answer>False</answer>."
)
_DATASET_SEED = 20260716
_DATASET_SIZE = 24
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _generate_expression(rng: random.Random, depth: int) -> tuple[str, bool]:
    if depth <= 0:
        value = rng.choice((True, False))
        return str(value), value

    operation = rng.choices(("not", "and", "or"), weights=(3, 4, 4), k=1)[0]
    if operation == "not":
        child_text, child_value = _generate_expression(rng, depth - 1)
        return f"not ( {child_text} )", not child_value

    left_text, left_value = _generate_expression(rng, depth - 1)
    right_depth = rng.randint(max(0, depth - 2), depth - 1)
    right_text, right_value = _generate_expression(rng, right_depth)
    expression = f"( {left_text} {operation} {right_text} )"
    if operation == "and":
        return expression, left_value and right_value
    return expression, left_value or right_value


def extract_answer(text: str) -> str | None:
    if "<think>" in text and "</think>" not in text:
        return None

    answer_text = text
    if "</think>" in answer_text:
        answer_text = answer_text.split("</think>", 1)[1]

    matches = _ANSWER_PATTERN.findall(answer_text)
    if not matches:
        return None

    normalized = matches[-1].strip().casefold()
    if normalized == "true":
        return "True"
    if normalized == "false":
        return "False"
    return None


def build_dataset() -> list[dict]:
    rng = random.Random(_DATASET_SEED)
    rows = []
    for index in range(_DATASET_SIZE):
        depth = 1 + index % 5
        expression, value = _generate_expression(rng, depth)
        answer = str(value)
        rows.append(
            {
                "id": f"boolean-expression-{index:03d}",
                "input": (
                    "Evaluate the following boolean expression:\n\n"
                    f"{expression}\n\n"
                    "Is the expression True or False?"
                ),
                "output": f"<answer>{answer}</answer>",
                "metadata": {
                    "answer": answer,
                    "expression": expression,
                    "depth": depth,
                },
            }
        )
    return rows


class LogicBooleanEnvironment(EnvironmentSingleTurn):
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
        expected = str((example.metadata or {})["answer"])
        correct = predicted == expected
        if correct:
            reason = "exact answer match"
        elif predicted is None:
            reason = "missing or malformed final answer"
        else:
            reason = "final answer was incorrect"
        return RewardResult(
            score=1.0 if correct else 0.0,
            success=correct,
            reason=reason,
        )


def load_environment(**kwargs: object) -> LogicBooleanEnvironment:
    _ = kwargs
    return LogicBooleanEnvironment()

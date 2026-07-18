"""Single-turn SFT and GRPO environment for nested boolean expressions."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

SYSTEM_PROMPT = (
    "Please solve the following boolean logic problem. You may reason inside "
    "<think>...</think>. Evaluate the expression and end with exactly one final "
    "answer: <answer>True</answer> or <answer>False</answer>."
)
_DATASET_SEED = 20260716
_DATASET_SIZE = 24
_DATASET_PATH = Path(__file__).parent / "data" / "train.jsonl"
_ANSWER_PATTERN = re.compile(
    r"\s*(?:<think>.*?</think>\s*)?<answer>(True|False)</answer>\s*",
    re.DOTALL,
)


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
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    match = _ANSWER_PATTERN.fullmatch(text)
    return match.group(1) if match is not None else None


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


def load_distilled_dataset(path: str | Path = _DATASET_PATH) -> list[dict]:
    rows = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            answer = extract_answer(str(row["output"]))
            if answer is None:
                raise ValueError(f"distilled row {index} violates the strict answer contract")
            rows.append(
                {
                    "id": f"boolean-distilled-{index:04d}",
                    "input": row["input"],
                    "output": row["output"],
                    "metadata": {"answer": answer},
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
        values = example.metadata or dict(example.record.get("metadata") or {})
        expected = str(values["answer"])
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
    environment = LogicBooleanEnvironment()
    if bool(kwargs.get("use_distilled", False)):
        environment.dataset = load_distilled_dataset()
    return environment

"""Single-turn GRPO environment for boxed-answer mathematics."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

SYSTEM_PROMPT = (
    "Solve the following math problem. Explain your reasoning and put the final answer in "
    "\\boxed{}."
)
_BOXED_MARKER = r"\boxed{"
_DATASET_PATH = Path(__file__).parent / "data" / "train.jsonl"
_SIMPLE_LATEX_FRACTION = re.compile(
    r"\\frac\s*\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}"
    r"\s*\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}"
)

_PROBLEMS = (
    ("Compute 47 + 38.", "85"),
    ("Subtract 68 from 125.", "57"),
    ("Compute 14 times 9.", "126"),
    ("Divide 144 by 12.", "12"),
    ("Evaluate 7(8 - 3) + 6.", "41"),
    ("What is 15 percent of 240?", "36"),
    ("Compute 3/4 + 5/8 and give the result in simplest form.", "11/8"),
    ("Compute 7/9 - 1/3 and give the result in simplest form.", "4/9"),
    ("Compute (2/5)(15/8) and give the result in simplest form.", "3/4"),
    ("Compute (3/4) divided by (5/6) and give the result in simplest form.", "9/10"),
    (
        "Nora spends 1/4 of her savings on a book and 3/8 of her original savings "
        "on a ticket. What fraction of her original savings remains?",
        "3/8",
    ),
    (
        "A recipe uses 3/4 cup of oats for 6 servings. How many cups of oats are "
        "needed for 20 servings?",
        "5/2",
    ),
    ("Solve for x: 3x + 7 = 31.", "8"),
    ("Solve for x: 5(x - 2) = 35.", "9"),
    ("Solve for x: 2x + 3 = x + 11.", "8"),
    ("Solve for x: x/4 + 6 = 11.", "20"),
    (
        "Three consecutive integers have sum 51. What is the middle integer?",
        "17",
    ),
    (
        "A rectangle has perimeter 46 units and length 15 units. What is its width?",
        "8",
    ),
    ("Find the greatest common divisor of 84 and 126.", "42"),
    ("Find the least common multiple of 12 and 18.", "36"),
    ("What is the remainder when 157 is divided by 12?", "1"),
    ("Find the smallest positive integer divisible by 6, 8, and 15.", "120"),
    (
        "The angles of a triangle are in the ratio 2:3:4. What is the largest angle "
        "in degrees?",
        "80",
    ),
    (
        "A bag contains 5 red marbles and 3 blue marbles. If one marble is chosen "
        "uniformly at random, what is the probability it is red?",
        "5/8",
    ),
)


def extract_boxed_answer(text: str) -> str | None:
    answers: list[str] = []
    search_from = 0
    while True:
        marker_index = text.find(_BOXED_MARKER, search_from)
        if marker_index < 0:
            break
        opening_index = marker_index + len(_BOXED_MARKER) - 1
        depth = 0
        closing_index = None
        for index in range(opening_index, len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    closing_index = index
                    break
        if closing_index is None:
            search_from = marker_index + len(_BOXED_MARKER)
            continue
        answers.append(text[opening_index + 1 : closing_index])
        search_from = closing_index + 1
    return answers[-1] if answers else None


def normalize_answer(text: str) -> str:
    normalized = text.strip()
    while len(normalized) >= 2 and normalized.startswith("$") and normalized.endswith("$"):
        normalized = normalized[1:-1].strip()
    normalized = normalized.replace(",", "")
    normalized = _SIMPLE_LATEX_FRACTION.sub(r"\1/\2", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    if normalized.startswith("+"):
        normalized = normalized[1:]
    return normalized


def _parse_rational(text: str) -> Fraction | None:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def answers_equivalent(predicted: str, expected: str) -> bool:
    normalized_predicted = normalize_answer(predicted)
    normalized_expected = normalize_answer(expected)
    predicted_rational = _parse_rational(normalized_predicted)
    expected_rational = _parse_rational(normalized_expected)
    if predicted_rational is not None and expected_rational is not None:
        return predicted_rational == expected_rational
    return normalized_predicted.casefold() == normalized_expected.casefold()


def build_dataset() -> list[dict]:
    return [
        {
            "id": f"math-boxed-{index:03d}",
            "input": problem,
            "output": f"\\boxed{{{answer}}}",
            "metadata": {"answer": answer},
        }
        for index, (problem, answer) in enumerate(_PROBLEMS)
    ]


def load_distilled_dataset(path: str | Path = _DATASET_PATH) -> list[dict]:
    rows = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            answer = extract_boxed_answer(str(row["output"]))
            if answer is None:
                raise ValueError(f"distilled row {index} has no boxed final answer")
            rows.append(
                {
                    "id": f"math-boxed-distilled-{index:04d}",
                    "input": row["input"],
                    "output": row["output"],
                    "metadata": {"answer": answer},
                }
            )
    return rows


class MathBoxedEnvironment(EnvironmentSingleTurn):
    def __init__(self, dataset_path: str | Path = _DATASET_PATH) -> None:
        self.dataset = load_distilled_dataset(dataset_path)

    def build_prompt_messages(
        self, example: TaskExample, prompt_text: str
    ) -> list[dict[str, str]]:
        _ = prompt_text
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(example.input)},
        ]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        predicted = extract_boxed_answer(response_text)
        if predicted is None:
            return RewardResult(
                score=0.0,
                success=False,
                reason="missing boxed final answer",
            )
        expected = str((example.metadata or {})["answer"])
        correct = answers_equivalent(predicted, expected)
        return RewardResult(
            score=1.0 if correct else 0.0,
            success=correct,
            reason=(
                "boxed answer is equivalent"
                if correct
                else "boxed answer was incorrect"
            ),
        )


def load_environment(**kwargs: object) -> MathBoxedEnvironment:
    dataset_path = kwargs.get("dataset_path", _DATASET_PATH)
    return MathBoxedEnvironment(dataset_path=str(dataset_path))

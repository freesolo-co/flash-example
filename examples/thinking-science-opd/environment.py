"""Single-turn GRPO environment for original science questions."""

from __future__ import annotations

import re

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

SYSTEM_PROMPT = (
    "Choose the best answer. You may reason before answering, but end with exactly "
    "one final line in the form Answer: A, Answer: B, Answer: C, or Answer: D."
)
_FINAL_LETTER = re.compile(r"(?:^|\n)Answer:\s*([ABCD])\s*$", re.IGNORECASE)

_QUESTIONS = (
    ("Which state of matter has a fixed volume but takes the shape of its container?", ("Solid", "Liquid", "Gas", "Plasma"), "B"),
    ("What force pulls objects toward Earth?", ("Magnetism", "Friction", "Gravity", "Buoyancy"), "C"),
    ("Which organ pumps blood through the body?", ("Lung", "Heart", "Kidney", "Stomach"), "B"),
    ("Plants primarily use which gas during photosynthesis?", ("Oxygen", "Nitrogen", "Carbon dioxide", "Helium"), "C"),
    ("Which material is the best electrical conductor?", ("Copper", "Rubber", "Glass", "Wood"), "A"),
    ("What causes day and night on Earth?", ("Earth's rotation", "Earth's orbit", "The Moon's orbit", "Cloud movement"), "A"),
    ("Which simple machine is a ramp?", ("Pulley", "Lever", "Inclined plane", "Wheel and axle"), "C"),
    ("At sea level, pure water freezes at what Celsius temperature?", ("0", "10", "50", "100"), "A"),
)


def extract_answer(text: str) -> str | None:
    match = _FINAL_LETTER.search(text.strip())
    return match.group(1).upper() if match else None


def format_question(question: str, choices: tuple[str, str, str, str]) -> str:
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices, strict=True)
    )
    return f"{question}\n{options}"


def build_dataset() -> list[dict]:
    rows = []
    for index in range(24):
        question, choices, answer = _QUESTIONS[index % len(_QUESTIONS)]
        rows.append(
            {
                "id": f"science-{index:03d}",
                "input": format_question(question, choices),
                "output": f"Answer: {answer}",
                "metadata": {"answer": answer},
            }
        )
    return rows


class ThinkingScienceEnvironment(EnvironmentSingleTurn):
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
        return RewardResult(
            score=1.0 if correct else 0.0,
            success=correct,
            reason="exact final answer letter" if correct else "final answer letter was incorrect",
        )


def load_environment(**kwargs: object) -> ThinkingScienceEnvironment:
    _ = kwargs
    return ThinkingScienceEnvironment()

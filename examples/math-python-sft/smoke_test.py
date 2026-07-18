"""Network-free smoke test for the math Python environment."""

from __future__ import annotations

from environment import (
    build_dataset,
    extract_boxed_answer,
    extract_python_code,
    load_environment,
    normalize_answer,
)
from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEpisode


def task_example(row: dict) -> TaskExample:
    return TaskExample(
        record=row,
        id=row["id"],
        input=row["input"],
        output=row["output"],
        metadata=dict(row["metadata"]),
    )


def main() -> None:
    rows = build_dataset()
    assert len(rows) == 24
    assert rows == build_dataset()
    assert extract_python_code("```python\nprint(1)\n```\n```python\nprint(2)\n```") == "print(2)"
    assert extract_boxed_answer(r"work \\boxed{\\frac{1}{2}}") == r"\\frac{1}{2}"
    assert extract_boxed_answer(r"incomplete \\boxed{12") is None
    assert normalize_answer("0.5") == normalize_answer("1/2")

    environment = load_environment()
    assert len(environment.dataset) == 97
    example = task_example(rows[0])
    initial = environment.start_episode(example, "")
    code_turn = "I will calculate it.\n```python\nprint(987654321 * 123456789)\n```"
    tool_step = environment.step_episode(example, initial, code_turn)
    assert tool_step.done is False
    assert len(tool_step.messages) == 1
    assert "121932631112635269" in tool_step.messages[0]["content"]

    final_turn = r"The computed result is \\boxed{121932631112635269}."
    transcript = tuple(initial) + (
        {"role": "assistant", "content": code_turn},
        tool_step.messages[0],
        {"role": "assistant", "content": final_turn},
    )
    final_step = environment.step_episode(example, list(transcript[:-1]), final_turn)
    assert final_step.done is True
    correct_episode = EnvironmentEpisode(messages=transcript, response_text=final_turn)
    correct_reward = environment.score_episode(example, correct_episode)
    assert correct_reward.score == 1.0
    assert correct_reward.success is True

    wrong_turn = r"The computed result is \\boxed{0}."
    wrong_transcript = transcript[:-1] + ({"role": "assistant", "content": wrong_turn},)
    wrong_episode = EnvironmentEpisode(messages=wrong_transcript, response_text=wrong_turn)
    wrong_reward = environment.score_episode(example, wrong_episode)
    assert wrong_reward.score == 0.0
    assert wrong_reward.success is False
    print("math python smoke passed")


if __name__ == "__main__":
    main()

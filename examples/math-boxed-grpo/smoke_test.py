from __future__ import annotations

from environment import build_dataset, extract_boxed_answer, load_environment
from freesolo.datasets import TaskExample


def task_example(row: dict) -> TaskExample:
    return TaskExample(
        record=row,
        id=row["id"],
        input=str(row["input"]),
        output=row["output"],
        metadata=dict(row["metadata"]),
    )


def main() -> None:
    rows = build_dataset()
    assert len(rows) == 24
    assert rows == build_dataset()
    nested = r"First \boxed{3}, but the final result is \boxed{\frac{1}{2}}."
    assert extract_boxed_answer(nested) == r"\frac{1}{2}"

    environment = load_environment()
    assert len(environment.dataset) == 143
    example = task_example(rows[0])
    correct = environment.score_response(example, r"47 + 38 = 85, so \boxed{85}.")
    wrong = environment.score_response(example, r"I obtain \boxed{84}.")
    assert correct.score == 1.0 and correct.success is True
    assert wrong.score == 0.0 and wrong.success is False

    fraction_row = {
        "id": "fraction-check",
        "input": "Write one half as a decimal.",
        "output": r"\boxed{1/2}",
        "metadata": {"answer": "1/2"},
    }
    decimal = environment.score_response(
        task_example(fraction_row), r"One half is 0.5, so \boxed{0.5}."
    )
    assert decimal.score == 1.0 and decimal.success is True
    print("math boxed smoke passed")


if __name__ == "__main__":
    main()

from __future__ import annotations

from environment import LogicBooleanEnvironment, build_dataset, extract_answer
from freesolo.datasets import TaskExample


def main() -> None:
    rows = build_dataset()
    assert len(rows) == 24
    assert rows == build_dataset()

    for row in rows:
        expression = str(row["metadata"]["expression"])
        trusted_value = eval(expression, {"__builtins__": {}}, {})
        assert isinstance(trusted_value, bool)
        assert str(trusted_value) == row["metadata"]["answer"]

    environment = LogicBooleanEnvironment()
    first = rows[0]
    example = TaskExample(
        record=first,
        id=first["id"],
        input=first["input"],
        output=first["output"],
        metadata=first["metadata"],
    )
    expected = str(first["metadata"]["answer"])
    wrong = "False" if expected == "True" else "True"

    correct_result = environment.score_response(
        example,
        f"<think>evaluate structurally</think><answer>{expected.lower()}</answer>",
    )
    wrong_result = environment.score_response(
        example,
        f"<answer>{wrong}</answer>",
    )
    unfinished_result = environment.score_response(
        example,
        "<think>still evaluating",
    )

    assert correct_result.score == 1.0 and correct_result.success
    assert wrong_result.score == 0.0 and not wrong_result.success
    assert unfinished_result.score == 0.0 and not unfinished_result.success
    assert extract_answer("<answer>False</answer><answer>True</answer>") == "True"
    assert extract_answer("<answer>True") is None
    print("logic boolean smoke passed")


if __name__ == "__main__":
    main()

"""Network-free checks for the Sudoku multi-turn environment."""

from __future__ import annotations

from environment import (
    SudokuBoard,
    build_dataset,
    extract_move,
    format_move,
    load_environment,
    parse_move_string,
)
from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEpisode


def task_example(row: dict) -> TaskExample:
    return TaskExample(
        record=row,
        id=row.get("id"),
        input=str(row["input"]),
        output=row.get("output"),
        metadata=dict(row.get("metadata") or {}),
    )


def main() -> None:
    assert parse_move_string(extract_move("<think>check</think><move>A1=5</move>")) == (0, 0, 5)
    assert parse_move_string(extract_move("<MOVE>i9=0</MOVE>")) == (8, 8, 0)
    assert parse_move_string(extract_move("A1=5")) is None
    assert build_dataset(2) == build_dataset(2)

    environment = load_environment()
    assert len(environment.dataset) == 99
    row = environment.dataset[0]
    example = task_example(row)
    metadata = dict(example.metadata or {})
    puzzle = metadata["puzzle"]
    solution = metadata["solution"]
    assert SudokuBoard(solution).is_solved()

    initial_messages = environment.start_episode(example, "")
    invalid = environment.step_episode(example, initial_messages, "A1=5")
    assert invalid.done is False
    assert "Invalid move format" in invalid.messages[0]["content"]
    assert "<move>A1=5</move>" in invalid.messages[0]["content"]

    messages = environment.start_episode(example, "")
    last_response = ""
    empty_cells = [
        (row_index, col_index)
        for row_index in range(9)
        for col_index in range(9)
        if puzzle[row_index][col_index] == 0
    ]
    assert 8 <= len(empty_cells) <= 12
    for move_index, (row_index, col_index) in enumerate(empty_cells):
        move = format_move(row_index, col_index, solution[row_index][col_index])
        last_response = f"<think>apply the known valid value</think>\n<move>{move}</move>"
        step = environment.step_episode(example, messages, last_response)
        messages.append({"role": "assistant", "content": last_response})
        messages.extend(step.messages)
        assert step.done is (move_index == len(empty_cells) - 1)

    episode = EnvironmentEpisode(messages=tuple(messages), response_text=last_response)
    reward = environment.score_episode(example, episode)
    assert reward.success is True
    assert reward.score > 0
    print("sudoku smoke passed")


if __name__ == "__main__":
    main()

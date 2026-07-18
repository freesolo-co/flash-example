"""Drive a deployed Sudoku adapter through a complete multi-turn episode."""

from __future__ import annotations

import os

from environment import (
    SudokuBoard,
    extract_move,
    format_move,
    load_environment,
    parse_move_string,
)
from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEpisode
from openai import OpenAI


def task_example(row: dict) -> TaskExample:
    return TaskExample(
        record=row,
        id=row.get("id"),
        input=str(row["input"]),
        output=row.get("output"),
        metadata=dict(row.get("metadata") or {}),
    )


def main() -> None:
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    environment = load_environment(num_examples=1)
    example = task_example(environment.dataset[0])
    metadata = dict(example.metadata or {})
    solution = metadata["solution"]
    messages = environment.start_episode(example, "")
    last_response = ""

    for turn in range(1, environment.max_episode_turns(example) + 1):
        response = client.chat.completions.create(
            model=os.environ["FLASH_RUN_ID"],
            messages=messages,
            temperature=0,
            max_tokens=384,
        )
        last_response = response.choices[0].message.content or ""
        parsed = parse_move_string(extract_move(last_response))
        if parsed is None:
            predicted = "invalid format"
            expected = "one action like <move>A1=5</move>"
        else:
            row, col, num = parsed
            predicted = format_move(row, col, num)
            expected = format_move(row, col, solution[row][col])
        print(f"turn={turn} predicted={predicted} expected={expected}")

        step = environment.step_episode(example, messages, last_response)
        messages.append({"role": "assistant", "content": last_response})
        messages.extend(step.messages)
        if step.done:
            break

    episode = EnvironmentEpisode(messages=tuple(messages), response_text=last_response)
    reward = environment.score_episode(example, episode)
    predicted_board = SudokuBoard(metadata["puzzle"])
    for message in messages:
        if message["role"] != "assistant":
            continue
        move = parse_move_string(extract_move(str(message["content"])))
        if move is not None:
            predicted_board.make_move(*move)

    print("predicted board:")
    print(predicted_board.to_string())
    print("expected board:")
    print(SudokuBoard(solution).to_string())
    if not reward.success or predicted_board.board != solution:
        raise ValueError(f"model did not solve the Sudoku puzzle: {reward.reason}")
    print(f"score={reward.score:.3f} success={reward.success}")


if __name__ == "__main__":
    main()

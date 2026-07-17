"""Evaluate a deployed math Python adapter through a local tool loop."""

from __future__ import annotations

import os

from environment import extract_boxed_answer, load_environment, normalize_answer
from freesolo.datasets import TaskExample
from openai import OpenAI


def task_example(row: dict) -> TaskExample:
    return TaskExample(
        record=row,
        id=row["id"],
        input=row["input"],
        output=row["output"],
        metadata=dict(row["metadata"]),
    )


def main() -> None:
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    environment = load_environment(num_examples=24, max_turns=4)
    row = environment.dataset[9]
    example = task_example(row)
    expected = str(row["metadata"]["answer"])
    messages = environment.start_episode(example, "")
    predicted = None

    for _ in range(environment.max_episode_turns(example)):
        response = client.chat.completions.create(
            model=os.environ["FLASH_RUN_ID"],
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        predicted = extract_boxed_answer(content)
        step = environment.step_episode(example, messages, content)
        messages.extend(step.messages)
        if step.done:
            break

    print(f"predicted={predicted!r}")
    print(f"expected={expected!r}")
    if predicted is None or normalize_answer(predicted) != normalize_answer(expected):
        raise ValueError("deployed adapter returned an incorrect boxed answer")


if __name__ == "__main__":
    main()

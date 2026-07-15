"""Drive a deployed running-total adapter through a full dialogue."""

from __future__ import annotations

import os

from openai import OpenAI


def main() -> None:
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    numbers = [3, 5, 2]
    messages = [
        {
            "role": "system",
            "content": "Reply with only the running total after each number.",
        }
    ]
    total = 0
    for number in numbers:
        messages.append({"role": "user", "content": f"Number: {number}"})
        response = client.chat.completions.create(
            model=os.environ["FLASH_RUN_ID"], messages=messages, temperature=0
        )
        answer = (response.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": answer})
        total += number
        print(f"number={number} model={answer} expected={total}")
        if answer != str(total):
            raise ValueError("running total was incorrect")


if __name__ == "__main__":
    main()

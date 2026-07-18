"""Call and evaluate a deployed boxed-answer math adapter."""

from __future__ import annotations

import os

from environment import SYSTEM_PROMPT, answers_equivalent, extract_boxed_answer
from openai import OpenAI


def main() -> None:
    expected = "11/8"
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    response = client.chat.completions.create(
        model=os.environ["FLASH_RUN_ID"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Compute 3/4 + 5/8 and give the result in simplest form.",
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    predicted = extract_boxed_answer(content)
    print(content)
    print(f"predicted={predicted} expected={expected}")
    if predicted is None or not answers_equivalent(predicted, expected):
        raise ValueError("boxed final answer was missing or incorrect")


if __name__ == "__main__":
    main()

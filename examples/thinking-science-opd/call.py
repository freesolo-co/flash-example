"""Call and evaluate a deployed thinking science adapter."""

from __future__ import annotations

import os
import re

from openai import OpenAI

_FINAL_ANSWER = re.compile(r"(?:^|\n)Answer:\s*([ABCD])\s*$", re.IGNORECASE)


def main() -> None:
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    response = client.chat.completions.create(
        model=os.environ["FLASH_RUN_ID"],
        messages=[
            {
                "role": "system",
                "content": "Choose the best answer and end with Answer: <letter>.",
            },
            {
                "role": "user",
                "content": (
                    "Which process changes liquid water into water vapor?\n"
                    "A. Freezing\nB. Melting\nC. Evaporation\nD. Condensation"
                ),
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    match = _FINAL_ANSWER.search(content.strip())
    predicted = match.group(1).upper() if match else None
    print(content)
    print(f"predicted={predicted} expected=C")
    if predicted != "C":
        raise ValueError("final answer letter was incorrect")


if __name__ == "__main__":
    main()

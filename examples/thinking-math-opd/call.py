"""Call and evaluate a deployed thinking math adapter."""

from __future__ import annotations

import os
import re

from openai import OpenAI

_FINAL_ANSWER = re.compile(r"(?:^|\n)Answer:\s*(-?\d+)\s*$", re.IGNORECASE)


def main() -> None:
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    response = client.chat.completions.create(
        model=os.environ["FLASH_RUN_ID"],
        messages=[
            {"role": "system", "content": "Solve carefully and end with Answer: <integer>."},
            {
                "role": "user",
                "content": "A tray has 6 rows with 4 cookies in each row. How many cookies are there?",
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    match = _FINAL_ANSWER.search(content.strip())
    predicted = int(match.group(1)) if match else None
    print(content)
    print(f"predicted={predicted} expected=24")
    if predicted != 24:
        raise ValueError("final answer was incorrect")


if __name__ == "__main__":
    main()

"""Drive a deployed number-guess adapter using strict JSON turns."""

from __future__ import annotations

import json
import os

from openai import OpenAI

SCHEMA = {
    "type": "object",
    "properties": {"guess": {"type": "integer", "minimum": 1, "maximum": 100}},
    "required": ["guess"],
    "additionalProperties": False,
}


def main() -> None:
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    secret = 42
    messages = [
        {
            "role": "system",
            "content": "Guess an integer from 1 through 100. Reply only as JSON with key guess.",
        },
        {"role": "user", "content": "Find my secret number."},
    ]
    for turn in range(1, 8):
        response = client.chat.completions.create(
            model=os.environ["FLASH_RUN_ID"],
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "number_guess", "strict": True, "schema": SCHEMA},
            },
        )
        content = response.choices[0].message.content or ""
        guess = json.loads(content)["guess"]
        messages.append({"role": "assistant", "content": content})
        feedback = "correct" if guess == secret else "higher" if guess < secret else "lower"
        print(f"turn={turn} guess={guess} feedback={feedback}")
        if feedback == "correct":
            return
        messages.append({"role": "user", "content": feedback})
    raise ValueError("model did not find the secret within seven turns")


if __name__ == "__main__":
    main()

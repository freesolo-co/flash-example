"""Drive a deployed number-guess adapter using strict JSON turns."""

from __future__ import annotations

import json
import os

from openai import OpenAI


def schema_for_metadata(metadata: dict[str, object]) -> dict[str, object]:
    low = int(metadata["low"])
    high = int(metadata["high"])
    if low > high:
        raise ValueError("number-guess lower bound must not exceed upper bound")
    return {
        "type": "object",
        "properties": {
            "guess": {"type": "integer", "minimum": low, "maximum": high}
        },
        "required": ["guess"],
        "additionalProperties": False,
    }


def main() -> None:
    client = OpenAI(
        base_url=os.environ["FLASH_OPENAI_BASE_URL"],
        api_key=os.environ["FREESOLO_API_KEY"],
    )
    low = 1
    high = 100
    secret = 42
    schema = schema_for_metadata({"low": low, "high": high})
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
                "json_schema": {
                    "name": "number_guess",
                    "strict": True,
                    "schema": schema,
                },
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

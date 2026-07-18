"""Call and evaluate a deployed logic boolean adapter."""

from __future__ import annotations

import os
import re

from openai import OpenAI

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    if "<think>" in text and "</think>" not in text:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    matches = _ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    normalized = matches[-1].strip().casefold()
    if normalized == "true":
        return "True"
    if normalized == "false":
        return "False"
    return None


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
                "content": (
                    "Evaluate the boolean expression. You may reason inside "
                    "<think>...</think>, then end with <answer>True</answer> or "
                    "<answer>False</answer>."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evaluate the following boolean expression:\n\n"
                    "( True and not ( False or False ) )\n\n"
                    "Is the expression True or False?"
                ),
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    predicted = extract_answer(content)
    print(content)
    print(f"predicted={predicted} expected=True")
    if predicted != "True":
        raise ValueError("final boolean answer was incorrect")


if __name__ == "__main__":
    main()

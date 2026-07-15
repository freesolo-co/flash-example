"""Call a deployed JSON extraction adapter and verify its schema."""

from __future__ import annotations

import json
import os

from openai import OpenAI

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["billing", "delivery", "returns", "account", "product"],
        },
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string"},
    },
    "required": ["category", "priority", "summary"],
    "additionalProperties": False,
}


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
                "content": "Extract category, priority, and a short summary as JSON.",
            },
            {"role": "user", "content": "My package arrived with a broken handle."},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "customer_request", "strict": True, "schema": SCHEMA},
        },
    )
    content = response.choices[0].message.content or ""
    parsed = json.loads(content)
    if set(parsed) != {"category", "priority", "summary"}:
        raise ValueError("response did not match the expected fields")
    print(json.dumps(parsed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

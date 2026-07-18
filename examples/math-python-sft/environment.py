"""Multi-turn GRPO environment for tool-using Python math."""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import unicodedata
from fractions import Fraction
from pathlib import Path

from freesolo.datasets import TaskExample
from freesolo.environments import (
    EnvironmentEpisode,
    EnvironmentMultiTurn,
    EnvironmentStepResult,
    RewardResult,
)

SYSTEM_PROMPT = (
    "Use Python for all calculations. Give your answer inside \\boxed{}.\n\n"
    "In addition to the Python standard library, you have access to: math, fractions."
)
_OUTPUT_LIMIT = 4096
_EXECUTION_TIMEOUT_SECONDS = 5
_DATASET_PATH = Path(__file__).parent / "data" / "train.jsonl"
_PYTHON_BLOCK = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_NUMERIC_ANSWER = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?"
)

_PROBLEMS: tuple[tuple[str, str], ...] = (
    ("Compute exactly: 987654321 multiplied by 123456789.", "121932631112635269"),
    ("Compute exactly: 10^20 minus 9876543210.", "99999999990123456790"),
    ("Compute 123456789012345 + 987654321098765.", "1111111110111110"),
    ("Compute exactly: 314159265 multiplied by 271828182.", "85397341863406230"),
    ("Find gcd(1234567890, 9876543210).", "90"),
    ("Find lcm(840, 630).", "2520"),
    ("Find gcd(2^20 - 1, 2^15 - 1).", "31"),
    ("Find lcm(144, 360).", "720"),
    ("Compute 7^222 modulo 1000.", "49"),
    ("Compute 123456789^12345 modulo 1000003.", "11592"),
    ("Compute 2^1000 modulo 123457.", "25251"),
    ("Compute 17^321 modulo 99991.", "48164"),
    ("Compute 15 factorial.", "1307674368000"),
    ("Compute the binomial coefficient C(40, 6).", "3838380"),
    ("How many ordered selections of 5 distinct items can be made from 12 items?", "95040"),
    ("Compute the binomial coefficient C(30, 15).", "155117520"),
    ("Find the sum of the decimal digits of 2^100.", "115"),
    ("Find the sum of the decimal digits of 50 factorial.", "216"),
    ("Find the sum of the decimal digits of 987654321^3.", "126"),
    ("Find the sum of the decimal digits of 12345678901234567890.", "90"),
    ("Compute the sum of all integers from 1 through 1000.", "500500"),
    ("Compute the sum of k^2 for integers k from 1 through 100.", "338350"),
    ("Compute the sum of k^3 for integers k from 1 through 50.", "1625625"),
    ("Compute 1 - 2 + 3 - 4 + ... + 99 - 100.", "-50"),
)


def build_dataset(num_examples: int = 24, max_turns: int = 4) -> list[dict]:
    if not 1 <= num_examples <= len(_PROBLEMS):
        raise ValueError(f"num_examples must be from 1 through {len(_PROBLEMS)}")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    return [
        {
            "id": f"math-python-{index:03d}",
            "input": question,
            "output": f"\\boxed{{{answer}}}",
            "metadata": {"answer": answer, "max_turns": max_turns},
        }
        for index, (question, answer) in enumerate(_PROBLEMS[:num_examples])
    ]


def extract_python_code(text: str) -> str | None:
    blocks = _PYTHON_BLOCK.findall(text)
    return blocks[-1].strip() if blocks else None


def extract_boxed_answer(text: str) -> str | None:
    marker = "\\boxed{"
    answer = None
    search_from = 0
    while True:
        start = text.find(marker, search_from)
        if start < 0:
            return answer
        content_start = start + len(marker)
        depth = 1
        for index in range(content_start, len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    answer = text[content_start:index]
                    search_from = index + 1
                    break
        else:
            return answer


def normalize_answer(value: str) -> tuple[str, str]:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if _NUMERIC_ANSWER.fullmatch(normalized):
        try:
            fraction = Fraction(normalized.replace(" ", ""))
        except (ValueError, ZeroDivisionError):
            pass
        else:
            return "number", f"{fraction.numerator}/{fraction.denominator}"
    return "text", " ".join(normalized.split()).casefold()


def execute_python(code: str) -> str:
    captured = bytearray()
    truncated = False
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="math-python-") as temp_dir:
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=temp_dir,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONIOENCODING": "utf-8",
                },
                start_new_session=True,
            )
            assert process.stdout is not None

            def drain_output() -> None:
                nonlocal truncated
                while chunk := process.stdout.read(8192):
                    remaining = _OUTPUT_LIMIT - len(captured)
                    if remaining > 0:
                        captured.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated = True

            reader = threading.Thread(target=drain_output, daemon=True)
            reader.start()
            try:
                process.wait(timeout=_EXECUTION_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                reader.join(timeout=1)
                if reader.is_alive():
                    process.stdout.close()
                    reader.join(timeout=1)
        except Exception as error:
            return (str(error) or type(error).__name__)[:_OUTPUT_LIMIT].rstrip()

    if timed_out:
        return "execution timed out"
    text = captured.decode("utf-8", errors="replace")
    if not text:
        return "(no output)"
    if truncated:
        marker = "\n[output truncated]"
        text = text[: _OUTPUT_LIMIT - len(marker)] + marker
    return text.rstrip()


def example_metadata(example: TaskExample) -> tuple[str, int]:
    values = example.metadata or {}
    return str(values["answer"]), int(values["max_turns"])


def has_python_tool_result(
    messages: list[dict[str, str]] | tuple[dict[str, str], ...],
) -> bool:
    return any(
        message["role"] == "user"
        and str(message["content"]).startswith("```output\n")
        for message in messages
    )


def load_distilled_dataset(path: str | Path = _DATASET_PATH) -> list[dict]:
    rows = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["output"]["messages"]
            final_assistant = next(
                message for message in reversed(messages) if message["role"] == "assistant"
            )
            answer = extract_boxed_answer(str(final_assistant["content"]))
            if answer is None:
                raise ValueError(f"distilled row {index} has no boxed final answer")
            rows.append(
                {
                    "id": f"math-python-distilled-{index:04d}",
                    "input": row["input"],
                    "output": row["output"],
                    "metadata": {"answer": answer, "max_turns": 4},
                }
            )
    return rows


class MathPythonEnvironment(EnvironmentMultiTurn):
    def __init__(self, dataset_path: str | Path = _DATASET_PATH) -> None:
        self.dataset = load_distilled_dataset(dataset_path)

    def start_episode(
        self, example: TaskExample, prompt_text: str
    ) -> list[dict[str, str]]:
        _ = prompt_text
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(example.input)},
        ]

    def max_episode_turns(self, example: TaskExample) -> int:
        return example_metadata(example)[1]

    def step_episode(
        self,
        example: TaskExample,
        messages: list[dict[str, str]],
        assistant_response: str,
    ) -> EnvironmentStepResult:
        _ = example
        boxed_answer = extract_boxed_answer(assistant_response)
        if boxed_answer is not None and has_python_tool_result(messages):
            return EnvironmentStepResult(done=True, messages=())
        code = extract_python_code(assistant_response)
        if code is None:
            if boxed_answer is not None:
                reply = (
                    "Run at least one fenced ```python block before giving the final answer "
                    "inside \\boxed{}."
                )
            else:
                reply = (
                    "Provide a fenced ```python block or give the final answer inside "
                    "\\boxed{} after using the tool."
                )
        else:
            output = execute_python(code)
            reply = (
                f"```output\n{output}\n```\n"
                "Continue, or give the final answer inside \\boxed{}."
            )
        return EnvironmentStepResult(
            done=False, messages=({"role": "user", "content": reply},)
        )

    def score_episode(
        self, example: TaskExample, episode: EnvironmentEpisode
    ) -> RewardResult:
        expected, _ = example_metadata(example)
        assistant_messages = [
            str(message["content"])
            for message in episode.messages
            if message["role"] == "assistant"
        ]
        predicted = extract_boxed_answer(assistant_messages[-1]) if assistant_messages else None
        tool_used = has_python_tool_result(episode.messages)
        correct = (
            tool_used
            and predicted is not None
            and normalize_answer(predicted) == normalize_answer(expected)
        )
        if correct:
            reason = "final boxed answer matches after python tool execution"
        elif not tool_used:
            reason = "no python tool turn was executed before the final answer"
        elif predicted is None:
            reason = "final assistant message has no complete boxed answer"
        else:
            reason = "final boxed answer is incorrect"
        return RewardResult(score=1.0 if correct else 0.0, success=correct, reason=reason)

    def sft_completion(self, example: TaskExample) -> list[dict[str, str]]:
        output = example.output
        if not isinstance(output, dict) or not isinstance(output.get("messages"), list):
            raise ValueError("math-python SFT rows require output.messages")
        return [dict(message) for message in output["messages"]]


def load_environment(**kwargs: object) -> MathPythonEnvironment:
    dataset_path = kwargs.get("dataset_path", _DATASET_PATH)
    environment = MathPythonEnvironment(dataset_path=str(dataset_path))
    if "num_examples" in kwargs:
        environment.dataset = build_dataset(
            num_examples=int(kwargs["num_examples"]),
            max_turns=int(kwargs.get("max_turns", 4)),
        )
    return environment

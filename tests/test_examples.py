from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

from freesolo.environments import EnvironmentMultiTurn

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "json-extraction-sft",
    "running-total-sft",
    "structured-number-guess-grpo",
    "thinking-math-opd",
    "thinking-science-grpo",
)


def load_environment_module(name: str):
    path = ROOT / "examples" / name / "environment.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(name: str) -> dict:
    path = ROOT / "examples" / name / "train.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_all_environments_load_with_deterministic_datasets() -> None:
    for name in EXAMPLES:
        module = load_environment_module(name)
        first = module.load_environment()
        second = module.load_environment()
        assert len(first.dataset) >= 16
        assert first.dataset == second.dataset


def test_json_extraction_contract() -> None:
    module = load_environment_module("json-extraction-sft")
    row = module.build_dataset()[0]
    assert module.parse_label(row["output"]) == row["metadata"]["expected"]
    assert module.parse_label('{"category":"billing"}') is None


def test_running_total_contract() -> None:
    module = load_environment_module("running-total-sft")
    assert module.running_totals([3, 5, 2]) == [3, 8, 10]
    assert module.gold_completion([3, 5])[-1]["content"] == "8"


def test_structured_number_guess_contract() -> None:
    module = load_environment_module("structured-number-guess-grpo")
    assert module.parse_guess('{"guess":42}') == 42
    assert module.parse_guess('{"guess":true}') is None
    assert module.parse_guess('guess 42') is None


def test_thinking_answer_contracts() -> None:
    math = load_environment_module("thinking-math-opd")
    science = load_environment_module("thinking-science-grpo")
    assert math.extract_answer("work\nAnswer: 24") == 24
    assert math.extract_answer("24") is None
    assert science.extract_answer("work\nAnswer: C") == "C"
    assert science.extract_answer("C") is None


def test_training_matrix_uses_current_bounded_contract() -> None:
    expected = {
        "json-extraction-sft": ("sft", False),
        "running-total-sft": ("sft", False),
        "structured-number-guess-grpo": ("grpo", False),
        "thinking-math-opd": ("opd", True),
        "thinking-science-grpo": ("grpo", True),
    }
    for name, (algorithm, thinking) in expected.items():
        config = load_config(name)
        train = config["train"]
        assert config["algorithm"] == algorithm
        assert config["thinking"] is thinking
        assert train["max_steps"] == 2
        assert train["save_every"] == 1
        assert train["lora_rank"] == 16
        assert config["environment"]["id"].startswith("clay/")

    structured = load_config("structured-number-guess-grpo")["train"]
    assert "structured_outputs" in structured
    assert "structured_outputs" not in load_config("json-extraction-sft")["train"]
    opd = load_config("thinking-math-opd")["train"]
    assert opd["teacher_model"] == "glm-5.2"
    assert opd["max_context_tokens"] > opd["max_completion_tokens"]


class _RolloutExample:
    # minimal stand-in exposing only the attributes environments read while
    # building the opening prompt; enough to catch a str.format brace bug in a
    # prompt template before a paid run reaches it
    def __init__(self, row: dict) -> None:
        self.id = row.get("id")
        self.input = row["input"]
        self.output = row.get("output")
        self.metadata = dict(row.get("metadata") or {})


def test_environments_build_opening_messages_without_format_errors() -> None:
    for name in EXAMPLES:
        module = load_environment_module(name)
        env = module.load_environment()
        example = _RolloutExample(env.dataset[0])
        if isinstance(env, EnvironmentMultiTurn):
            messages = env.start_episode(example, "")
        else:
            messages = env.build_prompt_messages(example, "")
        assert isinstance(messages, list) and messages
        for message in messages:
            assert {"role", "content"} <= set(message)
            assert isinstance(message["content"], str)

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEpisode, EnvironmentMultiTurn

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "running-total-sft",
    "logic-boolean-grpo",
    "structured-number-guess-grpo",
    "thinking-science-grpo",
    "math-boxed-grpo",
    "math-python-grpo",
    "thinking-math-opd",
    "sudoku-grpo",
)
CONFIGS = {
    "running-total-sft": ("train.toml",),
    "logic-boolean-grpo": ("train.toml",),
    "structured-number-guess-grpo": ("train_sft.toml", "train_grpo.toml"),
    "thinking-science-grpo": ("train.toml",),
    "math-boxed-grpo": ("train.toml",),
    "math-python-grpo": ("train.toml",),
    "thinking-math-opd": ("train_sft.toml", "train_opd.toml"),
    "sudoku-grpo": ("train_sft.toml", "train_grpo.toml"),
}


def load_environment_module(name: str):
    path = ROOT / "examples" / name / "environment.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(name: str, filename: str) -> dict:
    path = ROOT / "examples" / name / filename
    with path.open("rb") as handle:
        return tomllib.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_example(row: dict) -> TaskExample:
    return TaskExample(
        record=row,
        id=row.get("id"),
        input=str(row["input"]),
        output=row.get("output"),
        metadata=dict(row.get("metadata") or {}),
    )


def episode(*messages: dict[str, str]) -> EnvironmentEpisode:
    return EnvironmentEpisode(messages=messages, response_text="")


def test_all_environments_load_with_deterministic_datasets() -> None:
    for name in EXAMPLES:
        module = load_environment_module(name)
        first = module.load_environment()
        second = module.load_environment()
        assert len(first.dataset) >= 16
        assert first.dataset == second.dataset


def test_distilled_data_is_present_and_disjoint() -> None:
    expected_train_rows = {
        "running-total-sft": 100,
        "logic-boolean-grpo": 150,
        "structured-number-guess-grpo": 100,
        "thinking-science-grpo": 135,
        "math-boxed-grpo": 143,
        "math-python-grpo": 97,
        "thinking-math-opd": 143,
        "sudoku-grpo": 99,
    }
    for name in EXAMPLES:
        data_dir = ROOT / "examples" / name / "data"
        train = read_jsonl(data_dir / "train.jsonl")
        heldout = json.loads((data_dir / "heldout.json").read_text())
        assert len(train) == expected_train_rows[name]
        assert len(heldout) == 50
        train_inputs = {str(row["input"]) for row in train}
        heldout_inputs = {str(row["input"]) for row in heldout}
        assert train_inputs.isdisjoint(heldout_inputs)


def test_sft_environments_load_the_bundled_data() -> None:
    for name in (
        "running-total-sft",
        "structured-number-guess-grpo",
        "math-boxed-grpo",
        "math-python-grpo",
        "thinking-math-opd",
        "sudoku-grpo",
    ):
        module = load_environment_module(name)
        environment = module.load_environment()
        train_rows = read_jsonl(ROOT / "examples" / name / "data" / "train.jsonl")
        assert len(environment.dataset) == len(train_rows)


def test_shipped_training_matrix() -> None:
    expected = {
        ("running-total-sft", "train.toml"): ("Qwen/Qwen3.5-2B", "sft", 100),
        ("logic-boolean-grpo", "train.toml"): ("Qwen/Qwen3.5-4B", "grpo", 50),
        ("structured-number-guess-grpo", "train_sft.toml"): (
            "Qwen/Qwen3.5-2B",
            "sft",
            75,
        ),
        ("structured-number-guess-grpo", "train_grpo.toml"): (
            "Qwen/Qwen3.5-2B",
            "grpo",
            30,
        ),
        ("thinking-science-grpo", "train.toml"): ("Qwen/Qwen3.5-9B", "opd", 50),
        ("math-boxed-grpo", "train.toml"): ("Qwen/Qwen3.5-9B", "sft", 75),
        ("math-python-grpo", "train.toml"): ("Qwen/Qwen3.5-4B", "sft", 75),
        ("thinking-math-opd", "train_sft.toml"): ("Qwen/Qwen3.5-9B", "sft", 75),
        ("thinking-math-opd", "train_opd.toml"): ("Qwen/Qwen3.5-9B", "opd", 40),
        ("sudoku-grpo", "train_sft.toml"): ("Qwen/Qwen3.5-4B", "sft", 75),
        ("sudoku-grpo", "train_grpo.toml"): ("Qwen/Qwen3.5-4B", "grpo", 20),
    }
    assert {(name, filename) for name, files in CONFIGS.items() for filename in files} == set(
        expected
    )
    for (name, filename), (model, algorithm, max_steps) in expected.items():
        config = load_config(name, filename)
        assert config["model"] == model
        assert config["algorithm"] == algorithm
        assert config["thinking"] is False
        assert config["train"]["max_steps"] == max_steps
        assert config["environment"]["id"].startswith("clay/")


def test_warm_start_configs_inherit_parent_lora_shape() -> None:
    expected_parents = {
        ("structured-number-guess-grpo", "train_grpo.toml"): "flash-1784319868-8f8ee7be",
        ("thinking-math-opd", "train_opd.toml"): "flash-1784325488-7a1d31b9",
        ("sudoku-grpo", "train_grpo.toml"): "flash-1784324917-1d8f0ce8",
    }
    for (name, filename), parent in expected_parents.items():
        train = load_config(name, filename)["train"]
        assert train["init_from_adapter"] == parent
        assert "lora_rank" not in train
        assert "lora_alpha" not in train

    assert load_config("thinking-math-opd", "train_opd.toml")["train"]["teacher_model"] == "kimi-k2.6"
    assert (
        load_config("thinking-science-grpo", "train.toml")["train"]["teacher_model"]
        == "kimi-k2.6"
    )
    assert load_config("thinking-math-opd", "train_opd.toml")["environment"]["params"] == {
        "dataset": "generated"
    }


def test_stage_specific_environment_parameters_select_generated_data() -> None:
    cases = (
        ("running-total-sft", {"num_examples": 3, "seed": 7}),
        (
            "structured-number-guess-grpo",
            {"num_examples": 3, "low": 101, "high": 200, "max_turns": 7},
        ),
        ("math-python-grpo", {"num_examples": 3, "max_turns": 4}),
        (
            "sudoku-grpo",
            {"num_examples": 3, "max_turns": 30, "seed": 42, "difficulty": "easy"},
        ),
    )
    for name, kwargs in cases:
        module = load_environment_module(name)
        assert len(module.load_environment(**kwargs).dataset) == 3

    thinking_math = load_environment_module("thinking-math-opd")
    assert len(thinking_math.load_environment(dataset="generated").dataset) == 24


def test_running_total_lifecycle_and_partial_score() -> None:
    module = load_environment_module("running-total-sft")
    assert module.running_totals([3, 5, 2]) == [3, 8, 10]
    assert module.gold_completion([3, 5])[-1]["content"] == "8"
    env = module.load_environment()
    example = task_example(
        {
            "id": "running-total-test",
            "input": "Numbers: 3 5 2",
            "output": {"messages": module.gold_completion([3, 5, 2])},
            "metadata": {"numbers": [3, 5, 2]},
        }
    )
    opening = env.start_episode(example, "")
    assert opening[-1]["content"] == "Number: 3"
    next_step = env.step_episode(
        example, [*opening, {"role": "assistant", "content": "3"}], "3"
    )
    assert next_step.done is False
    assert next_step.messages[0]["content"] == "Number: 5"
    partial = env.score_episode(
        example,
        episode(
            {"role": "assistant", "content": "3"},
            {"role": "assistant", "content": "8"},
        ),
    )
    complete = env.score_episode(
        example,
        episode(
            {"role": "assistant", "content": "3"},
            {"role": "assistant", "content": "8"},
            {"role": "assistant", "content": "10"},
        ),
    )
    assert partial.score == 2 / 3 and partial.success is False
    assert complete.score == 1.0 and complete.success is True


def test_number_guess_lifecycle_and_reward() -> None:
    module = load_environment_module("structured-number-guess-grpo")
    assert module.parse_guess('{"guess":42}') == 42
    assert module.parse_guess('{"guess":true}') is None
    assert module.parse_guess("guess 42") is None
    env = module.load_environment()
    example = task_example(
        {
            "id": "number-guess-test",
            "input": "Find the secret number from 1 through 100.",
            "output": "42",
            "metadata": {"low": 1, "high": 100, "secret": 42, "max_turns": 7},
        }
    )
    assert env.step_episode(example, [], '{"guess":41}').messages[0]["content"] == "higher"
    assert env.step_episode(example, [], '{"guess":43}').messages[0]["content"] == "lower"
    assert env.step_episode(example, [], '{"guess":42}').done is True
    solved = env.score_episode(
        example, episode({"role": "assistant", "content": '{"guess":42}'})
    )
    assert solved.score == 1.0 and solved.success is True


def test_single_turn_reward_contracts() -> None:
    for name in (
        "logic-boolean-grpo",
        "thinking-science-grpo",
        "math-boxed-grpo",
        "thinking-math-opd",
    ):
        module = load_environment_module(name)
        env = module.load_environment()
        row = env.dataset[0]
        example = task_example(row)
        correct = env.score_response(example, str(row["output"]))
        incorrect = env.score_response(example, "not the expected response")
        assert correct.score == 1.0 and correct.success is True
        assert incorrect.score == 0.0 and incorrect.success is False


def test_thinking_answer_parsers() -> None:
    math = load_environment_module("thinking-math-opd")
    science = load_environment_module("thinking-science-grpo")
    assert math.extract_answer("work\nAnswer: 24") == 24
    assert math.extract_answer("24") is None
    assert science.extract_answer("work\nAnswer: C") == "C"
    assert science.extract_answer("C") is None


def test_math_boxed_contract() -> None:
    module = load_environment_module("math-boxed-grpo")
    assert module.extract_boxed_answer(r"first \boxed{3}, final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert module.answers_equivalent("0.5", "1/2") is True
    env = module.load_environment()
    example = task_example(
        {
            "id": "math-boxed-test",
            "input": "Compute 47 + 38.",
            "output": r"\boxed{85}",
            "metadata": {"answer": "85"},
        }
    )
    assert env.score_response(example, r"work \boxed{85}").success is True
    assert env.score_response(example, r"work \boxed{84}").success is False


def test_logic_boolean_contract() -> None:
    module = load_environment_module("logic-boolean-grpo")
    assert module.extract_answer("<answer>False</answer><answer>True</answer>") == "True"
    assert module.extract_answer("<think>still open") is None
    env = module.load_environment()
    example = task_example(env.dataset[0])
    expected = str(example.metadata["answer"])
    wrong = "False" if expected == "True" else "True"
    assert env.score_response(example, f"<answer>{expected}</answer>").success is True
    assert env.score_response(example, f"<answer>{wrong}</answer>").success is False


def test_math_python_lifecycle_and_tool_output() -> None:
    module = load_environment_module("math-python-grpo")
    env = module.load_environment()
    example = task_example(
        {
            "id": "math-python-test",
            "input": "Compute exactly: 987654321 multiplied by 123456789.",
            "output": r"\boxed{121932631112635269}",
            "metadata": {"answer": "121932631112635269", "max_turns": 4},
        }
    )
    opening = env.start_episode(example, "")
    code_turn = "compute it\n```python\nprint(987654321 * 123456789)\n```"
    tool_step = env.step_episode(example, opening, code_turn)
    assert tool_step.done is False
    assert "121932631112635269" in tool_step.messages[0]["content"]
    final_turn = r"the result is \boxed{121932631112635269}."
    transcript = (
        *opening,
        {"role": "assistant", "content": code_turn},
        tool_step.messages[0],
        {"role": "assistant", "content": final_turn},
    )
    correct = env.score_episode(example, episode(*transcript))
    wrong = env.score_episode(
        example, episode({"role": "assistant", "content": r"\boxed{0}"})
    )
    assert correct.score == 1.0 and correct.success is True
    assert wrong.score == 0.0 and wrong.success is False


def test_sudoku_move_parsing_and_full_solve() -> None:
    module = load_environment_module("sudoku-grpo")
    assert module.parse_move_string(module.extract_move("<think>x</think><move>A1=5</move>")) == (
        0,
        0,
        5,
    )
    assert module.parse_move_string(module.extract_move("A1=5")) is None
    env = module.load_environment()
    example = task_example(env.dataset[0])
    puzzle = example.metadata["puzzle"]
    solution = example.metadata["solution"]
    messages = env.start_episode(example, "")
    empties = [(row, col) for row in range(9) for col in range(9) if puzzle[row][col] == 0]
    for index, (row, col) in enumerate(empties):
        move = module.format_move(row, col, solution[row][col])
        response = f"<think>known value</think><move>{move}</move>"
        step = env.step_episode(example, messages, response)
        messages.append({"role": "assistant", "content": response})
        messages.extend(step.messages)
        assert step.done is (index == len(empties) - 1)
    reward = env.score_episode(example, episode(*messages))
    assert reward.success is True and reward.score > 0


class _RolloutExample:
    # minimal stand-in exposing only the attributes environments read
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

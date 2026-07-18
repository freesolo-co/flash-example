from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

from freesolo.datasets import TaskExample
from freesolo.environments import EnvironmentEpisode, EnvironmentMultiTurn

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "running-total-sft",
    "logic-boolean-sft-grpo",
    "structured-number-guess-sft-grpo",
    "thinking-science-opd",
    "math-boxed-sft",
    "math-python-sft",
    "thinking-math-sft-opd",
    "sudoku-sft-grpo",
)
CONFIGS = {
    "running-total-sft": ("train.toml",),
    "logic-boolean-sft-grpo": ("train_sft.toml", "train_grpo.toml"),
    "structured-number-guess-sft-grpo": ("train_sft.toml", "train_grpo.toml"),
    "thinking-science-opd": ("train.toml",),
    "math-boxed-sft": ("train.toml",),
    "math-python-sft": ("train.toml",),
    "thinking-math-sft-opd": ("train_sft.toml", "train_opd.toml"),
    "sudoku-sft-grpo": ("train_sft.toml", "train_grpo.toml"),
}


def load_python_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_environment_module(name: str):
    path = ROOT / "examples" / name / "environment.py"
    return load_python_module(path, name.replace("-", "_"))


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
        "logic-boolean-sft-grpo": 150,
        "structured-number-guess-sft-grpo": 100,
        "thinking-science-opd": 135,
        "math-boxed-sft": 143,
        "math-python-sft": 97,
        "thinking-math-sft-opd": 143,
        "sudoku-sft-grpo": 99,
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


def test_number_guess_data_has_varied_disjoint_secrets() -> None:
    data_dir = ROOT / "examples" / "structured-number-guess-sft-grpo" / "data"
    train = read_jsonl(data_dir / "train.jsonl")
    heldout = json.loads((data_dir / "heldout.json").read_text())
    train_secrets = {
        json.loads(row["output"]["messages"][-2]["content"])["guess"] for row in train
    }
    heldout_secrets = {row["metadata"]["secret"] for row in heldout}
    offsets = {
        json.loads(row["output"]["messages"][-2]["content"])["guess"]
        - int(str(row["input"]).split(" from ", 1)[1].split(" through ", 1)[0])
        for row in train
    }
    assert len(offsets) > 1
    assert train_secrets.isdisjoint(heldout_secrets)


def test_logic_distilled_data_is_strictly_normalized() -> None:
    module = load_environment_module("logic-boolean-sft-grpo")
    rows = read_jsonl(
        ROOT / "examples" / "logic-boolean-sft-grpo" / "data" / "train.jsonl"
    )
    for row in rows:
        output = str(row["output"])
        answer = module.extract_answer(output)
        assert answer is not None
        assert row["metadata"] == {"answer": answer}
        assert output.count("<answer>") == 1
        assert output.count("</answer>") == 1
        assert output.endswith(f"<answer>{answer}</answer>")
        if output.startswith("<think>"):
            assert output.count("<think>") == 1
            assert output.count("</think>") == 1
            assert "</think>\n<answer>" in output
        else:
            assert output == f"<answer>{answer}</answer>"


def test_sft_environments_load_the_bundled_data() -> None:
    for name in (
        "running-total-sft",
        "logic-boolean-sft-grpo",
        "structured-number-guess-sft-grpo",
        "math-boxed-sft",
        "math-python-sft",
        "thinking-math-sft-opd",
        "sudoku-sft-grpo",
    ):
        module = load_environment_module(name)
        kwargs = {"use_distilled": True} if name == "logic-boolean-sft-grpo" else {}
        environment = module.load_environment(**kwargs)
        train_rows = read_jsonl(ROOT / "examples" / name / "data" / "train.jsonl")
        assert len(environment.dataset) == len(train_rows)


def test_shipped_training_matrix() -> None:
    expected = {
        ("running-total-sft", "train.toml"): ("Qwen/Qwen3.5-2B", "sft", 100),
        ("logic-boolean-sft-grpo", "train_sft.toml"): ("Qwen/Qwen3.5-4B", "sft", 100),
        ("logic-boolean-sft-grpo", "train_grpo.toml"): ("Qwen/Qwen3.5-4B", "grpo", 50),
        ("structured-number-guess-sft-grpo", "train_sft.toml"): (
            "Qwen/Qwen3.5-2B",
            "sft",
            75,
        ),
        ("structured-number-guess-sft-grpo", "train_grpo.toml"): (
            "Qwen/Qwen3.5-2B",
            "grpo",
            30,
        ),
        ("thinking-science-opd", "train.toml"): ("Qwen/Qwen3.5-9B", "opd", 50),
        ("math-boxed-sft", "train.toml"): ("Qwen/Qwen3.5-9B", "sft", 75),
        ("math-python-sft", "train.toml"): ("Qwen/Qwen3.5-4B", "sft", 75),
        ("thinking-math-sft-opd", "train_sft.toml"): ("Qwen/Qwen3.5-9B", "sft", 75),
        ("thinking-math-sft-opd", "train_opd.toml"): ("Qwen/Qwen3.5-9B", "opd", 40),
        ("sudoku-sft-grpo", "train_sft.toml"): ("Qwen/Qwen3.5-4B", "sft", 75),
        ("sudoku-sft-grpo", "train_grpo.toml"): ("Qwen/Qwen3.5-4B", "grpo", 20),
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
        ("logic-boolean-sft-grpo", "train_grpo.toml"): "flash-1784350210-5225a6a4",
        ("structured-number-guess-sft-grpo", "train_grpo.toml"): "flash-1784319868-8f8ee7be",
        ("thinking-math-sft-opd", "train_opd.toml"): "flash-1784325488-7a1d31b9",
        ("sudoku-sft-grpo", "train_grpo.toml"): "flash-1784324917-1d8f0ce8",
    }
    for (name, filename), parent in expected_parents.items():
        train = load_config(name, filename)["train"]
        assert train["init_from_adapter"] == parent
        assert "lora_rank" not in train
        assert "lora_alpha" not in train

    assert load_config("thinking-math-sft-opd", "train_opd.toml")["train"]["teacher_model"] == "kimi-k2.6"
    assert (
        load_config("thinking-science-opd", "train.toml")["train"]["teacher_model"]
        == "kimi-k2.6"
    )
    assert load_config("thinking-math-sft-opd", "train_opd.toml")["environment"]["params"] == {
        "dataset": "generated"
    }
    assert load_config("logic-boolean-sft-grpo", "train_sft.toml")["environment"]["params"] == {
        "use_distilled": True
    }


def test_logic_default_generation_splits_are_disjoint() -> None:
    distill = load_python_module(
        ROOT / "data-generation" / "distill.py", "distill_regression"
    )
    adapter = distill.SingleTurnAdapter("logic-boolean-sft-grpo")
    train, heldout, _ = distill.build_splits(
        "logic-boolean-sft-grpo",
        adapter,
        train_size=150,
        heldout_size=50,
        seed=20260717,
    )
    assert len(train) == 150
    assert len(heldout) == 50
    assert {distill.input_digest(row.input) for row in train}.isdisjoint(
        {distill.input_digest(row.input) for row in heldout}
    )


def test_stage_specific_environment_parameters_select_generated_data() -> None:
    cases = (
        ("running-total-sft", {"num_examples": 3, "seed": 7}),
        (
            "structured-number-guess-sft-grpo",
            {"num_examples": 3, "low": 101, "high": 200, "max_turns": 7},
        ),
        ("math-python-sft", {"num_examples": 3, "max_turns": 4}),
        (
            "sudoku-sft-grpo",
            {"num_examples": 3, "max_turns": 30, "seed": 42, "difficulty": "easy"},
        ),
    )
    for name, kwargs in cases:
        module = load_environment_module(name)
        assert len(module.load_environment(**kwargs).dataset) == 3

    thinking_math = load_environment_module("thinking-math-sft-opd")
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
    module = load_environment_module("structured-number-guess-sft-grpo")
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

    record_only = SimpleNamespace(
        metadata={},
        record={"metadata": {"low": 1, "high": 100, "secret": 42, "max_turns": 7}},
    )
    assert module.metadata(record_only) == (1, 100, 42, 7)


def test_single_turn_reward_contracts() -> None:
    for name in (
        "logic-boolean-sft-grpo",
        "thinking-science-opd",
        "math-boxed-sft",
        "thinking-math-sft-opd",
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
    math = load_environment_module("thinking-math-sft-opd")
    science = load_environment_module("thinking-science-opd")
    assert math.extract_answer("work\nAnswer: 24") == 24
    assert math.extract_answer("24") is None
    assert science.extract_answer("work\nAnswer: C") == "C"
    assert science.extract_answer("C") is None


def test_math_boxed_contract() -> None:
    module = load_environment_module("math-boxed-sft")
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
    module = load_environment_module("logic-boolean-sft-grpo")
    assert module.extract_answer("<answer>True</answer>") == "True"
    assert module.extract_answer("<think>reason</think>\n<answer>False</answer>") == "False"
    assert module.extract_answer("\n<answer>True</answer>\n") == "True"
    assert module.extract_answer("<answer>False</answer><answer>True</answer>") is None
    assert module.extract_answer("<answer>True</answer> trailing") is None
    assert module.extract_answer("prefix <answer>True</answer>") is None
    assert module.extract_answer("<think>still open") is None
    env = module.load_environment()
    example = task_example(env.dataset[0])
    expected = str(example.metadata["answer"])
    wrong = "False" if expected == "True" else "True"
    assert env.score_response(example, f"<answer>{expected}</answer>").success is True
    assert env.score_response(example, f"<answer>{wrong}</answer>").success is False


def test_math_python_lifecycle_and_tool_output() -> None:
    module = load_environment_module("math-python-sft")
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
    final_turn = r"the result is \boxed{121932631112635269}."
    premature_messages = [
        *opening,
        {"role": "assistant", "content": final_turn},
    ]
    premature_step = env.step_episode(example, premature_messages, final_turn)
    assert premature_step.done is False
    assert "Run at least one fenced ```python block" in premature_step.messages[0]["content"]
    premature_reward = env.score_episode(example, episode(*premature_messages))
    assert premature_reward.success is False
    assert premature_reward.reason == "no python tool turn was executed before the final answer"

    code_turn = "compute it\n```python\nprint(987654321 * 123456789)\n```"
    tool_step = env.step_episode(example, opening, code_turn)
    assert tool_step.done is False
    assert "121932631112635269" in tool_step.messages[0]["content"]
    transcript = (
        *opening,
        {"role": "assistant", "content": code_turn},
        tool_step.messages[0],
        {"role": "assistant", "content": final_turn},
    )
    final_step = env.step_episode(example, list(transcript), final_turn)
    assert final_step.done is True
    correct = env.score_episode(example, episode(*transcript))
    wrong = env.score_episode(
        example,
        episode(
            {"role": "assistant", "content": code_turn},
            tool_step.messages[0],
            {"role": "assistant", "content": r"\boxed{0}"},
        ),
    )
    assert correct.score == 1.0 and correct.success is True
    assert wrong.score == 0.0 and wrong.success is False


def test_math_python_execution_is_bounded(monkeypatch) -> None:
    module = load_environment_module("math-python-sft")
    output = module.execute_python("print('x' * 10000)")
    assert len(output) <= module._OUTPUT_LIMIT
    assert output.endswith("[output truncated]")

    monkeypatch.setattr(module, "_EXECUTION_TIMEOUT_SECONDS", 0.05)
    assert module.execute_python("import time; time.sleep(10)") == "execution timed out"


def test_eval_requires_math_python_unsafe_opt_in_and_uses_case_bounds(
    monkeypatch, tmp_path
) -> None:
    evaluate = load_python_module(
        ROOT / "eval" / "evaluate_suite.py", "evaluate_suite_regression"
    )
    evaluate.require_local_execution_opt_in(
        "math-python-sft", dry_run=True, allowed=False
    )
    try:
        evaluate.require_local_execution_opt_in(
            "math-python-sft", dry_run=False, allowed=False
        )
    except RuntimeError as error:
        assert "--allow-unsafe-local-code-execution" in str(error)
    else:
        raise AssertionError("math-python evaluation did not require unsafe opt-in")

    expected_adapter_models = {
        "running-total-sft": "flash-1784320041-9c4a32b8",
        "logic-boolean-sft-grpo": "flash-1784350975-6dc07970",
        "structured-number-guess-sft-grpo": "flash-1784349640-ceb35b18",
        "thinking-science-opd": "flash-1784321193-dab97aef",
        "math-boxed-sft": "flash-1784263689-f98515ce",
        "math-python-sft": "flash-1784322317-e152ffdd",
        "thinking-math-sft-opd": "flash-1784326094-ab33c65b",
        "sudoku-sft-grpo": "flash-1784327754-e1d3a602",
    }
    assert {
        name: profile.adapter_model for name, profile in evaluate.PROFILES.items()
    } == expected_adapter_models

    loaded = evaluate.load_example(evaluate.PROFILES["structured-number-guess-sft-grpo"])
    row = loaded.rows[0]
    schema = evaluate.response_schema_for_row(loaded, row)
    assert schema is not None
    guess_schema = schema["properties"]["guess"]
    assert guess_schema["minimum"] == row["metadata"]["low"]
    assert guess_schema["maximum"] == row["metadata"]["high"]

    logic = evaluate.load_example(evaluate.PROFILES["logic-boolean-sft-grpo"])
    requests = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<answer>True"),
                    finish_reason="stop",
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    completion = evaluate.request_completion(
        client, logic, "test-model", [], logic.rows[0]
    )
    assert requests[0]["stop"] == ["</answer>"]
    assert completion.content == "<answer>True</answer>"

    def create_truncated(**request):
        _ = request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<answer>True"),
                    finish_reason="length",
                )
            ]
        )

    truncated_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_truncated))
    )
    truncated = evaluate.request_completion(
        truncated_client, logic, "test-model", [], logic.rows[0]
    )
    assert truncated.content == "<answer>True"
    assert truncated.finish_reason == "length"

    sys.modules["evaluate_suite"] = evaluate
    evaluate_gpt55 = load_python_module(
        ROOT / "eval" / "evaluate_gpt55.py", "evaluate_gpt55_regression"
    )
    monkeypatch.setattr(
        evaluate_gpt55,
        "parse_args",
        lambda: SimpleNamespace(
            example="math-python-sft",
            workers=1,
            base_url="http://unused.invalid/v1",
            output=tmp_path / "unsafe.json",
            dry_run=False,
            allow_unsafe_local_code_execution=False,
        ),
    )
    try:
        evaluate_gpt55.main()
    except RuntimeError as error:
        assert "--allow-unsafe-local-code-execution" in str(error)
    else:
        raise AssertionError("gpt-5.5 math-python evaluation bypassed unsafe opt-in")

    monkeypatch_sleep = evaluate_gpt55.time.sleep
    evaluate_gpt55.time.sleep = lambda _: None

    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)
            self.is_error = status_code >= 400

        def json(self):
            return self._payload

    class HttpClient:
        def __init__(self):
            self.calls = 0

        def post(self, path, json):
            _ = path, json
            self.calls += 1
            if self.calls == 1:
                return Response(429, {"detail": "rate limited"})
            return Response(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

    try:
        http_client = HttpClient()
        gateway = evaluate_gpt55.GatewayCaseClient(http_client)
        response = gateway.create(messages=[], model="test", max_tokens=1)
        assert http_client.calls == 2
        assert response.choices[0].message.content == "ok"
    finally:
        evaluate_gpt55.time.sleep = monkeypatch_sleep


def test_multiturn_distillation_requires_unsafe_opt_in_and_rejects_invalid_sudoku_moves() -> None:
    distill = load_python_module(
        ROOT / "data-generation" / "distill.py", "distill_multiturn_dependency"
    )
    sys.modules["distill"] = distill
    multiturn = load_python_module(
        ROOT / "data-generation" / "distill_multiturn.py",
        "distill_multiturn_regression",
    )
    try:
        multiturn.run(
            SimpleNamespace(
                task="math-python-sft",
                allow_unsafe_local_code_execution=False,
            )
        )
    except RuntimeError as error:
        assert "--allow-unsafe-local-code-execution" in str(error)
    else:
        raise AssertionError("math-python distillation bypassed unsafe opt-in")

    adapter = multiturn.SudokuAdapter()
    problem = adapter.build_splits(1, 1, 20260717)[0][0]
    puzzle = problem.row["metadata"]["puzzle"]
    clue_row, clue_col = next(
        (row, col)
        for row in range(9)
        for col in range(9)
        if puzzle[row][col] != 0
    )
    invalid_clear = adapter.module.format_move(clue_row, clue_col, 0)
    valid, reason = adapter.validate_protocol(
        problem,
        [{"role": "assistant", "content": f"<move>{invalid_clear}</move>"}],
        SimpleNamespace(success=True, reason="solved"),
    )
    assert valid is False
    assert reason == "assistant turn contained a move rejected by the environment"


def test_committed_sudoku_transcripts_replay_only_valid_moves() -> None:
    module = load_environment_module("sudoku-sft-grpo")
    source = {
        str(row["input"]): row
        for row in module.build_dataset(
            num_examples=100,
            max_turns=30,
            seed=20260717,
            difficulty="easy",
        )
    }
    rows = read_jsonl(
        ROOT / "examples" / "sudoku-sft-grpo" / "data" / "train.jsonl"
    )
    for row in rows:
        board = module.SudokuBoard(source[str(row["input"])]["metadata"]["puzzle"])
        for message in row["output"]["messages"]:
            if message["role"] != "assistant":
                continue
            move = module.parse_move_string(module.extract_move(str(message["content"])))
            assert move is not None
            assert board.make_move(*move) is True


def test_sudoku_move_parsing_and_full_solve() -> None:
    module = load_environment_module("sudoku-sft-grpo")
    assert module.parse_move_string(module.extract_move("<think>x</think><move>A1=5</move>")) == (
        0,
        0,
        5,
    )
    assert module.extract_move("<move>A1=5</move><move>A2=6</move>") == ""
    assert module.extract_move("<move>A1=5</move> trailing") == ""
    assert module.parse_move_string(module.extract_move("A1=5")) is None

    board = module.SudokuBoard([[5, *([0] * 8)], *[[0] * 9 for _ in range(8)]])
    assert board.make_move(0, 0, 0) is False
    assert board.make_move(0, 0, 4) is False
    assert board.board[0][0] == 5
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


def test_sudoku_reward_requires_metadata_solution() -> None:
    module = load_environment_module("sudoku-sft-grpo")
    env = module.load_environment()
    completed_board = [
        [((row * 3 + row // 3 + col) % 9) + 1 for col in range(9)]
        for row in range(9)
    ]
    different_solution = [
        [((value % 9) + 1) for value in row] for row in completed_board
    ]
    example = task_example(
        {
            "id": "sudoku-unique-solution-test",
            "input": "synthetic puzzle",
            "output": "",
            "metadata": {
                "puzzle": [[0] * 9 for _ in range(9)],
                "solution": different_solution,
                "max_turns": 81,
            },
        }
    )
    messages = tuple(
        {
            "role": "assistant",
            "content": f"<move>{module.format_move(row, col, value)}</move>",
        }
        for row, values in enumerate(completed_board)
        for col, value in enumerate(values)
    )
    reward = env.score_episode(example, episode(*messages))
    assert module._replay_board(example, messages).is_solved() is True
    assert reward.success is False
    assert reward.score < 5.0


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
